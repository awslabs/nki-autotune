"""Unit tests for the op-level dependency graph."""

import pytest

from nkigym.codegen.dep_graph import DepGraph


def test_dep_graph_is_frozen_dataclass_with_expected_fields() -> None:
    """DepGraph exposes producer, consumers, reads, writes as the four persisted maps."""
    dep = DepGraph(producer={}, consumers={}, reads={}, writes={})
    assert dep.producer == {}
    assert dep.consumers == {}
    assert dep.reads == {}
    assert dep.writes == {}


from nkigym.codegen.dep_graph import build_dep_graph
from nkigym.codegen.graph import ParsedOp, parse_and_resolve
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


@nkigym_kernel
def _two_op_chain(lhs):
    lhs_sbuf = NKILoad()(data=lhs)
    out = NKIStore()(data=lhs_sbuf)
    return out


@nkigym_kernel
def _rmsnorm_matmul(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=1e-6)(data=sum_sq)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


_CHAIN_SPECS = {"lhs": ((128, 256), "bfloat16")}
_RMSNORM_SPECS = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}


def test_build_dep_graph_two_op_chain_producer_consumers() -> None:
    """load -> store: load writes lhs_sbuf, store reads it."""
    g = parse_and_resolve(_two_op_chain, _CHAIN_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    assert dep.producer["lhs"] is None
    assert dep.producer["lhs_sbuf"] == 0
    assert dep.producer["out"] == 1
    assert dep.consumers["lhs"] == (0,)
    assert dep.consumers["lhs_sbuf"] == (1,)
    assert dep.consumers["out"] == ()


def test_build_dep_graph_two_op_chain_reads_writes() -> None:
    """Op-level reads/writes use OPERAND_AXES (reads) and output_names (writes)."""
    g = parse_and_resolve(_two_op_chain, _CHAIN_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    assert dep.reads[0] == frozenset({"lhs"})
    assert dep.writes[0] == frozenset({"lhs_sbuf"})
    assert dep.reads[1] == frozenset({"lhs_sbuf"})
    assert dep.writes[1] == frozenset({"out"})


def test_build_dep_graph_rmsnorm_matmul_shared_producer_has_multiple_consumers() -> None:
    """sbuf_lhs (the Load's output) is consumed by both ActivationReduce and TensorScalar."""
    g = parse_and_resolve(_rmsnorm_matmul, _RMSNORM_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    """Op indices: 0=Load(lhs), 1=Load(rhs), 2=ActivationReduce,
       3=Activation, 4=TensorScalar, 5=Transpose, 6=Matmul, 7=Store."""
    assert dep.producer["lhs_sbuf"] == 0
    assert dep.consumers["lhs_sbuf"] == (2, 4)


def test_build_dep_graph_param_tensors_have_none_producer() -> None:
    """Parameter tensors never have a producer; they appear with producer=None."""
    g = parse_and_resolve(_rmsnorm_matmul, _RMSNORM_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    assert dep.producer["lhs"] is None
    assert dep.producer["rhs"] is None


def test_build_dep_graph_every_tensor_has_producer_key() -> None:
    """Every tensor in op_graph.tensors appears as a producer key."""
    g = parse_and_resolve(_rmsnorm_matmul, _RMSNORM_SPECS)
    dep = build_dep_graph(g.ops, g.tensors)
    for name in g.tensors:
        assert name in dep.producer


def test_build_dep_graph_rejects_duplicate_writes() -> None:
    """Two ops writing the same tensor name raise ValueError."""

    class _FakeOpCls:
        OPERAND_AXES = {"data": ("P",)}
        OUTPUT_AXES = {"output": ("P",)}

    op_a = ParsedOp(
        idx=0,
        op_cls=_FakeOpCls,
        operand_names={"data": "x"},
        op_kwargs={},
        output_names=["y"],
        axis_map={"P": "d0"},
        touched_dims=("d0",),
        dim_role={},
    )
    op_b = ParsedOp(
        idx=1,
        op_cls=_FakeOpCls,
        operand_names={"data": "x"},
        op_kwargs={},
        output_names=["y"],
        axis_map={"P": "d0"},
        touched_dims=("d0",),
        dim_role={},
    )
    with pytest.raises(ValueError, match="duplicate write"):
        build_dep_graph([op_a, op_b], tensors={"x": None, "y": None})


from nkigym.codegen.dep_graph import commutes, subtree_ops, subtree_reads, subtree_writes
from nkigym.codegen.loop_forest import BodyLeaf, LoopNode
from nkigym.ops.base import AxisRole


def _leaf(op_idx: int) -> BodyLeaf:
    return BodyLeaf(op_idx=op_idx)


def _singleton_loop(op_idx: int) -> LoopNode:
    return LoopNode(dim_id="d0", trip_count=1, role=AxisRole.PARALLEL, children=[_leaf(op_idx)])


def _dep_two_ops(r0: set[str], w0: set[str], r1: set[str], w1: set[str]) -> DepGraph:
    """Build a DepGraph for two ops with hand-picked read/write sets."""
    producer: dict[str, int | None] = {}
    for t in w0:
        producer[t] = 0
    for t in w1:
        producer[t] = 1
    return DepGraph(
        producer=producer,
        consumers={},
        reads={0: frozenset(r0), 1: frozenset(r1)},
        writes={0: frozenset(w0), 1: frozenset(w1)},
    )


def test_subtree_ops_collects_body_leaves() -> None:
    """subtree_ops walks to every BodyLeaf under the given subtree."""
    node = LoopNode(
        dim_id="d0",
        trip_count=4,
        role=AxisRole.PARALLEL,
        children=[_leaf(7), LoopNode(dim_id="d1", trip_count=2, role=AxisRole.PARALLEL, children=[_leaf(9)])],
    )
    assert subtree_ops(node) == frozenset({7, 9})


def test_subtree_ops_on_leaf_returns_singleton() -> None:
    """Passing a BodyLeaf directly returns just its op_idx."""
    assert subtree_ops(_leaf(3)) == frozenset({3})


def test_subtree_reads_writes_unions_per_op_edges() -> None:
    """subtree_reads/subtree_writes union per-op maps over all leaves."""
    dep = _dep_two_ops(r0={"a"}, w0={"b"}, r1={"c"}, w1={"d"})
    node = LoopNode(dim_id="d0", trip_count=4, role=AxisRole.PARALLEL, children=[_leaf(0), _leaf(1)])
    assert subtree_reads(node, dep) == frozenset({"a", "c"})
    assert subtree_writes(node, dep) == frozenset({"b", "d"})


def test_commutes_accepts_disjoint_pair() -> None:
    """Two subtrees touching disjoint tensor sets commute."""
    dep = _dep_two_ops(r0={"a"}, w0={"b"}, r1={"c"}, w1={"d"})
    assert commutes(_singleton_loop(0), _singleton_loop(1), dep) is True


def test_commutes_rejects_raw_edge() -> None:
    """RAW: a writes X, b reads X — does not commute."""
    dep = _dep_two_ops(r0=set(), w0={"x"}, r1={"x"}, w1=set())
    assert commutes(_singleton_loop(0), _singleton_loop(1), dep) is False


def test_commutes_rejects_war_edge() -> None:
    """WAR: a reads X, b writes X — does not commute."""
    dep = _dep_two_ops(r0={"x"}, w0=set(), r1=set(), w1={"x"})
    assert commutes(_singleton_loop(0), _singleton_loop(1), dep) is False


def test_commutes_rejects_waw_edge() -> None:
    """WAW: a writes X, b writes X — does not commute.

    Note this case cannot arise from a real OpGraph (SSA forbids
    duplicate writes, which ``build_dep_graph`` enforces), but the
    subtree ``commutes`` predicate must still reject it for safety
    under hand-built test forests and future rewrites that might
    introduce mutation.
    """
    dep = _dep_two_ops(r0=set(), w0={"x"}, r1=set(), w1={"x"})
    assert commutes(_singleton_loop(0), _singleton_loop(1), dep) is False
