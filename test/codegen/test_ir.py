"""Tests for core IR types in nkigym.codegen.ir."""

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.dep_cache import DepCache
from nkigym.codegen.ir import (
    BodyLeaf,
    DimInfo,
    KernelModule,
    LoopNode,
    Tensor,
    leaves_under,
    replace_at_path,
    resolve_node,
    validate_dataflow_ordering,
)
from nkigym.ops import nkigym_kernel
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore


def test_tensor_auto_populates_buffer_degree():
    t = Tensor(name="x", dim_ids=("d0", "d1"), shape=(128, 256), dtype="bfloat16", origin="intermediate")
    assert t.buffer_degree == {"d0": 1, "d1": 1}


def test_loop_node_defaults():
    n = LoopNode(dim_id="d0", trip_count=16, role=AxisRole.PARALLEL)
    assert n.children == []
    assert n.reduce_op is None
    assert n.name is None
    assert n.pipeline_depth == 1


def test_body_leaf_self_describing_fields():
    leaf = BodyLeaf(
        op_cls=object,
        phase="main",
        reads={"data": "x"},
        writes=("y",),
        kwargs={"op": "square"},
        axis_map={"P": "d0", "F": "d1"},
        dim_role={"d0": AxisRole.PARALLEL, "d1": AxisRole.PARALLEL},
        op_local_buffers={},
    )
    assert leaf.reads == {"data": "x"}
    assert leaf.writes == ("y",)
    assert leaf.kwargs == {"op": "square"}


def test_kernel_module_construction():
    km = KernelModule(
        func_name="f",
        param_names=["x"],
        return_name="y",
        tensors={"x": Tensor("x", ("d0",), (128,), "bfloat16", "param")},
        dims={"d0": DimInfo(dim_id="d0", total_size=128, tile_size=128, num_tiles=1)},
        body=[],
    )
    assert km.func_name == "f"
    assert km.body == []


def test_resolve_node_returns_leaf():
    leaf = BodyLeaf(op_cls=object, phase="main")
    forest = [leaf]
    assert resolve_node(forest, (0,)) is leaf


def test_resolve_node_returns_nested_loop():
    inner = LoopNode("d0", 4, AxisRole.PARALLEL)
    outer = LoopNode("d0", 1, AxisRole.PARALLEL, children=[inner])
    forest = [outer]
    assert resolve_node(forest, (0, 0)) is inner


def test_resolve_node_returns_none_on_bad_path():
    assert resolve_node([], (0,)) is None
    leaf = BodyLeaf(op_cls=object, phase="main")
    assert resolve_node([leaf], (0, 0)) is None


def test_replace_at_path_replaces_target():
    a = BodyLeaf(op_cls=object, phase="main")
    b = BodyLeaf(op_cls=object, phase="other")
    forest = [a, a]
    new_forest = replace_at_path(forest, (1,), b)
    assert new_forest[0] is a
    assert new_forest[1] is b
    assert forest[1] is a


def test_leaves_under_returns_all_leaves():
    a = BodyLeaf(op_cls=object, phase="a")
    b = BodyLeaf(op_cls=object, phase="b")
    loop = LoopNode("d0", 4, AxisRole.PARALLEL, children=[a, b])
    assert list(leaves_under(loop)) == [a, b]


@nkigym_kernel
def _matmul_lhsT_rhs_fixture(lhs_T, rhs):
    """Canonical lhs_T.T @ rhs kernel used by the dataflow validator test."""
    lhs_T_sbuf = NKILoad()(data=lhs_T)
    rhs_sbuf = NKILoad()(data=rhs)
    prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


def test_validate_dataflow_ordering_accepts_canonical():
    """Canonical matmul module has every read written by an earlier leaf (or a param)."""
    specs = {"lhs_T": {"shape": (256, 128), "dtype": "bfloat16"}, "rhs": {"shape": (256, 512), "dtype": "bfloat16"}}
    module = build_canonical_module(_matmul_lhsT_rhs_fixture, specs)
    assert validate_dataflow_ordering(module)


def test_validate_dataflow_ordering_rejects_out_of_order():
    """A forest where the consumer appears before its producer is rejected."""
    producer = BodyLeaf(op_cls=object, phase="main", reads={}, writes=("intermediate",))
    consumer = BodyLeaf(op_cls=object, phase="main", reads={"x": "intermediate"}, writes=("out",))
    module = KernelModule(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors={
            "intermediate": Tensor("intermediate", (), (), "bfloat16", "intermediate"),
            "out": Tensor("out", (), (), "bfloat16", "return"),
        },
        dims={},
        body=[consumer, producer],
        dep=DepCache(scopes={}),
    )
    assert not validate_dataflow_ordering(module)


def test_validate_dataflow_ordering_accepts_param_reads():
    """Reading a param tensor never requires a writer."""
    module = KernelModule(
        func_name="f",
        param_names=["x"],
        return_name="out",
        tensors={"x": Tensor("x", (), (), "bfloat16", "param"), "out": Tensor("out", (), (), "bfloat16", "return")},
        dims={},
        body=[BodyLeaf(op_cls=object, phase="main", reads={"data": "x"}, writes=("out",))],
        dep=DepCache(scopes={}),
    )
    assert validate_dataflow_ordering(module)


def test_validate_dataflow_ordering_rejects_missing_return_writer():
    """A return tensor with no writer (e.g. Store leaf dropped) is rejected."""
    producer = BodyLeaf(op_cls=object, phase="main", reads={}, writes=("intermediate",))
    module = KernelModule(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors={
            "intermediate": Tensor("intermediate", (), (), "bfloat16", "intermediate"),
            "out": Tensor("out", (), (), "bfloat16", "return"),
        },
        dims={},
        body=[producer],
        dep=DepCache(scopes={}),
    )
    assert not validate_dataflow_ordering(module)


def test_validate_dataflow_ordering_rejects_matmul_phase_swap():
    """A forest where compute precedes psum_init for the same matmul is rejected."""
    reads = {"stationary": "lhs_sbuf", "moving": "rhs_sbuf"}
    writes = ("prod",)
    psum_init = BodyLeaf(op_cls=NKIMatmul, phase="psum_init", reads=dict(reads), writes=writes, axis_map={"M": "d0"})
    compute = BodyLeaf(op_cls=NKIMatmul, phase="compute", reads=dict(reads), writes=writes, axis_map={"M": "d0"})
    drain = BodyLeaf(op_cls=NKIMatmul, phase="drain", reads=dict(reads), writes=writes, axis_map={"M": "d0"})
    module = KernelModule(
        func_name="f",
        param_names=["lhs_sbuf", "rhs_sbuf"],
        return_name="prod",
        tensors={
            "lhs_sbuf": Tensor("lhs_sbuf", (), (), "bfloat16", "param"),
            "rhs_sbuf": Tensor("rhs_sbuf", (), (), "bfloat16", "param"),
            "prod": Tensor("prod", (), (), "bfloat16", "return"),
        },
        dims={},
        body=[compute, psum_init, drain],
        dep=DepCache(scopes={}),
    )
    assert not validate_dataflow_ordering(module)
    good = KernelModule(
        func_name="f",
        param_names=module.param_names,
        return_name=module.return_name,
        tensors=module.tensors,
        dims={},
        body=[psum_init, compute, drain],
        dep=DepCache(scopes={}),
    )
    assert validate_dataflow_ordering(good)


def test_validate_dataflow_ordering_traverses_loops():
    """Pre-order DFS descends into ``LoopNode`` children; deeply nested leaves are reached."""
    producer = BodyLeaf(op_cls=object, phase="main", reads={}, writes=("t",))
    consumer = BodyLeaf(op_cls=object, phase="main", reads={"x": "t"}, writes=("out",))
    outer = LoopNode("d0", 1, AxisRole.PARALLEL, children=[producer])
    inner = LoopNode("d0", 1, AxisRole.PARALLEL, children=[consumer])
    module = KernelModule(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors={
            "t": Tensor("t", (), (), "bfloat16", "intermediate"),
            "out": Tensor("out", (), (), "bfloat16", "return"),
        },
        dims={},
        body=[outer, inner],
        dep=DepCache(scopes={}),
    )
    assert validate_dataflow_ordering(module)
    swapped = KernelModule(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors=module.tensors,
        dims={},
        body=[inner, outer],
        dep=DepCache(scopes={}),
    )
    assert not validate_dataflow_ordering(swapped)
