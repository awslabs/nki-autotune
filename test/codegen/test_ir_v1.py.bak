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
    t = Tensor(
        name="x", dim_ids=("d0", "d1"), shape=(128, 256), dtype="bfloat16", origin="intermediate", location="sbuf"
    )
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
        reads={"data": "x"},
        writes=("y",),
        kwargs={"op": "square"},
        axis_map={"P": "d0", "F": "d1"},
        dim_role={"d0": AxisRole.PARALLEL, "d1": AxisRole.PARALLEL},
    )
    assert leaf.reads == {"data": "x"}
    assert leaf.writes == ("y",)
    assert leaf.kwargs == {"op": "square"}


def test_kernel_module_construction():
    km = KernelModule(
        func_name="f",
        param_names=["x"],
        return_name="y",
        tensors={"x": Tensor("x", ("d0",), (128,), "bfloat16", "param", location="hbm")},
        dims={"d0": DimInfo(dim_id="d0", total_size=128, tile_size=128, num_tiles=1)},
        body=[],
    )
    assert km.func_name == "f"
    assert km.body == []


def test_resolve_node_returns_leaf():
    leaf = BodyLeaf(op_cls=object)
    forest = [leaf]
    assert resolve_node(forest, (0,)) is leaf


def test_resolve_node_returns_nested_loop():
    inner = LoopNode("d0", 4, AxisRole.PARALLEL)
    outer = LoopNode("d0", 1, AxisRole.PARALLEL, children=[inner])
    forest = [outer]
    assert resolve_node(forest, (0, 0)) is inner


def test_resolve_node_returns_none_on_bad_path():
    assert resolve_node([], (0,)) is None
    leaf = BodyLeaf(op_cls=object)
    assert resolve_node([leaf], (0, 0)) is None


def test_replace_at_path_replaces_target():
    a = BodyLeaf(op_cls=object)
    b = BodyLeaf(op_cls=object)
    forest = [a, a]
    new_forest = replace_at_path(forest, (1,), b)
    assert new_forest[0] is a
    assert new_forest[1] is b
    assert forest[1] is a


def test_leaves_under_returns_all_leaves():
    a = BodyLeaf(op_cls=object)
    b = BodyLeaf(op_cls=object)
    loop = LoopNode("d0", 4, AxisRole.PARALLEL, children=[a, b])
    assert list(leaves_under(loop)) == [a, b]


from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul_lhsT_rhs_fixture(lhs_T, rhs):
    """Canonical lhs_T.T @ rhs kernel used by the dataflow validator test."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(256, 128), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(256, 512), dtype="bfloat16")()
    psum_prod = NKIAlloc(location="psum", shape=(128, 512), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
    out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_prod)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_prod)
    NKITensorCopy()(src=psum_prod, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=out)
    return out


def test_validate_dataflow_ordering_accepts_canonical():
    """Canonical matmul module has every read written by an earlier leaf (or a param)."""
    specs = {"lhs_T": {"shape": (256, 128), "dtype": "bfloat16"}, "rhs": {"shape": (256, 512), "dtype": "bfloat16"}}
    module = build_canonical_module(_matmul_lhsT_rhs_fixture, specs)
    assert validate_dataflow_ordering(module)


def test_validate_dataflow_ordering_rejects_out_of_order():
    """A forest where the consumer appears before its producer is rejected."""

    _FakeAlloc = type("NKIAlloc", (), {})

    alloc_intermediate = BodyLeaf(op_cls=_FakeAlloc, reads={}, writes=("intermediate",))
    alloc_out = BodyLeaf(op_cls=_FakeAlloc, reads={}, writes=("out",))
    producer = BodyLeaf(op_cls=object, reads={}, writes=("intermediate",))
    consumer = BodyLeaf(op_cls=object, reads={"x": "intermediate"}, writes=("out",))
    module = KernelModule(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors={
            "intermediate": Tensor("intermediate", (), (), "bfloat16", "intermediate", "sbuf"),
            "out": Tensor("out", (), (), "bfloat16", "return", "hbm"),
        },
        dims={},
        body=[alloc_intermediate, alloc_out, consumer, producer],
        dep=DepCache(scopes={}),
    )
    assert not validate_dataflow_ordering(module)


def test_validate_dataflow_ordering_accepts_param_reads():
    """Reading a param tensor never requires a writer."""

    _FakeAlloc = type("NKIAlloc", (), {})

    alloc_out = BodyLeaf(op_cls=_FakeAlloc, reads={}, writes=("out",))
    module = KernelModule(
        func_name="f",
        param_names=["x"],
        return_name="out",
        tensors={
            "x": Tensor("x", (), (), "bfloat16", "param", "hbm"),
            "out": Tensor("out", (), (), "bfloat16", "return", "hbm"),
        },
        dims={},
        body=[alloc_out, BodyLeaf(op_cls=object, reads={"data": "x"}, writes=("out",))],
        dep=DepCache(scopes={}),
    )
    assert validate_dataflow_ordering(module)


def test_validate_dataflow_ordering_rejects_missing_return_writer():
    """A return tensor with no writer (e.g. Store leaf dropped) is rejected."""

    _FakeAlloc = type("NKIAlloc", (), {})

    alloc_intermediate = BodyLeaf(op_cls=_FakeAlloc, reads={}, writes=("intermediate",))
    alloc_out = BodyLeaf(op_cls=_FakeAlloc, reads={}, writes=("out",))
    producer = BodyLeaf(op_cls=object, reads={}, writes=("intermediate",))
    module = KernelModule(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors={
            "intermediate": Tensor("intermediate", (), (), "bfloat16", "intermediate", "sbuf"),
            "out": Tensor("out", (), (), "bfloat16", "return", "hbm"),
        },
        dims={},
        body=[alloc_intermediate, alloc_out, producer],
        dep=DepCache(scopes={}),
    )
    assert not validate_dataflow_ordering(module)


def test_validate_dataflow_ordering_traverses_loops():
    """Pre-order DFS descends into ``LoopNode`` children; deeply nested leaves are reached."""

    _FakeAlloc = type("NKIAlloc", (), {})

    alloc_t = BodyLeaf(op_cls=_FakeAlloc, reads={}, writes=("t",))
    alloc_out = BodyLeaf(op_cls=_FakeAlloc, reads={}, writes=("out",))
    producer = BodyLeaf(op_cls=object, reads={}, writes=("t",))
    consumer = BodyLeaf(op_cls=object, reads={"x": "t"}, writes=("out",))
    outer = LoopNode("d0", 1, AxisRole.PARALLEL, children=[producer])
    inner = LoopNode("d0", 1, AxisRole.PARALLEL, children=[consumer])
    module = KernelModule(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors={
            "t": Tensor("t", (), (), "bfloat16", "intermediate", "sbuf"),
            "out": Tensor("out", (), (), "bfloat16", "return", "hbm"),
        },
        dims={},
        body=[alloc_t, alloc_out, outer, inner],
        dep=DepCache(scopes={}),
    )
    assert validate_dataflow_ordering(module)
    swapped = KernelModule(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors=module.tensors,
        dims={},
        body=[alloc_t, alloc_out, inner, outer],
        dep=DepCache(scopes={}),
    )
    assert not validate_dataflow_ordering(swapped)


def test_validate_dataflow_ordering_rmw_requires_prior_writer():
    """An RMW operand in reads_writes must have a prior writer.
    Models the bug scenario: NKIMatmul reads_writes=psum_acc must come
    after NKIMemset writes=psum_acc."""
    from nkigym.codegen.ir import BodyLeaf, DimInfo, KernelModule, Tensor, validate_dataflow_ordering

    class _FakeMemset:
        __name__ = "NKIMemset"

    class _FakeMatmul:
        __name__ = "NKIMatmul"

    _FakeAlloc = type("NKIAlloc", (), {})

    tensors = {
        "psum_acc": Tensor(
            name="psum_acc", dim_ids=("d0",), shape=(128,), dtype="float32", origin="intermediate", location="psum"
        )
    }
    dims = {"d0": DimInfo(dim_id="d0", total_size=128, tile_size=128, num_tiles=1)}

    alloc_leaf = BodyLeaf(op_cls=_FakeAlloc, reads={}, writes=("psum_acc",), reads_writes=())
    memset_leaf = BodyLeaf(op_cls=_FakeMemset, reads={}, writes=("psum_acc",), reads_writes=())
    matmul_leaf = BodyLeaf(op_cls=_FakeMatmul, reads={}, writes=(), reads_writes=("psum_acc",))

    good_module = KernelModule(
        func_name="f",
        param_names=[],
        return_name="psum_acc",
        tensors=tensors,
        dims=dims,
        body=[alloc_leaf, memset_leaf, matmul_leaf],
    )
    bad_module = KernelModule(
        func_name="f",
        param_names=[],
        return_name="psum_acc",
        tensors=tensors,
        dims=dims,
        body=[alloc_leaf, matmul_leaf, memset_leaf],
    )

    assert validate_dataflow_ordering(good_module) is True
    assert validate_dataflow_ordering(bad_module) is False


def test_validate_dataflow_ordering_rmw_leaf_counts_as_writer_for_next_leaf():
    """After an RMW leaf fires, the tensor counts as written for subsequent reads."""
    from nkigym.codegen.ir import BodyLeaf, DimInfo, KernelModule, Tensor, validate_dataflow_ordering

    _FakeAlloc = type("NKIAlloc", (), {})

    class _FakeMemset:
        __name__ = "NKIMemset"

    class _FakeMatmul:
        __name__ = "NKIMatmul"

    class _FakeCopy:
        __name__ = "NKITensorCopy"

    tensors = {
        "psum_acc": Tensor("psum_acc", ("d0",), (128,), "float32", "intermediate", "psum"),
        "sbuf_prod": Tensor("sbuf_prod", ("d0",), (128,), "bfloat16", "intermediate", "sbuf"),
    }
    dims = {"d0": DimInfo(dim_id="d0", total_size=128, tile_size=128, num_tiles=1)}

    body = [
        BodyLeaf(op_cls=_FakeAlloc, writes=("psum_acc",)),
        BodyLeaf(op_cls=_FakeAlloc, writes=("sbuf_prod",)),
        BodyLeaf(op_cls=_FakeMemset, writes=("psum_acc",)),
        BodyLeaf(op_cls=_FakeMatmul, reads_writes=("psum_acc",)),
        BodyLeaf(op_cls=_FakeCopy, reads={"src": "psum_acc"}, writes=("sbuf_prod",)),
    ]
    mod = KernelModule(func_name="f", param_names=[], return_name="sbuf_prod", tensors=tensors, dims=dims, body=body)
    assert validate_dataflow_ordering(mod) is True
