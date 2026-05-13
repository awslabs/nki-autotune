"""Unit tests for IterVar (TVM-style iter-var identity)."""

import pytest

from nkigym.ir.ir import (
    AccessRange,
    Axis,
    BufferAccess,
    ForNode,
    IterVar,
    KernelIR,
    NKIOpCall,
    SBlock,
    Tensor,
    blocks_under,
    replace_at_path,
    resolve_node,
    validate_dataflow_ordering,
)
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy


def test_iter_var_is_frozen() -> None:
    """IterVar must be immutable — atoms retire and replace, never mutate."""
    iv = IterVar(var_id=0, axis_id=0, extent=4, role=AxisRole.PARALLEL)
    with pytest.raises(AttributeError):
        iv.extent = 8  # type: ignore[misc]


def test_iter_var_equality_by_fields() -> None:
    """Two IterVars with the same fields compare equal."""
    a = IterVar(var_id=0, axis_id=0, extent=4, role=AxisRole.PARALLEL)
    b = IterVar(var_id=0, axis_id=0, extent=4, role=AxisRole.PARALLEL)
    assert a == b
    assert hash(a) == hash(b)


def test_iter_var_distinct_ids() -> None:
    """Same axis, different var_ids → unequal (distinct iter vars)."""
    a = IterVar(var_id=0, axis_id=0, extent=4, role=AxisRole.PARALLEL)
    b = IterVar(var_id=1, axis_id=0, extent=4, role=AxisRole.PARALLEL)
    assert a != b


def test_access_range_immutable() -> None:
    """AccessRange must be frozen."""
    ar = AccessRange.make(coeffs={0: 1}, const_offset=0, extent=128)
    with pytest.raises(AttributeError):
        ar.const_offset = 4  # type: ignore[misc]


def test_access_range_simple_1to1() -> None:
    """coeffs={iv: 1}, offset=0, extent=tile_size encodes direct indexing."""
    ar = AccessRange.make(coeffs={7: 1}, const_offset=0, extent=128)
    assert ar.coeffs == {7: 1}
    assert ar.extent == 128


def test_access_range_split_rewrite() -> None:
    """After Split(factor=2), an iter var coefficient becomes inner_extent."""
    ar = AccessRange.make(coeffs={0: 2, 1: 1}, const_offset=0, extent=128)
    assert ar.coeffs[0] == 2
    assert ar.coeffs[1] == 1


def test_access_range_make_normalizes_ordering() -> None:
    """AccessRange.make sorts coeff pairs for hashability + determinism."""
    a = AccessRange.make(coeffs={1: 5, 0: 3}, const_offset=0, extent=128)
    b = AccessRange.make(coeffs={0: 3, 1: 5}, const_offset=0, extent=128)
    assert a == b
    assert hash(a) == hash(b)


def test_buffer_access_immutable() -> None:
    """BufferAccess is frozen."""
    ba = BufferAccess(
        tensor_name="x", iter_var_ids=(0,), pattern=(AccessRange.make(coeffs={0: 1}, const_offset=0, extent=128),)
    )
    with pytest.raises(AttributeError):
        ba.tensor_name = "y"  # type: ignore[misc]


def test_buffer_access_uses_tuple_for_patterns() -> None:
    """pattern field must be a tuple (frozen dataclass requires hashable fields)."""
    p = (
        AccessRange.make(coeffs={0: 1}, const_offset=0, extent=128),
        AccessRange.make(coeffs={1: 1}, const_offset=0, extent=2048),
    )
    ba = BufferAccess(tensor_name="t", iter_var_ids=(0, 1), pattern=p)
    assert len(ba.pattern) == 2


def test_nki_op_call_immutable() -> None:
    """NKIOpCall is frozen."""
    call = NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})
    with pytest.raises(AttributeError):
        call.op_cls = object  # type: ignore[misc]


def test_sblock_single_leaf_canonical() -> None:
    """Canonical build emits single-NKIOpCall SBlocks."""
    call = NKIOpCall(
        op_cls=NKIMatmul,
        kwargs={},
        axis_map={"K": 0, "M": 1, "N": 3},
        dim_role={0: AxisRole.ACCUMULATION, 1: AxisRole.PARALLEL, 3: AxisRole.PARALLEL},
    )
    lhs_access = BufferAccess(
        tensor_name="lhs_T_sbuf",
        iter_var_ids=(0, 1),
        pattern=(AccessRange.make({0: 1}, 0, 128), AccessRange.make({1: 1}, 0, 128)),
    )
    block = SBlock(
        iter_vars=[IterVar(0, 0, 4, AxisRole.ACCUMULATION)],
        reads={"stationary": lhs_access},
        writes={},
        reads_writes={},
        body=[call],
    )
    assert len(block.body) == 1
    assert block.iter_vars[0].var_id == 0


def test_sblock_supports_multi_leaf_body() -> None:
    """SBlock.body is a list; future fused-block atoms will emit len > 1."""
    calls = [
        NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={}),
        NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={}),
    ]
    block = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=calls)
    assert len(block.body) == 2


def test_sblock_annotations_default_empty() -> None:
    """SBlock.annotations defaults to empty dict."""
    block = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=[])
    assert block.annotations == {}


def test_for_node_binds_iter_var_by_reference() -> None:
    """ForNode stores an IterVar; multiple ForNodes can bind distinct iter
    vars on the same axis."""
    iv_outer = IterVar(0, 0, 2, AxisRole.PARALLEL)
    iv_inner = IterVar(1, 0, 2, AxisRole.PARALLEL)
    outer = ForNode(iter_var=iv_outer, children=[])
    inner = ForNode(iter_var=iv_inner, children=[])
    outer.children.append(inner)
    assert outer.iter_var.var_id == 0
    assert outer.children[0].iter_var.var_id == 1


def test_for_node_annotations_default_empty() -> None:
    """annotations default to empty dict."""
    iv = IterVar(0, 0, 4, AxisRole.PARALLEL)
    fn = ForNode(iter_var=iv, children=[])
    assert fn.annotations == {}


def test_for_node_supports_sblock_child() -> None:
    """ForNode.children can hold SBlocks and nested ForNodes."""
    iv = IterVar(0, 0, 4, AxisRole.PARALLEL)
    block = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=[])
    fn = ForNode(iter_var=iv, children=[block])
    assert isinstance(fn.children[0], SBlock)


def test_for_node_name_defaults_none() -> None:
    """ForNode.name defaults to None; set by canonicalize pass."""
    iv = IterVar(0, 0, 4, AxisRole.PARALLEL)
    fn = ForNode(iter_var=iv, children=[])
    assert fn.name is None


def test_kernel_ir_minimal() -> None:
    """KernelIR carries signature + tensors + axes + body + iter_var_counter."""
    tensors = {
        "x": Tensor(
            name="x", dim_ids=("d0",), shape=(128,), dtype="float32", origin="param", location="hbm", buffer_degree={}
        )
    }
    m = KernelIR(
        func_name="f", param_names=["x"], return_name="x", tensors=tensors, axes={}, iter_var_counter=0, body=[]
    )
    m.allocate_axis(name="d0", total_size=128)
    assert m.iter_var_counter == 0
    assert m.body == []


def test_kernel_ir_allocates_monotonic_iter_var_ids() -> None:
    """allocate_iter_var bumps the counter and returns a fresh IterVar."""
    m = KernelIR(func_name="f", param_names=[], return_name="", tensors={}, axes={}, iter_var_counter=0, body=[])
    axis = m.allocate_axis(name="d0", total_size=128)
    iv1 = m.allocate_iter_var(axis.axis_id, extent=4, role=AxisRole.PARALLEL)
    iv2 = m.allocate_iter_var(axis.axis_id, extent=4, role=AxisRole.PARALLEL)
    assert iv1.var_id == 0
    assert iv2.var_id == 1
    assert m.iter_var_counter == 2


def test_kernel_ir_default_body_and_dep() -> None:
    """Empty body and fresh DepCache by default."""
    m = KernelIR(func_name="f", param_names=[], return_name="", tensors={}, axes={})
    assert m.body == []
    assert m.iter_var_counter == 0
    assert m.dep is not None


def _mk_mod_with_single_block() -> KernelIR:
    """Build a minimal module: one ForNode → SBlock tree."""
    m = KernelIR(func_name="f", param_names=[], return_name="", tensors={}, axes={}, iter_var_counter=0, body=[])
    axis = m.allocate_axis(name="d0", total_size=128)
    iv = m.allocate_iter_var(axis.axis_id, 4, AxisRole.PARALLEL)
    block = SBlock(iter_vars=[iv], reads={}, writes={}, reads_writes={}, body=[])
    root = ForNode(iter_var=iv, children=[block])
    m.body.append(root)
    return m


def test_resolve_node_root() -> None:
    """Path (0,) returns the first root."""
    m = _mk_mod_with_single_block()
    node = resolve_node(m.body, (0,))
    assert isinstance(node, ForNode)


def test_resolve_node_nested_block() -> None:
    """Path (0, 0) returns the SBlock under the root ForNode."""
    m = _mk_mod_with_single_block()
    node = resolve_node(m.body, (0, 0))
    assert isinstance(node, SBlock)


def test_resolve_node_invalid_path_out_of_range() -> None:
    """Out-of-range path returns None."""
    m = _mk_mod_with_single_block()
    assert resolve_node(m.body, (5,)) is None


def test_resolve_node_empty_path_returns_none() -> None:
    """Empty path returns None — there is no 'forest node'."""
    m = _mk_mod_with_single_block()
    assert resolve_node(m.body, ()) is None


def test_resolve_node_descends_through_sblock_returns_none() -> None:
    """Cannot path-descend through an SBlock (SBlocks are leaves)."""
    m = _mk_mod_with_single_block()
    assert resolve_node(m.body, (0, 0, 0)) is None


def test_blocks_under_yields_descendant_sblocks() -> None:
    """blocks_under walks the subtree yielding every SBlock."""
    m = _mk_mod_with_single_block()
    root = m.body[0]
    assert isinstance(root, ForNode)
    blocks = list(blocks_under(root))
    assert len(blocks) == 1
    assert isinstance(blocks[0], SBlock)


def test_blocks_under_yields_self_when_sblock() -> None:
    """blocks_under(sblock) yields [sblock]."""
    block = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=[])
    assert list(blocks_under(block)) == [block]


def test_replace_at_path_swaps_subtree() -> None:
    """replace_at_path returns a new body with the node at path replaced."""
    m = _mk_mod_with_single_block()
    new_block = SBlock(
        iter_vars=[],
        reads={},
        writes={},
        reads_writes={},
        body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})],
    )
    new_body = replace_at_path(m.body, (0, 0), new_block)
    new_root = new_body[0]
    assert isinstance(new_root, ForNode)
    assert new_root.children[0] is new_block
    assert m.body[0] is not new_root  # type: ignore[comparison-overlap]


def test_replace_at_path_empty_raises() -> None:
    """Empty path is illegal."""
    m = _mk_mod_with_single_block()
    with pytest.raises(ValueError):
        replace_at_path(m.body, (), SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=[]))


def test_replace_at_path_replaces_root() -> None:
    """Path (0,) replaces the first root element."""
    m = _mk_mod_with_single_block()
    new_root = SBlock(iter_vars=[], reads={}, writes={}, reads_writes={}, body=[])
    new_body = replace_at_path(m.body, (0,), new_root)
    assert new_body[0] is new_root


def _mk_access(name: str, iv_id: int) -> BufferAccess:
    """Simple 1:1 access helper."""
    return BufferAccess(tensor_name=name, iter_var_ids=(iv_id,), pattern=(AccessRange.make({iv_id: 1}, 0, 128),))


def test_validate_rejects_read_before_alloc() -> None:
    """Non-alloc block reading an un-alloc'd tensor must fail validation (rule 2)."""
    tensors = {
        "x": Tensor(
            name="x",
            dim_ids=("d0",),
            shape=(128,),
            dtype="float32",
            origin="intermediate",
            location="sbuf",
            buffer_degree={},
        )
    }
    m = KernelIR(
        func_name="f",
        param_names=[],
        return_name="x",
        tensors=tensors,
        axes={},
        iter_var_counter=0,
        body=[SBlock(iter_vars=[], reads={"src": _mk_access("x", 0)}, writes={}, reads_writes={}, body=[])],
    )
    m.allocate_axis(name="d0", total_size=128)
    assert not validate_dataflow_ordering(m)


def test_validate_accepts_alloc_then_write_then_read() -> None:
    """Canonical alloc → write → read is legal."""
    tensors = {
        "x": Tensor(
            name="x",
            dim_ids=("d0",),
            shape=(128,),
            dtype="float32",
            origin="intermediate",
            location="sbuf",
            buffer_degree={},
        )
    }
    alloc_call = NKIOpCall(
        op_cls=NKIAlloc,
        kwargs={"tensor_name": "x", "location": "sbuf", "shape": (128,), "dtype": "float32"},
        axis_map={},
        dim_role={},
    )
    alloc_block = SBlock(
        iter_vars=[], reads={}, writes={"output": _mk_access("x", 0)}, reads_writes={}, body=[alloc_call]
    )
    writer_call = NKIOpCall(op_cls=NKILoad, kwargs={}, axis_map={}, dim_role={})
    writer = SBlock(iter_vars=[], reads={}, writes={"dst": _mk_access("x", 0)}, reads_writes={}, body=[writer_call])
    m = KernelIR(
        func_name="f",
        param_names=[],
        return_name="x",
        tensors=tensors,
        axes={},
        iter_var_counter=0,
        body=[alloc_block, writer],
    )
    m.allocate_axis(name="d0", total_size=128)
    assert validate_dataflow_ordering(m)


def test_validate_param_tensors_are_pre_allocated() -> None:
    """Params are considered allocated from the start (rule 2 exemption)."""
    tensors = {
        "p": Tensor(
            name="p", dim_ids=("d0",), shape=(128,), dtype="float32", origin="param", location="hbm", buffer_degree={}
        ),
        "out": Tensor(
            name="out",
            dim_ids=("d0",),
            shape=(128,),
            dtype="float32",
            origin="return",
            location="hbm",
            buffer_degree={},
        ),
    }
    alloc_call = NKIOpCall(
        op_cls=NKIAlloc,
        kwargs={"tensor_name": "out", "location": "hbm", "shape": (128,), "dtype": "float32"},
        axis_map={},
        dim_role={},
    )
    alloc_block = SBlock(
        iter_vars=[], reads={}, writes={"output": _mk_access("out", 0)}, reads_writes={}, body=[alloc_call]
    )
    copy = SBlock(
        iter_vars=[],
        reads={"src": _mk_access("p", 0)},
        writes={"dst": _mk_access("out", 0)},
        reads_writes={},
        body=[NKIOpCall(op_cls=NKILoad, kwargs={}, axis_map={}, dim_role={})],
    )
    m = KernelIR(
        func_name="f",
        param_names=["p"],
        return_name="out",
        tensors=tensors,
        axes={},
        iter_var_counter=0,
        body=[alloc_block, copy],
    )
    m.allocate_axis(name="d0", total_size=128)
    assert validate_dataflow_ordering(m)


def test_validate_rejects_alloc_only_return() -> None:
    """Rule 5: an alloc alone does NOT satisfy the return-produced rule.

    The return tensor needs a real value-producing write — an ``NKIAlloc``
    declares storage but writes nothing meaningful.
    """
    tensors = {
        "out": Tensor(
            name="out",
            dim_ids=("d0",),
            shape=(128,),
            dtype="float32",
            origin="return",
            location="hbm",
            buffer_degree={},
        )
    }
    alloc_call = NKIOpCall(
        op_cls=NKIAlloc,
        kwargs={"tensor_name": "out", "location": "hbm", "shape": (128,), "dtype": "float32"},
        axis_map={},
        dim_role={},
    )
    alloc_block = SBlock(
        iter_vars=[], reads={}, writes={"output": _mk_access("out", 0)}, reads_writes={}, body=[alloc_call]
    )
    m = KernelIR(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors=tensors,
        axes={},
        iter_var_counter=0,
        body=[alloc_block],
    )
    m.allocate_axis(name="d0", total_size=128)
    assert not validate_dataflow_ordering(m)


def test_validate_rejects_read_between_rmw_writes() -> None:
    """Rule 4: non-RMW read of T between two RMW writes of T must fail.

    Construction: alloc T; memset T; RMW write #1; non-RMW read of T;
    RMW write #2. Rule 4 requires the non-RMW read to come after the
    LAST RMW write. The memset exists so that the failing rule here is
    specifically rule 4 (not rule 3 — the RMW has a prior real write).
    """
    tensors = {
        "psum": Tensor(
            name="psum",
            dim_ids=("d0",),
            shape=(128,),
            dtype="float32",
            origin="intermediate",
            location="psum",
            buffer_degree={},
        ),
        "out": Tensor(
            name="out",
            dim_ids=("d0",),
            shape=(128,),
            dtype="bfloat16",
            origin="return",
            location="hbm",
            buffer_degree={},
        ),
    }

    alloc_psum = SBlock(
        iter_vars=[],
        reads={},
        writes={"output": _mk_access("psum", 0)},
        reads_writes={},
        body=[
            NKIOpCall(
                op_cls=NKIAlloc,
                kwargs={"tensor_name": "psum", "location": "psum", "shape": (128,), "dtype": "float32"},
                axis_map={},
                dim_role={},
            )
        ],
    )
    alloc_out = SBlock(
        iter_vars=[],
        reads={},
        writes={"output": _mk_access("out", 0)},
        reads_writes={},
        body=[
            NKIOpCall(
                op_cls=NKIAlloc,
                kwargs={"tensor_name": "out", "location": "hbm", "shape": (128,), "dtype": "bfloat16"},
                axis_map={},
                dim_role={},
            )
        ],
    )
    memset_psum = SBlock(
        iter_vars=[],
        reads={},
        writes={"dst": _mk_access("psum", 0)},
        reads_writes={},
        body=[NKIOpCall(op_cls=NKIMemset, kwargs={"value": 0.0}, axis_map={}, dim_role={})],
    )
    rmw_1 = SBlock(
        iter_vars=[],
        reads={},
        writes={},
        reads_writes={"dst": _mk_access("psum", 0)},
        body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})],
    )
    reader = SBlock(
        iter_vars=[],
        reads={"src": _mk_access("psum", 0)},
        writes={"dst": _mk_access("out", 0)},
        reads_writes={},
        body=[NKIOpCall(op_cls=NKITensorCopy, kwargs={}, axis_map={}, dim_role={})],
    )
    rmw_2 = SBlock(
        iter_vars=[],
        reads={},
        writes={},
        reads_writes={"dst": _mk_access("psum", 0)},
        body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})],
    )
    m = KernelIR(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors=tensors,
        axes={},
        iter_var_counter=0,
        body=[alloc_psum, alloc_out, memset_psum, rmw_1, reader, rmw_2],
    )
    m.allocate_axis(name="d0", total_size=128)
    assert not validate_dataflow_ordering(m)


def test_validate_accepts_read_after_all_rmw_writes() -> None:
    """Rule 4 positive case: non-RMW read of T after ALL RMW writes is legal.

    Includes a memset initializer before the RMW sequence — under the
    tightened rule 3, the first RMW requires a prior real write.
    """
    tensors = {
        "psum": Tensor(
            name="psum",
            dim_ids=("d0",),
            shape=(128,),
            dtype="float32",
            origin="intermediate",
            location="psum",
            buffer_degree={},
        ),
        "out": Tensor(
            name="out",
            dim_ids=("d0",),
            shape=(128,),
            dtype="bfloat16",
            origin="return",
            location="hbm",
            buffer_degree={},
        ),
    }

    alloc_psum = SBlock(
        iter_vars=[],
        reads={},
        writes={"output": _mk_access("psum", 0)},
        reads_writes={},
        body=[
            NKIOpCall(
                op_cls=NKIAlloc,
                kwargs={"tensor_name": "psum", "location": "psum", "shape": (128,), "dtype": "float32"},
                axis_map={},
                dim_role={},
            )
        ],
    )
    alloc_out = SBlock(
        iter_vars=[],
        reads={},
        writes={"output": _mk_access("out", 0)},
        reads_writes={},
        body=[
            NKIOpCall(
                op_cls=NKIAlloc,
                kwargs={"tensor_name": "out", "location": "hbm", "shape": (128,), "dtype": "bfloat16"},
                axis_map={},
                dim_role={},
            )
        ],
    )
    memset_psum = SBlock(
        iter_vars=[],
        reads={},
        writes={"dst": _mk_access("psum", 0)},
        reads_writes={},
        body=[NKIOpCall(op_cls=NKIMemset, kwargs={"value": 0.0}, axis_map={}, dim_role={})],
    )
    rmw_1 = SBlock(
        iter_vars=[],
        reads={},
        writes={},
        reads_writes={"dst": _mk_access("psum", 0)},
        body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})],
    )
    rmw_2 = SBlock(
        iter_vars=[],
        reads={},
        writes={},
        reads_writes={"dst": _mk_access("psum", 0)},
        body=[NKIOpCall(op_cls=NKIMatmul, kwargs={}, axis_map={}, dim_role={})],
    )
    reader = SBlock(
        iter_vars=[],
        reads={"src": _mk_access("psum", 0)},
        writes={"dst": _mk_access("out", 0)},
        reads_writes={},
        body=[NKIOpCall(op_cls=NKITensorCopy, kwargs={}, axis_map={}, dim_role={})],
    )
    m = KernelIR(
        func_name="f",
        param_names=[],
        return_name="out",
        tensors=tensors,
        axes={},
        iter_var_counter=0,
        body=[alloc_psum, alloc_out, memset_psum, rmw_1, rmw_2, reader],
    )
    m.allocate_axis(name="d0", total_size=128)
    assert validate_dataflow_ordering(m)


def test_validate_rejects_unwritten_return() -> None:
    """Rule 5 negative case: return tensor with NO writer (no alloc, no op)
    must fail validation."""
    tensors = {
        "out": Tensor(
            name="out",
            dim_ids=("d0",),
            shape=(128,),
            dtype="bfloat16",
            origin="return",
            location="hbm",
            buffer_degree={},
        )
    }
    """Empty body — no alloc, no writers → "out" never enters `written`."""
    m = KernelIR(
        func_name="f", param_names=[], return_name="out", tensors=tensors, axes={}, iter_var_counter=0, body=[]
    )
    m.allocate_axis(name="d0", total_size=128)
    assert not validate_dataflow_ordering(m)


def test_axis_only_carries_identity_name_and_total_size() -> None:
    """Per-op tiling model: Axis holds identity + display name + total extent + source_axes."""
    from dataclasses import fields

    field_names = {f.name for f in fields(Axis)}
    assert field_names == {"axis_id", "name", "total_size", "source_axes"}
