"""Contract tests for dimension analysis and the BlockNode IR payloads."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from test.transforms._fixtures import INPUT_SPECS, f_lhs_matmul, f_matmul

import pytest

import nkigym.ir as ir_package
import nkigym.ir.dimension_analysis as dimension_analysis
from nkigym.ir import BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelIR, KernelTree, build_initial_ir
from nkigym.ir.arith.expr import Const, Var
from nkigym.ir.dimension_analysis import analyze_dimensions
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore


@nkigym_kernel
def _reduce_then_activate(x):
    """Reduce one free axis, activate the vector, and store it."""
    sx = NKILoad()(src=x)
    red = NKIActivationReduce(op="square", reduce_op="add")(data=sx)
    act = NKIActivation(op="rsqrt")(data=red)
    out = NKIStore()(src=act)
    return out


def _isa_leaves(ir: KernelIR) -> list[ISANode]:
    """Return every ISA payload in preorder."""
    return [ir.tree.isa(nid) for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode)]


def test_dimension_analysis_tracks_shapes_dtypes_and_ssa_outputs() -> None:
    """Tracing tracks transpose axes, one-dimensional reductions, SSA names, and parameter dtypes."""
    transpose = analyze_dimensions(f_lhs_matmul, {"lhs": ((256, 384), "bfloat16"), "rhs": ((384, 512), "bfloat16")})
    lhs = transpose.tensors["sbuf_lhs"]
    lhs_t = transpose.tensors["psum_lhs_T"]
    product = transpose.tensors["psum_prod"]
    assert lhs_t.shape == (384, 256)
    assert lhs_t.dim_ids == (lhs.dim_ids[1], lhs.dim_ids[0])
    assert lhs_t.storage_dtype is None
    assert product.storage_dtype == "float32"

    reduced = analyze_dimensions(_reduce_then_activate, {"x": ((128, 512), "bfloat16")})
    assert len(reduced.tensors["act"].dim_ids) == 1
    assert reduced.dim_sizes[reduced.tensors["act"].dim_ids[0]] == 128
    reduced_ir = build_initial_ir(_reduce_then_activate, {"x": ((128, 512), "bfloat16")})
    assert reduced_ir.buffer("act").shape == (128,)

    analysis = analyze_dimensions(f_matmul, INPUT_SPECS)
    tensors = analysis.tensors
    assert set(tensors) == {"lhs_T", "rhs", "sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod", "hbm_out"}
    assert tensors["psum_prod"].location == "psum"
    assert tensors["psum_prod"].dtype == "bfloat16"
    assert tensors["sbuf_prod"].location == "sbuf"
    assert tensors["sbuf_prod"].dtype == "bfloat16"
    assert tensors["hbm_out"].location == "shared_hbm"
    assert analysis.dim_sizes[tensors["psum_prod"].dim_ids[0]] == 2048
    assert analysis.dim_sizes[tensors["psum_prod"].dim_ids[1]] == 2048

    float16 = analyze_dimensions(f_matmul, {"lhs_T": ((2048, 2048), "float16"), "rhs": ((2048, 2048), "float16")})
    assert float16.tensors["lhs_T"].dtype == "float16"
    assert float16.tensors["rhs"].dtype == "float16"


def test_canonical_leaf_metadata_and_return_contract() -> None:
    """Canonical leaves carry only operation kwargs and analysis requires a named return."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    memsets = [leaf for leaf in _isa_leaves(ir) if leaf.op_cls.__name__ == "NKIMemset"]
    assert len(memsets) == 1
    assert memsets[0].kwargs == {"value": 0.0}
    for op_name in ("NKILoad", "NKIStore", "NKIMatmul", "NKITensorCopy"):
        for leaf in _isa_leaves(ir):
            if leaf.op_cls.__name__ == op_name:
                assert leaf.kwargs == {}, f"{op_name} leaf kwargs={leaf.kwargs}"
    assert ir.return_name == "hbm_out"

    @nkigym_kernel
    def no_return(x):
        """Build an intentionally invalid graph with no return statement."""
        sbuf_x = NKILoad()(src=x)

    with pytest.raises(ValueError, match="return"):
        analyze_dimensions(no_return, {"x": ((128, 128), "bfloat16")})


def test_kernel_tree_counts_and_filters_payloads() -> None:
    """KernelTree counts graph nodes and its block iterator excludes loop payloads."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    assert not hasattr(ir.tree, "dim_sizes")
    assert ir.tree.num_nodes == ir.tree.graph.number_of_nodes()

    tree = KernelTree()
    assert tree.num_nodes == 1
    before = tree.num_nodes
    tree.add_node(ForNode(loop_var="i", extent=2), parent=tree.root)
    assert tree.num_nodes == before + 1

    blocks = KernelTree()
    block = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    first = blocks.add_node(block, parent=blocks.root)
    second = blocks.add_node(block, parent=blocks.root)
    blocks.add_node(ForNode(loop_var="i_d0_0", extent=2), parent=first)
    assert set(blocks.blocks()) == {blocks.root, first, second}


def test_ir_exports_only_current_public_payloads() -> None:
    """The package exports current payloads and keeps tracer internals private."""
    for removed in ("DimensionAnalysis", "OpAxes", "analyze_dimensions"):
        assert not hasattr(ir_package, removed), f"nkigym.ir.{removed} should have been removed"
    assert not hasattr(dimension_analysis, "OpAxes")
    assert hasattr(dimension_analysis, "_OpRecord")
    assert ir_package.BlockNode is BlockNode
    assert ir_package.Buffer is Buffer
    assert ir_package.BufferRegion is BufferRegion
    assert ir_package.IterVar is IterVar


def test_envelope_and_dump_contain_only_text_artifacts(tmp_path: Path) -> None:
    """KernelIR metadata is complete and dump emits only Markdown and Python."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    envelope = ir._render_envelope_md()
    assert "# `f_matmul`" in envelope
    assert "`lhs_T`" in envelope and "`rhs`" in envelope
    assert "**Returns**: `hbm_out`" in envelope
    assert "## Buffers" in envelope
    assert "| Name | Location | Dtype | Shape |" in envelope
    assert "psum" in envelope or "sbuf" in envelope
    assert "bfloat16" in envelope or "float32" in envelope

    ir.dump(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {"envelope.md", "kernel.py"}
    assert (tmp_path / "envelope.md").stat().st_size > 0
    assert (tmp_path / "kernel.py").stat().st_size > 0


def test_payload_value_objects_have_structural_semantics() -> None:
    """IR payload dataclasses compare structurally and isolate mutable annotation defaults."""
    parallel = IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL)
    assert parallel == IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL)
    assert parallel != IterVar(axis="M", dom=(0, 2048), role=AxisRole.ACCUMULATION)

    buffer = Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    assert buffer == Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum")
    region = BufferRegion(tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)),))
    assert region == BufferRegion(tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)),))

    minimal = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=())
    assert minimal.iter_vars == ()
    assert minimal.alloc_buffers == ()
    assert minimal.annotations == {}
    full = BlockNode(
        iter_vars=(parallel, IterVar(axis="N", dom=(0, 2048), role=AxisRole.PARALLEL)),
        iter_values=(Var(name="i_M"), Var(name="i_N")),
        reads=(),
        writes=(region,),
        alloc_buffers=(buffer,),
    )
    assert len(full.iter_vars) == 2
    assert len(full.alloc_buffers) == 1
    other = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=())
    minimal.annotations["k"] = 1
    assert "k" not in other.annotations


def test_loop_and_isa_payloads_exclude_legacy_fields() -> None:
    """ForNode and ISANode expose current fields without legacy schedule metadata."""
    loop = ForNode(loop_var="i_M_0", extent=16)
    assert loop.loop_var == "i_M_0"
    assert loop.extent == 16
    assert not hasattr(loop, "dim")
    assert not hasattr(loop, "trip")

    bindings = {
        "stationary": BufferRegion(
            tensor="sbuf_lhs_T", ranges=((Var(name="vK"), Const(value=1)), (Var(name="vM"), Const(value=128)))
        ),
        "moving": BufferRegion(
            tensor="sbuf_rhs", ranges=((Var(name="vK"), Const(value=1)), (Var(name="vN"), Const(value=512)))
        ),
        "dst": BufferRegion(
            tensor="psum_prod", ranges=((Var(name="vM"), Const(value=128)), (Var(name="vN"), Const(value=512)))
        ),
    }
    node = ISANode(op_cls=NKIMatmul, operand_bindings=bindings, kwargs={})
    assert node.op_cls is NKIMatmul
    assert set(node.operand_bindings) == {"stationary", "moving", "dst"}
    for old in ("reads", "writes", "rmw", "axis_map", "tensorize_sizes", "location", "dtype"):
        assert not hasattr(node, old), f"ISANode unexpectedly carries legacy field {old!r}"


def test_kernel_ir_schema_helpers_and_canonical_allocations() -> None:
    """KernelIR stays slim while exposing buffer helpers and canonical allocations."""
    field_names = {field.name for field in dataclasses.fields(KernelIR)}
    assert field_names == {"func_name", "param_names", "return_name", "tree", "dependency", "param_buffers"}
    assert callable(KernelIR.all_buffers)
    assert callable(KernelIR.buffer)
    assert callable(KernelIR.axis_extent)

    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    root = ir.tree.data(ir.tree.root)
    assert isinstance(root, BlockNode)
    all_buffers = set()
    for nid in ir.tree.blocks():
        block = ir.tree.data(nid)
        assert isinstance(block, BlockNode)
        all_buffers.update(buffer.name for buffer in block.alloc_buffers)
    assert {"sbuf_lhs_T", "sbuf_rhs", "psum_prod", "sbuf_prod", "hbm_out"} <= all_buffers
