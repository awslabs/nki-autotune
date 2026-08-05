"""Contract tests for dimension analysis and the BlockNode IR payloads."""

from __future__ import annotations

from pathlib import Path
from test.transforms._fixtures import INPUT_SPECS, f_lhs_matmul, f_matmul

import pytest

from nkigym.ir import ISANode, KernelIR, build_initial_ir
from nkigym.ir.dimension_analysis import analyze_dimensions
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.load import NKILoad
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
