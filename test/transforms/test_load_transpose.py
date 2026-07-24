"""Tests for the load-transpose-drain DMA fusion."""

from __future__ import annotations

from test._simulation import _load_source
from test.transforms._fixtures import f_lhs_matmul

import numpy as np
import pytest

from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ForNode, ISANode
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import LoadTranspose, LoadTransposeOption, Split, SplitOption, TransformLegalityError

SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs": ((128, 128), "bfloat16"), "rhs": ((128, 512), "bfloat16")}
LARGE_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs": ((2048, 2048), "bfloat16"),
    "rhs": ((2048, 512), "bfloat16"),
}


def _op_names(ir: KernelIR) -> list[str]:
    """Return ISA class names in preorder."""
    return [ir.tree.isa(nid).op_cls.__name__ for nid in ir.tree.preorder() if isinstance(ir.tree.data(nid), ISANode)]


def test_apply_replaces_chain_and_removes_temporary_buffers():
    """The three-op chain becomes one direct HBM-to-SBUF DMA transpose."""
    ir = build_initial_ir(f_lhs_matmul, SPECS)
    transform = LoadTranspose()
    options = transform.analyze(ir)
    assert len(options) == 1

    transformed = transform.apply(ir, options[0])
    assert _op_names(transformed) == [
        "NKIDMATranspose",
        "NKILoad",
        "NKIMemset",
        "NKIMatmul",
        "NKITensorCopy",
        "NKIStore",
    ]
    assert "sbuf_lhs" not in transformed.all_buffers()
    assert "psum_lhs_T" not in transformed.all_buffers()
    dma_leaf = next(
        nid
        for nid in transformed.tree.preorder()
        if isinstance(transformed.tree.data(nid), ISANode)
        and transformed.tree.isa(nid).op_cls.__name__ == "NKIDMATranspose"
    )
    dma = transformed.tree.isa(dma_leaf)
    assert dma.operand_bindings["src"].tensor == "lhs"
    assert dma.operand_bindings["dst"].tensor == "sbuf_lhs_T"


def test_apply_is_pure_and_stale_option_fails_loudly():
    """Apply preserves its input and rejects an already-fused target."""
    ir = build_initial_ir(f_lhs_matmul, SPECS)
    before = render(ir)
    transform = LoadTranspose()
    option = transform.analyze(ir)[0]
    transformed = transform.apply(ir, option)
    assert render(ir) == before
    assert transform.analyze(transformed) == []
    with pytest.raises(TransformLegalityError, match="not an eligible canonical"):
        transform.apply(transformed, option)


def test_apply_rejects_unknown_target():
    """An option must name the load at the start of a canonical chain."""
    ir = build_initial_ir(f_lhs_matmul, SPECS)
    with pytest.raises(TransformLegalityError, match="not an eligible canonical"):
        LoadTranspose().apply(ir, LoadTransposeOption(target_nid=ir.tree.root))


def test_fused_dma_remains_correct_after_free_axis_split():
    """Normalizing the fused block preserves its M-by-K HBM orientation."""
    ir = build_initial_ir(f_lhs_matmul, LARGE_SPECS)
    transform = LoadTranspose()
    transformed = transform.apply(ir, transform.analyze(ir)[0])
    dma_leaf = next(
        nid
        for nid in transformed.tree.preorder()
        if isinstance(transformed.tree.data(nid), ISANode)
        and transformed.tree.isa(nid).op_cls.__name__ == "NKIDMATranspose"
    )
    free_loop = next(
        nid
        for nid in transformed.tree.ancestors(dma_leaf)
        if isinstance(transformed.tree.data(nid), ForNode) and transformed.tree.loop(nid).loop_var == "i_d1_0"
    )

    split = Split().apply(transformed, SplitOption(free_loop, (2, 8), None))
    source = render(split)
    assert (
        "src=lhs[i_d0_0 * 512:i_d0_0 * 512 + 512, " "i_d1_0 * 1024 + i_d1_1 * 128:i_d1_0 * 1024 + i_d1_1 * 128 + 128]"
    ) in source


def test_transformed_kernel_matches_numpy(tmp_path):
    """The direct DMA transpose kernel matches ``lhs @ rhs``."""
    ir = build_initial_ir(f_lhs_matmul, SPECS)
    transform = LoadTranspose()
    transformed = transform.apply(ir, transform.analyze(ir)[0])
    source = render(transformed)
    assert "nisa.dma_transpose(src=lhs" in source
    module = _load_source(source, tmp_path, "load_transpose")
    rng = np.random.default_rng(0)
    inputs = {
        "lhs": rng.standard_normal((128, 128)).astype(np.float32),
        "rhs": rng.standard_normal((128, 512)).astype(np.float32),
    }
    actual = np.asarray(simulate_fp32(module.nki_f_lhs_matmul)(**inputs))
    expected = inputs["lhs"] @ inputs["rhs"]
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
