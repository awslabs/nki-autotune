"""Tests for the ``A.T @ B = (B.T @ A).T`` matmul transform."""

from __future__ import annotations

from test._simulation import _load_source

import numpy as np
import pytest

from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ir.tree import ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import MatmulTranspose, MatmulTransposeOption, TransformLegalityError

SMALL_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs_T": ((128, 512), "bfloat16"),
    "rhs": ((128, 512), "bfloat16"),
}
RECTANGULAR_SPECS: dict[str, tuple[tuple[int, ...], str]] = {
    "lhs_T": ((128, 512), "bfloat16"),
    "rhs": ((128, 1024), "bfloat16"),
}


@nkigym_kernel
def _matmul(lhs_T, rhs):
    """Canonical matmul fixture."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def _leaves(ir: KernelIR, op_name: str) -> list[tuple[int, ISANode]]:
    """Return ISA leaves whose class name equals ``op_name``."""
    leaves: list[tuple[int, ISANode]] = []
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, ISANode) and node.op_cls.__name__ == op_name:
            leaves.append((nid, node))
    return leaves


def _apply(ir: KernelIR) -> KernelIR:
    """Apply the sole legal matmul transpose option."""
    transform = MatmulTranspose()
    options = transform.analyze(ir)
    assert len(options) == 1
    return transform.apply(ir, options[0])


def test_analyze_offers_canonical_matmul_once():
    """A canonical matmul has one option and the rewritten matmul has none."""
    ir = build_initial_ir(_matmul, SMALL_SPECS)
    transform = MatmulTranspose()
    options = transform.analyze(ir)
    assert len(options) == 1
    matmul_leaf = _leaves(ir, "NKIMatmul")[0][0]
    matmul_block = next(
        ancestor for ancestor in reversed(ir.tree.ancestors(matmul_leaf)) if ancestor in set(ir.tree.blocks())
    )
    assert options[0].target_nid == matmul_block

    transformed = transform.apply(ir, options[0])
    assert transform.analyze(transformed) == []


def test_apply_swaps_operands_and_inserts_transpose_chain():
    """The graph rewrite has the expected ops, buffers, shapes, and dependencies."""
    ir = build_initial_ir(_matmul, RECTANGULAR_SPECS)
    transformed = _apply(ir)
    op_names = [
        transformed.tree.isa(nid).op_cls.__name__
        for nid in transformed.tree.preorder()
        if isinstance(transformed.tree.data(nid), ISANode)
    ]
    assert op_names == [
        "NKILoad",
        "NKILoad",
        "NKIMemset",
        "NKIMatmul",
        "NKITensorCopy",
        "NKITranspose",
        "NKITensorCopy",
        "NKIStore",
    ]

    matmul_nid, matmul = _leaves(transformed, "NKIMatmul")[0]
    first_drain_nid, first_drain = _leaves(transformed, "NKITensorCopy")[0]
    transpose_nid, transpose = _leaves(transformed, "NKITranspose")[0]
    final_drain_nid, _final_drain = _leaves(transformed, "NKITensorCopy")[1]
    assert matmul.operand_bindings["stationary"].tensor == "sbuf_rhs"
    assert matmul.operand_bindings["moving"].tensor == "sbuf_lhs_T"
    assert transpose.operand_bindings["data"].tensor == first_drain.operand_bindings["dst"].tensor
    assert transpose.operand_bindings["dst"].tensor == "psum_prod"
    assert transformed.dependency.direct_consumers(matmul_nid) == [first_drain_nid]
    assert transformed.dependency.direct_consumers(first_drain_nid) == [transpose_nid]
    assert transformed.dependency.direct_consumers(transpose_nid) == [final_drain_nid]

    assert transformed.buffer("psum_prod").shape == (512, 1024)
    assert transformed.buffer("psum_prod").physical_dtype() == "bfloat16"
    swapped_name = matmul.operand_bindings["dst"].tensor
    assert transformed.buffer(swapped_name).shape == (1024, 512)
    assert transformed.buffer(swapped_name).physical_dtype() == "float32"


def test_apply_is_pure_and_stale_option_fails_loudly():
    """Apply leaves its input unchanged and rejects repeat application."""
    ir = build_initial_ir(_matmul, SMALL_SPECS)
    before = render(ir)
    transform = MatmulTranspose()
    option = transform.analyze(ir)[0]
    transformed = transform.apply(ir, option)
    assert render(ir) == before
    with pytest.raises(TransformLegalityError, match="not an eligible canonical"):
        transform.apply(transformed, option)


def test_apply_rejects_unknown_target():
    """An option must name a currently eligible matmul block."""
    ir = build_initial_ir(_matmul, SMALL_SPECS)
    with pytest.raises(TransformLegalityError, match="not an eligible canonical"):
        MatmulTranspose().apply(ir, MatmulTransposeOption(target_nid=ir.tree.root))


def test_transformed_kernel_matches_numpy(tmp_path):
    """The rendered swapped matmul and nc_transpose chain is numerically exact in fp32."""
    transformed = _apply(build_initial_ir(_matmul, SMALL_SPECS))
    module = _load_source(render(transformed), tmp_path, "matmul_transpose")
    rng = np.random.default_rng(0)
    inputs = {
        "lhs_T": rng.standard_normal((128, 512)).astype(np.float32),
        "rhs": rng.standard_normal((128, 512)).astype(np.float32),
    }
    actual = np.asarray(simulate_fp32(module.nki__matmul)(**inputs))
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
