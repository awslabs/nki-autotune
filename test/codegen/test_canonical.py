"""Task 11 iter-var canonical builder tests.

Tests verify the v2 canonical output. Old v1 tests (imported BodyLeaf /
LoopNode) migrated to iter-var equivalents here; the v1 counterparts live
in ``test_canonical_v1.py.bak`` if needed.
"""

import numpy as np

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, KernelModule, NKIOpCall, SBlock, validate_dataflow_ordering
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul_k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Simple matmul kernel (first-class buffers form) used as the canonical-module test fixture."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_INPUT_SPECS = {
    "lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def _collect_sblocks(module: KernelModule) -> list[SBlock]:
    """Pre-order DFS: every SBlock under every root."""
    blocks: list[SBlock] = []

    def walk(node: ForNode | SBlock) -> None:
        if isinstance(node, SBlock):
            blocks.append(node)
        else:
            for c in node.children:
                walk(c)

    for root in module.body:
        walk(root)
    return blocks


def test_builds_kernel_module_shape() -> None:
    """The builder returns a KernelModule with the expected surface shape.

    5 allocs (root SBlocks) + 6 compute trees (load, load, memset, matmul,
    tensor_copy, store) = 11 roots.
    """
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    assert km.func_name == "_matmul_k"
    assert km.param_names == ["lhs", "rhs"]
    assert len(km.body) == 11


def test_canonical_emits_sblock_per_op() -> None:
    """One SBlock per NKIOp call. 11 total = 5 allocs + 6 compute."""
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    blocks = _collect_sblocks(km)
    assert len(blocks) == 11


def test_canonical_iter_vars_per_block_distinct() -> None:
    """Two blocks touching d0 start with distinct iter var ids.

    Canonical build allocates fresh iter vars per block; shared var_ids
    would cross-couple unrelated blocks.
    """
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    blocks = _collect_sblocks(km)
    d0_ids_across_blocks: list[set[int]] = []
    for block in blocks:
        d0_ids = {iv.var_id for iv in block.iter_vars if iv.dim_id == "d0"}
        if d0_ids:
            d0_ids_across_blocks.append(d0_ids)
    flat = [i for s in d0_ids_across_blocks for i in s]
    assert len(set(flat)) == len(flat)
    assert len(d0_ids_across_blocks) >= 3


def test_canonical_alloc_sblocks_have_no_iter_vars() -> None:
    """Alloc SBlocks are root-level with empty iter_vars."""
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    alloc_roots = [
        root for root in km.body if isinstance(root, SBlock) and any(c.op_cls is NKIAlloc for c in root.body)
    ]
    assert len(alloc_roots) == 5
    for block in alloc_roots:
        assert block.iter_vars == []


def test_canonical_dataflow_validates() -> None:
    """Canonical module passes dataflow validation."""
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    assert validate_dataflow_ordering(km)


def test_matmul_block_operand_classification() -> None:
    """Matmul block puts stationary/moving in reads and dst in reads_writes."""
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    matmul_blocks = [b for b in _collect_sblocks(km) if b.body and b.body[0].op_cls is NKIMatmul]
    assert len(matmul_blocks) == 1
    block = matmul_blocks[0]
    assert set(block.reads.keys()) == {"stationary", "moving"}
    assert set(block.writes.keys()) == set()
    assert set(block.reads_writes.keys()) == {"dst"}
    assert block.reads["stationary"].tensor_name == "lhs_sbuf"
    assert block.reads["moving"].tensor_name == "rhs_sbuf"
    assert block.reads_writes["dst"].tensor_name == "psum_acc"


def test_fornode_nests_outermost_first() -> None:
    """ForNode wrapping preserves iter_vars order: outermost = iter_vars[0]."""
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    matmul_roots = [
        root
        for root in km.body
        if isinstance(root, ForNode) and any(b.body and b.body[0].op_cls is NKIMatmul for b in _blocks_under(root))
    ]
    assert len(matmul_roots) == 1
    root = matmul_roots[0]
    """Walk the single spine to the innermost SBlock, collecting each ForNode's iter_var."""
    node: ForNode | SBlock = root
    spine_ivs: list[int] = []
    while isinstance(node, ForNode):
        spine_ivs.append(node.iter_var.var_id)
        assert len(node.children) == 1
        node = node.children[0]
    assert isinstance(node, SBlock)
    assert spine_ivs == [iv.var_id for iv in node.iter_vars]


def test_compute_block_has_nkiopcall() -> None:
    """Every compute SBlock contains exactly one NKIOpCall with axis_map+dim_role."""
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    for block in _collect_sblocks(km):
        assert len(block.body) == 1
        call = block.body[0]
        assert isinstance(call, NKIOpCall)
        if call.op_cls is NKIAlloc:
            continue
        assert call.axis_map
        assert call.dim_role


def test_buffer_access_pattern_references_iter_vars() -> None:
    """Each BufferAccess's pattern coefficients reference block-local iter var ids."""
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    for block in _collect_sblocks(km):
        if block.body and block.body[0].op_cls is NKIAlloc:
            continue
        block_iv_ids = {iv.var_id for iv in block.iter_vars}
        for access in list(block.reads.values()) + list(block.writes.values()) + list(block.reads_writes.values()):
            for access_iv_id in access.iter_var_ids:
                assert access_iv_id in block_iv_ids


def _blocks_under(node: ForNode | SBlock) -> list[SBlock]:
    """Helper: list every SBlock under ``node``."""
    out: list[SBlock] = []
    if isinstance(node, SBlock):
        out.append(node)
    else:
        for c in node.children:
            out.extend(_blocks_under(c))
    return out


def test_canonical_render_matmul_lhsT_rhs_produces_valid_python() -> None:
    """Canonical render must parse as valid Python and contain expected ISA calls."""
    import ast

    from nkigym.codegen.render import render

    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    source = render(km)
    ast.parse(source)
    assert "nisa.dma_copy" in source
    assert "nisa.nc_matmul" in source
    assert "nisa.memset" in source
    assert "nl.ndarray" in source
    assert "@nki.jit" in source
    assert "def _matmul_k" in source
    assert "return hbm_out" in source


def test_canonical_render_iter_var_names_canonical() -> None:
    """Rendered kernel has canonical ``i_<dim>_<ordinal>`` loop variable names."""
    import re

    from nkigym.codegen.render import render

    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    source = render(km)
    pattern = re.compile(r"for i_[a-zA-Z0-9_]+_\d+ in range")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("for "):
            assert pattern.search(stripped), f"Loop var not canonical: {stripped}"


def test_canonical_render_matmul_sbuf_slices_are_3d() -> None:
    """SBUF/PSUM tensor slices must have 3 dims matching the 3D declaration.

    Regression: post-Task 13 ``place_buffers`` emits 3D SBUF/PSUM shapes
    ``(P_tile, num_P_slots, F_tile * num_F_tiles)``; the renderer must
    emit matching 3D slices (``[0:P_tile, slot_index, F_slice]``), not
    the 2D slice built naively from ``BufferAccess.pattern``. HBM
    tensors stay 2D with tile-size-scaled starts on every dim.
    """
    from nkigym.codegen.render import render

    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    source = render(km)

    """SBUF intermediate: 3D with partition/slot/F convention."""
    assert "lhs_sbuf[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128]" in source
    assert "rhs_sbuf[0:128, i_d0_0, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source
    assert "sbuf_prod[0:128, i_d1_0, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source

    """PSUM intermediate: same 3D convention (different shape)."""
    assert "psum_acc[0:128, i_d1_0, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source

    """HBM params + return: 2D with tile-size-scaled starts on every dim."""
    assert "lhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 128 : (i_d1_0) * 128 + 128]" in source
    assert "rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source
    assert "hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source

    """Regression guard: the pre-fix 2D SBUF slice pattern must not appear."""
    assert "lhs_sbuf[i_d0_0 : i_d0_0 + 128" not in source
    assert "psum_acc[i_d1_0 : i_d1_0 + 128" not in source
