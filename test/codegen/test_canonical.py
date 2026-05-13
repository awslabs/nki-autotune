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
    d0_axis_id = km.axis_id_by_name("d0")
    blocks = _collect_sblocks(km)
    d0_ids_across_blocks: list[set[int]] = []
    for block in blocks:
        d0_ids = {iv.var_id for iv in block.iter_vars if iv.axis_id == d0_axis_id}
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

    Per-op tiling: each op slices buffers according to its own tile sizes.
    NKILoad writes full-F (2048); NKIMatmul reads with M=128/N=512 tiles.
    Writer tile widths determine SBUF/PSUM buffer shape; each reader slices
    into those buffers using its own per-op tile widths.
    """
    from nkigym.codegen.render import render

    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    source = render(km)

    """SBUF intermediates lhs_sbuf/rhs_sbuf: writer is NKILoad with full-F
    extent (2048). Unbounded F yields a single (innermost-elided) iter-var,
    so the F slice renders as ``0 : 0 + 2048``. Matmul readers still slice
    with their own M/N tile widths via bounded iter-vars."""
    assert "lhs_sbuf[0:128, i_d0_0, 0 : 0 + 2048]" in source
    assert "rhs_sbuf[0:128, i_d0_0, 0 : 0 + 2048]" in source

    """NKIMatmul reads (M=128, N=512) — slices into the same SBUF tensors
    with its own tile widths."""
    assert "lhs_sbuf[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128]" in source
    assert "rhs_sbuf[0:128, i_d0_0, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source

    """NKIMatmul writes psum_acc (M=128, N=512)."""
    assert "psum_acc[0:128, i_d1_0, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source

    """NKIMemset writes psum_acc with its own (P=128, F=full) tiles — no
    F limit on NKIMemset. Unbounded F renders as ``0 : 0 + 2048``."""
    assert "psum_acc[0:128, i_d1_0, 0 : 0 + 2048]" in source

    """NKITensorCopy (PSUM drain → SBUF) has no F limit — writes sbuf_prod
    full-F and reads psum_acc full-F."""
    assert "sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048]" in source

    """HBM param / return slices follow the touching op's per-op tile.
    Loads scale HBM lhs/rhs by (P=128, F=2048 unbounded)."""
    assert "lhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, 0 : 0 + 2048]" in source
    assert "rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, 0 : 0 + 2048]" in source

    """NKIStore has no F limit — stores hbm_out full-F."""
    assert "hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, 0 : 0 + 2048]" in source

    """Regression guard."""
    assert "lhs_sbuf[i_d0_0 : i_d0_0 + 128" not in source
    assert "psum_acc[i_d1_0 : i_d1_0 + 128" not in source


def test_canonical_load_has_full_F_extent() -> None:
    """NKILoad has no F-axis TILE_LIMIT — canonical must tile F into a
    single whole-extent iteration, not carry NKIMatmul's M=128 through
    shared dim_ids. Regression: the old _derive_dims took a global min
    over ops touching the shared dim.

    Bounded P axis yields an outer+inner pair; unbounded F axis yields a
    single tile iter-var (extent = full axis). Total = 3 iter-vars ordered
    ``[P_outer, P_inner, F_tile]``.
    """
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    load_blocks = [b for b in _collect_sblocks(km) if b.body and b.body[0].op_cls is NKILoad]
    assert len(load_blocks) == 2
    for block in load_blocks:
        p_outer, p_inner, f_tile = block.iter_vars
        assert p_outer.extent == 16, f"expected P trip=16 (2048/128), got {p_outer.extent}"
        assert p_inner.extent == 128, f"expected P tile=128, got {p_inner.extent}"
        assert f_tile.extent == 2048, f"expected F single iter-var extent=2048, got {f_tile.extent}"
        f_ar = list(block.writes.values())[0].pattern[-1]
        assert f_ar.extent == 2048, f"expected F-tile=2048, got {f_ar.extent}"


def test_canonical_memset_has_full_F_extent() -> None:
    """NKIMemset has only P=128 in MAX_TILE_SIZE — its F iter-var must be
    full-extent under per-op tiling, independent of NKIMatmul's N=512.
    Guards against any cross-op coupling re-introduced via the
    memset-on-matmul-PSUM path.

    Bounded P yields outer+inner; unbounded F yields a single tile iter-var.
    """
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    memset_blocks = [b for b in _collect_sblocks(km) if b.body and b.body[0].op_cls is NKIMemset]
    assert len(memset_blocks) == 1
    block = memset_blocks[0]
    p_outer, p_inner, f_tile = block.iter_vars
    assert p_outer.extent == 16, f"expected P trip=16, got {p_outer.extent}"
    assert p_inner.extent == 128, f"expected P tile=128, got {p_inner.extent}"
    assert f_tile.extent == 2048, f"expected F single iter-var extent=2048, got {f_tile.extent}"
    f_ar = list(block.writes.values())[0].pattern[-1]
    assert f_ar.extent == 2048, f"expected F-tile=2048, got {f_ar.extent}"


def test_canonical_tensor_copy_has_full_F_extent() -> None:
    """NKITensorCopy (PSUM drain → SBUF) has only P=128 in MAX_TILE_SIZE —
    its F iter-var must stay full-extent. Guards against coupling via the
    matmul producer of psum_acc.

    Bounded P yields outer+inner; unbounded F yields a single tile iter-var.
    """
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    copy_blocks = [b for b in _collect_sblocks(km) if b.body and b.body[0].op_cls is NKITensorCopy]
    assert len(copy_blocks) == 1
    block = copy_blocks[0]
    p_outer, p_inner, f_tile = block.iter_vars
    assert p_outer.extent == 16, f"expected P trip=16, got {p_outer.extent}"
    assert p_inner.extent == 128, f"expected P tile=128, got {p_inner.extent}"
    assert f_tile.extent == 2048, f"expected F single iter-var extent=2048, got {f_tile.extent}"
    f_ar = list(block.writes.values())[0].pattern[-1]
    assert f_ar.extent == 2048, f"expected F-tile=2048, got {f_ar.extent}"


def test_canonical_store_has_full_F_extent() -> None:
    """NKIStore has only P=128 in MAX_TILE_SIZE — its F iter-var must be
    full-extent. Guards against coupling via the matmul producer chain.

    Bounded P yields outer+inner; unbounded F yields a single tile iter-var.
    """
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    store_blocks = [b for b in _collect_sblocks(km) if b.body and b.body[0].op_cls is NKIStore]
    assert len(store_blocks) == 1
    block = store_blocks[0]
    p_outer, p_inner, f_tile = block.iter_vars
    assert p_outer.extent == 16, f"expected P trip=16, got {p_outer.extent}"
    assert p_inner.extent == 128, f"expected P tile=128, got {p_inner.extent}"
    assert f_tile.extent == 2048, f"expected F single iter-var extent=2048, got {f_tile.extent}"
    f_ar = list(block.writes.values())[0].pattern[-1]
    assert f_ar.extent == 2048, f"expected F-tile=2048, got {f_ar.extent}"


def test_canonical_uses_max_tile_size() -> None:
    """Canonical tile sizes equal MAX_TILE_SIZE when declared, full extent otherwise.

    Post-Task 4: each dim yields an outer+inner iter-var pair, so a matmul
    (3 dims) has 6 iter-vars ordered ``[K_outer, K_inner, M_outer, M_inner,
    N_outer, N_inner]``.
    """
    km = build_canonical_module(_matmul_k, _INPUT_SPECS)
    matmul_blocks = [b for b in _collect_sblocks(km) if b.body and b.body[0].op_cls is NKIMatmul]
    assert len(matmul_blocks) == 1
    block = matmul_blocks[0]
    assert len(block.iter_vars) == 6
    k_outer, k_inner, m_outer, m_inner, n_outer, n_inner = block.iter_vars
    """K=128, M=128, N=512 from MAX_TILE_SIZE.
    Total dims are 2048 each, so outer trips are 16, 16, 4 and inner
    tiles are 128, 128, 512."""
    assert k_outer.extent == 16, f"K trip: expected 2048//128=16, got {k_outer.extent}"
    assert k_inner.extent == 128, f"K tile: expected 128, got {k_inner.extent}"
    assert m_outer.extent == 16, f"M trip: expected 2048//128=16, got {m_outer.extent}"
    assert m_inner.extent == 128, f"M tile: expected 128, got {m_inner.extent}"
    assert n_outer.extent == 4, f"N trip: expected 2048//512=4, got {n_outer.extent}"
    assert n_inner.extent == 512, f"N tile: expected 512, got {n_inner.extent}"
    """Access patterns should reflect the tile sizes."""
    call = block.body[0]
    assert isinstance(call, NKIOpCall)
    """Stationary operand (lhs_sbuf) has axes K, M."""
    stat = block.reads["stationary"]
    assert len(stat.pattern) == 2
    assert stat.pattern[0].extent == 128, f"K tile: expected 128, got {stat.pattern[0].extent}"
    assert stat.pattern[1].extent == 128, f"M tile: expected 128, got {stat.pattern[1].extent}"
    """Moving operand (rhs_sbuf) has axes K, N."""
    mov = block.reads["moving"]
    assert len(mov.pattern) == 2
    assert mov.pattern[0].extent == 128, f"K tile: expected 128, got {mov.pattern[0].extent}"
    assert mov.pattern[1].extent == 512, f"N tile: expected 512, got {mov.pattern[1].extent}"
    """Dst operand (psum_acc) has axes M, N."""
    dst = block.reads_writes["dst"]
    assert len(dst.pattern) == 2
    assert dst.pattern[0].extent == 128, f"M tile: expected 128, got {dst.pattern[0].extent}"
    assert dst.pattern[1].extent == 512, f"N tile: expected 512, got {dst.pattern[1].extent}"


def test_derive_op_tiles_raises_on_indivisible_extent() -> None:
    """Canonical build raises ValueError when axis extent is not divisible by MAX_TILE_SIZE.

    Matmul M has MAX_TILE_SIZE=128; an M-axis extent of 200 is indivisible.
    """
    import pytest

    @nkigym_kernel
    def _k(lhs_T, rhs):
        lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 200), dtype="bfloat16")()
        rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        psum = NKIAlloc(location="psum", shape=(200, 2048), dtype="float32")()
        hbm_out = NKIAlloc(location="hbm", shape=(200, 2048), dtype="bfloat16")()
        NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
        NKILoad()(src=rhs, dst=rhs_sbuf)
        NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum)
        NKIStore()(src=psum, dst=hbm_out)
        return hbm_out

    with pytest.raises(ValueError, match="not divisible by MAX_TILE_SIZE"):
        build_canonical_module(
            _k,
            input_specs={
                "lhs_T": {"shape": (2048, 200), "dtype": "bfloat16"},
                "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
            },
        )


def test_canonical_emits_outer_and_inner_tile_loops():
    """Each op axis produces TWO nested ForNodes in the IR: outer trip + inner tile.

    The ``_matmul_k`` fixture has ops whose tiles produce both a 16-trip
    outer (K/M at 128 over 2048) and multiple 128-wide inner loops (the
    K and M tiles of matmul, and all P tiles across the rest of the ops),
    so the asserted shape is reachable.
    """
    module = build_canonical_module(_matmul_k, _INPUT_SPECS)
    """Walk the body. After Task 4 each op axis yields a nested (outer,
    inner) ForNode pair. For NKIMatmul: K: trip=16 tile=128,
    M: trip=16 tile=128, N: trip=4 tile=512."""

    def collect_loops(nodes, found):
        for n in nodes:
            if isinstance(n, ForNode):
                found.append((module.axes[n.iter_var.axis_id].name, n.iter_var.extent))
                collect_loops(n.children, found)

    found = []
    collect_loops(module.body, found)
    assert any(extent == 128 for _, extent in found), f"expected an inner tile loop with extent 128; got {found}"
    assert any(extent == 16 for _, extent in found), f"expected an outer trip loop with extent 16; got {found}"
    inner_128_parents = [(dim, ext) for dim, ext in found if ext == 128]
    assert len(inner_128_parents) >= 2, f"expected >=2 inner-128 loops; got {inner_128_parents}"
