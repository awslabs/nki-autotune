"""Tests for nkigym.transforms.BufferLayout (tile-axis re-factorization)."""

from __future__ import annotations

from test.transforms import _matmul_lhsT_rhs_manual as manual_ladder
from test.transforms._fixtures import INPUT_SPECS, f_matmul
from test.transforms._helpers import leaf_for_op, matmul_loop
from test.transforms._ladder_compare import assert_matches_hand
from test.transforms._matmul_lhsT_rhs_ladder import _psum_memset_leaf

import pytest

from nkigym.codegen import render
from nkigym.codegen.compact import compact_shapes
from nkigym.ir import build_initial_ir
from nkigym.ir.buffer_placement import place_buffers
from nkigym.transforms import (
    BufferLayout,
    BufferLayoutOption,
    Reorder,
    ReorderOption,
    Split,
    SplitOption,
    TransformLegalityError,
)


def _canonical_ir():
    return build_initial_ir(f_matmul, INPUT_SPECS)


def _ir_at_manual_k6():
    """Drive canonical -> the manual-k6 packed nest (N > ko > Mo > Mi > ki).

    Reuses the verified pre-BufferLayout prefix from the test-only driven
    ladder: two atomic Reorders bubbling N outermost, Split K, Split M, then
    two Reorders. ``psum_prod`` is still packed ``(128, 16, 2048)`` here.
    """
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    ir = Reorder().apply(ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d1_0"), inner_nid=matmul_loop(ir, "i_d2_0")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_0"), inner_nid=matmul_loop(ir, "i_d2_0")))
    ir = Split().apply(ir, SplitOption(target_nid=matmul_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None))
    ir = Split().apply(ir, SplitOption(target_nid=matmul_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_1"), inner_nid=matmul_loop(ir, "i_d1_0")))
    ir = Reorder().apply(ir, ReorderOption(outer_nid=matmul_loop(ir, "i_d0_1"), inner_nid=matmul_loop(ir, "i_d1_1")))
    return ir


def _tile_count(ir, name):
    return ir.buffer(name).physical_shape()[1]


def _divisors(n):
    return {d for d in range(1, n + 1) if n % d == 0}


def _check_canonical_psum_is_packed_t16():
    """Guard: the canonical psum_prod is packed (128,16,2048) — T=16, list_len=1.

    The divisor-set / apply tests below assume T=16; this pins that assumption so a
    fixture change surfaces here, not as a confusing failure downstream."""
    ir = _canonical_ir()
    assert _tile_count(ir, "psum_prod") == 16
    assert ir.buffer("psum_prod").list_len == 1


def _check_analyze_enumerates_every_divisor_of_t():
    """A T=16 psum buffer offers list_len in divisors(16) minus its current layout (1)."""
    ir = _canonical_ir()
    opts = [o for o in BufferLayout().analyze(ir) if o.tensor == "psum_prod"]
    assert {o.list_len for o in opts} == _divisors(16) - {1}  # {2, 4, 8, 16}


def _check_analyze_skips_shared_hbm():
    """No option targets a shared_hbm buffer (no tile axis)."""
    ir = _canonical_ir()
    assert all(ir.buffer(o.tensor).location != "shared_hbm" for o in BufferLayout().analyze(ir))


def _check_apply_sets_list_len_full_split():
    """apply(psum_prod, 16) sets list_len=16; tree node count unchanged; original untouched."""
    ir = _canonical_ir()
    n_before = ir.tree.graph.number_of_nodes()
    new_ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    assert new_ir.buffer("psum_prod").list_len == 16
    assert ir.buffer("psum_prod").list_len == 1
    assert new_ir.tree.graph.number_of_nodes() == n_before


def _check_apply_conserves_total_tiles():
    """T (=list_len*a) is invariant across apply — re-factorize, never create."""
    ir = _canonical_ir()
    t_before = _tile_count(ir, "psum_prod")
    new_ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=4))
    b = new_ir.buffer("psum_prod")
    assert b.list_len * b.per_tile_physical_shape()[1] == t_before


def _check_apply_round_trip_identity():
    """list->pack->list returns to the same list_len."""
    ir = _canonical_ir()
    listed = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    packed = BufferLayout().apply(listed, BufferLayoutOption(tensor="psum_prod", list_len=1))
    assert packed.buffer("psum_prod").list_len == 1


def _check_apply_rejects_missing_tensor():
    ir = _canonical_ir()
    with pytest.raises(TransformLegalityError):
        BufferLayout().apply(ir, BufferLayoutOption(tensor="does_not_exist", list_len=2))


def _check_apply_rejects_non_divisor():
    """list_len must divide T; 3 does not divide 16."""
    ir = _canonical_ir()
    with pytest.raises(TransformLegalityError):
        BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=3))


def _check_apply_rejects_noop():
    """Setting list_len to its current value is rejected (no-op)."""
    ir = _canonical_ir()
    with pytest.raises(TransformLegalityError):
        BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=1))


def _check_prefix_reaches_manual_kernel_6():
    """Guard: the driven prefix renders byte-exact to manual kernel_6 (packed psum)."""
    assert_matches_hand(render(_ir_at_manual_k6()), manual_ladder.kernel_6)


def _check_k6_to_k7_reproduces_manual_kernel_7():
    """BufferLayout(psum_prod, 16) on the manual-k6 state renders byte-exact to manual
    kernel_7 — the standalone '# Buffer layout' rung (packed (128,16,2048) -> list-of-16)."""
    ir = _ir_at_manual_k6()
    ir = BufferLayout().apply(ir, BufferLayoutOption(tensor="psum_prod", list_len=16))
    assert_matches_hand(render(ir), manual_ladder.kernel_7)


def _check_list_buffer_idempotent_when_no_narrowing():
    """place_buffers + compact_shapes (the two passes CodeMotion reruns) are idempotent
    on a listed buffer whose touchers are NOT narrowed — list_len and per-tile survive.

    Neither pass had ever run on a list_len>1 buffer before this plan. This pins that
    they don't SPURIOUSLY shrink or drop list_len when nothing narrows (they replace
    whole Buffer objects, so list_len rides through)."""
    ir = BufferLayout().apply(_ir_at_manual_k6(), BufferLayoutOption(tensor="psum_prod", list_len=16))
    before = ir.buffer("psum_prod").per_tile_physical_shape()
    assert before == (128, 1, 2048)
    place_buffers(ir.tree)
    compact_shapes(ir.tree)
    buf = ir.buffer("psum_prod")
    assert buf.list_len == 16
    assert buf.per_tile_physical_shape() == before


def _check_compact_shapes_does_not_mis_shrink_list_tile_axis():
    """After tiling a listed psum's touchers on d2, compact_shapes leaves list_len=16 and
    the (128, 1, F) tile-axis form intact — it never collapses the LEADING/tile axis.

    Lists psum_prod, then Splits the memset + drain on d2 (4x512). Split makes a fresh
    INNER i_d2_1 loop, so each toucher still spans all 4 N-tiles (bbox free = 2048) — the
    free axis only shrinks to 512 once the touchers CO-LOCATE under the enclosing i_d2_0
    (the CodeMotion sink), which is verified byte-exact against the manual memset-sink rung
    in the driven manual ladder, NOT reproducible by compact_shapes alone. What THIS test
    pins is the composability SAFETY property: through Split + place + compact on a
    list_len>1 buffer, list_len stays 16 and per_tile_physical_shape never trips (a
    mis-shrunk leading axis would raise). Free stays 2048 here by construction."""
    ir = BufferLayout().apply(_ir_at_manual_k6(), BufferLayoutOption(tensor="psum_prod", list_len=16))
    ir = Split().apply(ir, SplitOption(target_nid=_psum_memset_leaf(ir), factors=(4, 512), target_axis="d2"))
    ir = Split().apply(ir, SplitOption(target_nid=leaf_for_op(ir, "NKITensorCopy"), factors=(4, 512), target_axis="d2"))
    place_buffers(ir.tree)
    compact_shapes(ir.tree)
    buf = ir.buffer("psum_prod")
    assert buf.list_len == 16
    assert buf.per_tile_physical_shape() == (128, 1, 2048)


def test_buffer_layout_analysis_contract() -> None:
    """Canonical analysis enumerates valid divisors and excludes HBM buffers."""
    _check_canonical_psum_is_packed_t16()
    _check_analyze_enumerates_every_divisor_of_t()
    _check_analyze_skips_shared_hbm()


def test_buffer_layout_apply_contract() -> None:
    """Applying layouts is pure, conserves tiles, and round-trips to packed form."""
    _check_apply_sets_list_len_full_split()
    _check_apply_conserves_total_tiles()
    _check_apply_round_trip_identity()


def test_buffer_layout_rejects_invalid_options() -> None:
    """Missing tensors, non-divisors, and no-op layouts fail loudly."""
    _check_apply_rejects_missing_tensor()
    _check_apply_rejects_non_divisor()
    _check_apply_rejects_noop()


def test_buffer_layout_matches_manual_ladder() -> None:
    """The packed prefix and list transition match their hand-written rungs."""
    _check_prefix_reaches_manual_kernel_6()
    _check_k6_to_k7_reproduces_manual_kernel_7()


def test_list_layout_survives_placement_and_compaction() -> None:
    """Placement and compaction preserve valid list geometry."""
    _check_list_buffer_idempotent_when_no_narrowing()
    _check_compact_shapes_does_not_mis_shrink_list_tile_axis()
