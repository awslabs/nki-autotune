"""Tests for nkigym.transforms.BufferLayout (tile-axis re-factorization)."""

from __future__ import annotations

from test.transforms._fixtures import INPUT_SPECS, f_matmul

import pytest

from nkigym.ir import build_initial_ir
from nkigym.transforms import BufferLayout, BufferLayoutOption, TransformLegalityError


def _canonical_ir():
    return build_initial_ir(f_matmul, INPUT_SPECS)


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


def _check_analyze_uses_logical_tiles_for_versioned_buffer():
    """Versions do not create additional BufferLayout divisors."""
    ir = _canonical_ir()
    object.__setattr__(ir.buffer("psum_prod"), "versions", 2)
    opts = [o for o in BufferLayout().analyze(ir) if o.tensor == "psum_prod"]
    assert {o.list_len for o in opts} == _divisors(16) - {1}
    assert all(option.list_len != 32 for option in opts)


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


def test_buffer_layout_analysis_contract() -> None:
    """Canonical analysis enumerates valid divisors and excludes HBM buffers."""
    _check_canonical_psum_is_packed_t16()
    _check_analyze_enumerates_every_divisor_of_t()
    _check_analyze_uses_logical_tiles_for_versioned_buffer()
    _check_analyze_skips_shared_hbm()


def test_buffer_layout_apply_contract() -> None:
    """Applying layouts is pure and conserves logical tiles."""
    _check_apply_sets_list_len_full_split()
    _check_apply_conserves_total_tiles()


def test_buffer_layout_rejects_invalid_options() -> None:
    """Missing tensors, non-divisors, and no-op layouts fail loudly."""
    _check_apply_rejects_missing_tensor()
    _check_apply_rejects_non_divisor()
    _check_apply_rejects_noop()
