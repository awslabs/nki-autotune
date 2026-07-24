"""Byte-exact gates for transform recipes and explicit manual matmul ladders."""

from __future__ import annotations

import inspect
from test.transforms import _matmul_lhs_rhs_manual as lhs_manual_ladder
from test.transforms import _matmul_lhsT_rhs_manual as lhs_t_manual_ladder
from test.transforms._ladder_compare import assert_matches_render_ordered
from test.transforms._matmul_lhs_rhs_ladder import _build_ladder as build_lhs_ladder
from test.transforms._matmul_lhsT_rhs_ladder import _build_ladder as build_lhs_t_ladder
from types import ModuleType

from nkigym.codegen import render
from nkigym.ir import KernelIR


def _assert_ladder_matches(ladder: list[tuple[str, KernelIR]], manual_ladder: ModuleType) -> None:
    """Assert every transform-produced rung matches its explicit NKI kernel."""
    for name, ir in ladder:
        hand_kernel = getattr(manual_ladder, name)
        try:
            assert_matches_render_ordered(render(ir), inspect.getsource(hand_kernel))
        except AssertionError as error:
            raise AssertionError(f"{manual_ladder.__name__}.{name}") from error


def test_lhs_t_transform_ladder_is_byte_exact_with_manual_ladder() -> None:
    """All 36 ``lhs_T.T @ rhs`` transform states match the hand ladder."""
    ladder = build_lhs_t_ladder()
    assert [name for name, _ir in ladder] == [f"kernel_{index}" for index in range(36)]
    _assert_ladder_matches(ladder, lhs_t_manual_ladder)


def test_lhs_transform_ladder_is_byte_exact_with_manual_ladder() -> None:
    """All 32 ``lhs @ rhs`` transform states match the hand ladder."""
    ladder = build_lhs_ladder()
    assert [name for name, _ir in ladder] == [f"kernel_{index}" for index in range(32)]
    _assert_ladder_matches(ladder, lhs_manual_ladder)
