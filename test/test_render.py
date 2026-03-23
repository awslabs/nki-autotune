"""Tests for render_schedule with hardcoded function + Schedule and expected NKI source."""

from golden.analyses import (
    ADD_ONLY_ANALYSIS,
    ADD_ONLY_OP_CALLS,
    ADD_ONLY_PARAMS,
    MATMUL_256_ANALYSIS,
    MATMUL_256_OP_CALLS,
    MATMUL_256_PARAMS,
    MATMUL_ADD_ANALYSIS,
    MATMUL_ADD_OP_CALLS,
    MATMUL_ADD_PARAMS,
    MATMUL_TANH_ANALYSIS,
    MATMUL_TANH_OP_CALLS,
    MATMUL_TANH_PARAMS,
)
from golden.render_data import (
    RENDER_1,
    RENDER_2,
    RENDER_3,
    RENDER_4,
    RENDER_5,
    RENDER_6,
    RENDER_SCHEDULE_4,
    RENDER_SCHEDULE_6,
)
from golden.schedules import ADD_ONLY_DEFAULT, MATMUL_256_DEFAULT, MATMUL_ADD_DEFAULT, MATMUL_TANH_DEFAULT

from nkigym.schedule.render import render_schedule


def test_render_simple_matmul() -> None:
    """Render 256x256 matmul with default schedule."""
    result = render_schedule(
        MATMUL_256_ANALYSIS, MATMUL_256_DEFAULT, MATMUL_256_OP_CALLS, MATMUL_256_PARAMS, "matmul_kernel"
    )
    assert result == RENDER_1


def test_render_matmul_tanh() -> None:
    """Render matmul+tanh with default schedule and post-compute activation."""
    result = render_schedule(
        MATMUL_TANH_ANALYSIS, MATMUL_TANH_DEFAULT, MATMUL_TANH_OP_CALLS, MATMUL_TANH_PARAMS, "matmul_tanh_kernel"
    )
    assert result == RENDER_2


def test_render_add_only() -> None:
    """Render element-wise add with no reduction ops."""
    result = render_schedule(ADD_ONLY_ANALYSIS, ADD_ONLY_DEFAULT, ADD_ONLY_OP_CALLS, ADD_ONLY_PARAMS, "add_kernel")
    assert result == RENDER_3


def test_render_matmul_swapped() -> None:
    """Render matmul with swapped parallel dims changes HBM index variables."""
    result = render_schedule(
        MATMUL_256_ANALYSIS, RENDER_SCHEDULE_4, MATMUL_256_OP_CALLS, MATMUL_256_PARAMS, "matmul_swapped_kernel"
    )
    assert result == RENDER_4


def test_render_matmul_add() -> None:
    """Render matmul+add with bias load at reduction level."""
    result = render_schedule(
        MATMUL_ADD_ANALYSIS, MATMUL_ADD_DEFAULT, MATMUL_ADD_OP_CALLS, MATMUL_ADD_PARAMS, "matmul_add_kernel"
    )
    assert result == RENDER_5


def test_render_reduction_middle() -> None:
    """Render matmul with reduction dim in middle of loop nest, not innermost."""
    result = render_schedule(
        MATMUL_256_ANALYSIS, RENDER_SCHEDULE_6, MATMUL_256_OP_CALLS, MATMUL_256_PARAMS, "matmul_red_middle_kernel"
    )
    assert result == RENDER_6
