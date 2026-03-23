"""Tests for combinatorial schedule enumeration."""

from golden.analyses import ADD_ONLY_ANALYSIS, ADD_ONLY_OP_CALLS, MATMUL_256_ANALYSIS, MATMUL_256_OP_CALLS
from golden.enumerate_data import (
    ADD_ONLY_BLOCKINGS,
    ADD_ONLY_DEFAULT,
    ADD_ONLY_LOOP_ORDERS,
    MATMUL_256_BLOCKINGS,
    MATMUL_256_DEFAULT,
    MATMUL_256_LOOP_ORDERS,
    MATMUL_256_OP_PLACEMENTS,
)

from nkigym.schedule.enumerate import (
    default_schedule,
    enumerate_all,
    enumerate_blocking,
    enumerate_loop_orders,
    enumerate_op_placements,
)


def test_matmul_loop_orders() -> None:
    """256x256 matmul with 3 dims produces 6 permutations."""
    orders = enumerate_loop_orders(MATMUL_256_ANALYSIS, MATMUL_256_OP_CALLS)
    assert sorted(orders) == sorted(MATMUL_256_LOOP_ORDERS)


def test_matmul_op_placements() -> None:
    """Two params with 2 dependent dims each produce 9 placement combos."""
    placements = enumerate_op_placements(MATMUL_256_ANALYSIS, ("a", "b"))
    assert sorted(placements) == sorted(MATMUL_256_OP_PLACEMENTS)


def test_matmul_blocking() -> None:
    """d1 has 2 tiles, d3 has 1 tile, d0 has 2 tiles: 2*1*2 = 4 blockings."""
    blockings = enumerate_blocking(MATMUL_256_ANALYSIS)
    assert sorted(blockings) == sorted(MATMUL_256_BLOCKINGS)


def test_add_loop_orders() -> None:
    """Element-wise add with 2 parallel dims produces 2 permutations."""
    orders = enumerate_loop_orders(ADD_ONLY_ANALYSIS, ADD_ONLY_OP_CALLS)
    assert sorted(orders) == sorted(ADD_ONLY_LOOP_ORDERS)


def test_add_blocking() -> None:
    """Two parallel dims with 2 tiles each produce 4 blockings."""
    blockings = enumerate_blocking(ADD_ONLY_ANALYSIS)
    assert sorted(blockings) == sorted(ADD_ONLY_BLOCKINGS)


def test_matmul_default_schedule() -> None:
    """Default matmul schedule: parallel dims first, tpb=1, natural placements."""
    result = default_schedule(MATMUL_256_ANALYSIS, MATMUL_256_OP_CALLS, ("a", "b"))
    assert result == MATMUL_256_DEFAULT


def test_add_default_schedule() -> None:
    """Default add schedule: parallel dims first, tpb=1, natural placements."""
    result = default_schedule(ADD_ONLY_ANALYSIS, ADD_ONLY_OP_CALLS, ("x", "y"))
    assert result == ADD_ONLY_DEFAULT


def test_enumerate_all_produces_valid_schedules() -> None:
    """All enumerated matmul schedules pass validation (no duplicates)."""
    schedules = enumerate_all(MATMUL_256_ANALYSIS, MATMUL_256_OP_CALLS, ("a", "b"), 2)
    assert len(schedules) == len(set(schedules))
    assert len(schedules) > 0


def test_enumerate_all_default_included() -> None:
    """Default schedule is always in the enumerated set."""
    schedules = enumerate_all(MATMUL_256_ANALYSIS, MATMUL_256_OP_CALLS, ("a", "b"), 2)
    assert MATMUL_256_DEFAULT in schedules
