"""Tests for rmsnorm+matmul multi-pass reduction pipeline."""

import numpy as np
from golden.analyses import RMSNORM_MATMUL_ANALYSIS, RMSNORM_MATMUL_OP_CALLS, RMSNORM_MATMUL_PARAMS
from golden.render_data import RENDER_7
from golden.schedules import RMSNORM_MATMUL_DEFAULT

from nkigym.codegen.analysis import analyze_dims
from nkigym.codegen.parse import find_func_def, parse_body
from nkigym.codegen.passes import assign_passes
from nkigym.ops.activation_1d import NKIActivation1D
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_scalar_const import NKITensorScalarConst
from nkigym.ops.transpose import NKITranspose
from nkigym.schedule.enumerate import default_schedule, enumerate_all, enumerate_loop_orders
from nkigym.schedule.render import render_schedule
from nkigym.simulate import simulate_kernel

_RMSNORM_SOURCE = """\
import numpy as np
import nkigym

def rmsnorm_matmul(a, b):
    sum_sq = nkigym.activation(a, op="square", reduce_op=np.add)
    scaled = nkigym.tensor_scalar(sum_sq, op0=np.multiply, operand0=1/2048, op1=np.add, operand1=1e-6)
    rsqrt_val = nkigym.activation(scaled, op="rsqrt")
    a_normed = nkigym.tensor_scalar(a, rsqrt_val, op0=np.multiply)
    a_t = nkigym.transpose(a_normed)
    result = nkigym.nc_matmul(a_t, b)
    return result
"""


def test_parse_op_types() -> None:
    """Parse rmsnorm_matmul produces 6 ops with correct types."""
    func_def = find_func_def(_RMSNORM_SOURCE)
    op_calls = parse_body(func_def)
    assert len(op_calls) == 6
    assert op_calls[0].stmt_type is NKIActivationReduce
    assert op_calls[1].stmt_type is NKITensorScalarConst
    assert op_calls[2].stmt_type is NKIActivation1D
    assert op_calls[3].stmt_type is NKITensorScalar
    assert op_calls[4].stmt_type is NKITranspose
    assert op_calls[5].stmt_type is NKIMatmul


def test_parse_output_vars() -> None:
    """Parse rmsnorm_matmul assigns correct SSA variable names."""
    func_def = find_func_def(_RMSNORM_SOURCE)
    op_calls = parse_body(func_def)
    output_vars = [op.output_var for op in op_calls]
    assert output_vars == ["sum_sq", "scaled", "rsqrt_val", "a_normed", "a_t", "result"]


def test_analysis_dims() -> None:
    """Dimension analysis: d0=M parallel, d1=K reduction, d3=N parallel."""
    func_def = find_func_def(_RMSNORM_SOURCE)
    op_calls = parse_body(func_def)
    analysis = analyze_dims(op_calls, RMSNORM_MATMUL_PARAMS, ((256, 256), (256, 256)))
    assert analysis.parallel_dims == ["d0", "d3"]
    assert analysis.reduction_dims == ["d1"]
    assert analysis.tile_counts == {"d0": 2, "d3": 1}
    assert analysis.reduction_tile_counts == {"d1": 2}
    assert analysis.dim_tile_sizes == {"d0": 128, "d1": 128, "d3": 256}


def test_analysis_var_dims() -> None:
    """Variable dims match golden data including transpose swap."""
    func_def = find_func_def(_RMSNORM_SOURCE)
    op_calls = parse_body(func_def)
    analysis = analyze_dims(op_calls, RMSNORM_MATMUL_PARAMS, ((256, 256), (256, 256)))
    assert analysis.var_dims == RMSNORM_MATMUL_ANALYSIS.var_dims


def test_pass_assignment_barriers() -> None:
    """Two barrier ops on d1: activation_reduce (pass 0), nc_matmul (pass 1)."""
    pa = assign_passes(RMSNORM_MATMUL_OP_CALLS, RMSNORM_MATMUL_ANALYSIS)
    assert pa.passes_per_dim == {"d1": 2}
    assert pa.barrier_ops == [(0, "d1", 0), (5, "d1", 1)]


def test_pass_assignment_classification() -> None:
    """Non-barrier ops classified: inter-pass 1D ops and pre-compute 2D ops."""
    pa = assign_passes(RMSNORM_MATMUL_OP_CALLS, RMSNORM_MATMUL_ANALYSIS)
    assert pa.inter_pass == {("d1", 0): [1, 2]}
    assert pa.pre_compute == {("d1", 1): [3, 4]}
    assert pa.post_compute == []


def test_enumerate_loop_orders() -> None:
    """4 items (d0, d3, d1_0, d1_1) with d1 passes ordered produce 12 orderings."""
    orders = enumerate_loop_orders(RMSNORM_MATMUL_ANALYSIS, RMSNORM_MATMUL_OP_CALLS, {"d1": 2})
    assert len(orders) == 12


def test_default_schedule() -> None:
    """Default schedule: parallel dims first, then d1 passes, tpb=1."""
    ds = default_schedule(RMSNORM_MATMUL_ANALYSIS, RMSNORM_MATMUL_OP_CALLS, RMSNORM_MATMUL_PARAMS, {"d1": 2})
    assert ds == RMSNORM_MATMUL_DEFAULT


def test_enumerate_all_count() -> None:
    """Enumeration produces 432 valid schedules for 256x256 rmsnorm_matmul."""
    schedules = enumerate_all(RMSNORM_MATMUL_ANALYSIS, RMSNORM_MATMUL_OP_CALLS, RMSNORM_MATMUL_PARAMS, {"d1": 2})
    assert len(schedules) == 432
    assert RMSNORM_MATMUL_DEFAULT in schedules


def test_render_default() -> None:
    """Render default schedule matches golden kernel string."""
    pa = assign_passes(RMSNORM_MATMUL_OP_CALLS, RMSNORM_MATMUL_ANALYSIS)
    result = render_schedule(
        RMSNORM_MATMUL_ANALYSIS,
        RMSNORM_MATMUL_DEFAULT,
        RMSNORM_MATMUL_OP_CALLS,
        RMSNORM_MATMUL_PARAMS,
        "rmsnorm_matmul",
        pa,
    )
    assert result == RENDER_7


def _rmsnorm_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute rmsnorm_matmul reference at float64."""
    sum_sq = np.sum(a**2, axis=-1)
    scaled = sum_sq * 0.00048828125 + 1e-6
    rsqrt = 1.0 / np.sqrt(scaled)
    a_normed = a * rsqrt[:, np.newaxis]
    return a_normed @ b


def test_simulate_default() -> None:
    """CPU simulation of default schedule matches numpy reference."""
    rng = np.random.RandomState(42)
    a = rng.randn(256, 256)
    b = rng.randn(256, 256)
    expected = _rmsnorm_reference(a, b)
    actual = simulate_kernel(RENDER_7, "rmsnorm_matmul", {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_simulate_rendered() -> None:
    """Render then simulate: full pipeline produces correct output."""
    pa = assign_passes(RMSNORM_MATMUL_OP_CALLS, RMSNORM_MATMUL_ANALYSIS)
    nki_source = render_schedule(
        RMSNORM_MATMUL_ANALYSIS,
        RMSNORM_MATMUL_DEFAULT,
        RMSNORM_MATMUL_OP_CALLS,
        RMSNORM_MATMUL_PARAMS,
        "rmsnorm_matmul",
        pa,
    )
    rng = np.random.RandomState(99)
    a = rng.randn(256, 256)
    b = rng.randn(256, 256)
    expected = _rmsnorm_reference(a, b)
    actual = simulate_kernel(nki_source, "rmsnorm_matmul", {"a": a, "b": b})
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
