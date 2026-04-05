"""Tests for the eager mode kernel generator (design doc section 3).

Verifies tracing, dimension unification, kernel generation with
6D tensor shapes, and simulation correctness for causal attention.
"""

import numpy as np
from golden.eager_data import (
    KERNEL_CONTAINS_1,
    KERNEL_DIM_COMMENT_1,
    KERNEL_SHAPE_DMA_Q_1,
    KERNEL_SHAPE_DMA_V_1,
    KERNEL_SHAPE_K_T_1,
    KERNEL_SHAPE_PSUM_S_1,
    KERNEL_SHAPE_Q_T_1,
    KERNEL_SHAPE_S_1,
    TRACE_DIM_IDS_1,
    TRACE_DIM_NUM_BLOCKS_1,
    TRACE_DIM_TILE_SIZES_1,
    TRACE_DIM_TOTAL_SIZES_1,
    TRACE_NUM_OPS_1,
    TRACE_OP_NAMES_1,
)

import nkigym
from nkigym.codegen.eager import EagerTracer, generate_eager_kernel
from nkigym.simulate import simulate_kernel

Q_256 = np.random.RandomState(42).randn(256, 128)
K_256 = np.random.RandomState(43).randn(256, 128)
V_256 = np.random.RandomState(44).randn(256, 128)
SCALE = 1.0 / np.sqrt(128)


def _trace_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> tuple[np.ndarray, EagerTracer]:
    """Trace the attention math function and return result + tracer."""
    tracer = EagerTracer()
    nkigym.reset_array_names()
    nkigym.set_tracer(tracer)
    try:
        tracer.register_input("Q", q.shape)
        tracer.register_input("K", k.shape)
        tracer.register_input("V", v.shape)
        nkigym._register_array(q, "Q")
        nkigym._register_array(k, "K")
        nkigym._register_array(v, "V")

        q_t = nkigym.nc_transpose(q)
        k_t = nkigym.nc_transpose(k)
        s = nkigym.nc_matmul(q_t, k_t)
        masked_s = nkigym.affine_select(
            s, pattern=[[-1, s.shape[1]]], channel_multiplier=1, on_false_value=float("-inf"), cmp_op="greater_equal"
        )
        scaled_s = nkigym.tensor_scalar(masked_s, scale, op0="multiply")
        neg_max_s = nkigym.tensor_reduce(scaled_s, op="max", negate=True)
        exp_s, sum_exp = nkigym.activation_reduce(scaled_s, op="exp", reduce_op="add", bias=neg_max_s)
        inv_sum = nkigym.activation(sum_exp, op="reciprocal")
        exp_s_t = nkigym.nc_transpose(exp_s)
        attn = nkigym.nc_matmul(exp_s_t, v)
        output = nkigym.tensor_scalar(attn, inv_sum, op0="multiply")
        return output, tracer
    finally:
        nkigym.set_tracer(None)


REFERENCE_256, TRACER_256 = _trace_attention(Q_256, K_256, V_256, SCALE)
KERNEL_SRC_256 = generate_eager_kernel(TRACER_256, "attention_kernel", scale_param="scale")


def test_trace_captures_all_ops() -> None:
    """Tracing attention captures exactly 11 ops in math function order."""
    assert len(TRACER_256.ops) == TRACE_NUM_OPS_1


def test_trace_op_names() -> None:
    """Traced ops match expected ISA operation names in order."""
    actual = [op.op.NAME for op in TRACER_256.ops]
    assert actual == TRACE_OP_NAMES_1


def test_dim_unification_ids() -> None:
    """Dimension unification produces exactly d0, d1, d2, d5."""
    assert set(TRACER_256.dims.keys()) == TRACE_DIM_IDS_1


def test_dim_total_sizes() -> None:
    """Each dimension has the correct total element count."""
    actual = {d: info.total_size for d, info in TRACER_256.dims.items()}
    assert actual == TRACE_DIM_TOTAL_SIZES_1


def test_dim_tile_sizes() -> None:
    """Tile sizes are max(all op limits) per dimension."""
    actual = {d: info.tile_size for d, info in TRACER_256.dims.items()}
    assert actual == TRACE_DIM_TILE_SIZES_1


def test_dim_num_blocks() -> None:
    """num_blocks = total_size / tile_size for each dimension."""
    actual = {d: info.num_blocks for d, info in TRACER_256.dims.items()}
    assert actual == TRACE_DIM_NUM_BLOCKS_1


def test_kernel_contains_required_constructs() -> None:
    """Generated kernel includes all expected NKI constructs."""
    for fragment in KERNEL_CONTAINS_1:
        assert fragment in KERNEL_SRC_256, f"Missing: {fragment!r}"


def test_kernel_6d_shapes() -> None:
    """Tensor allocations use 6D shapes with all nb/tpb entries."""
    assert KERNEL_SHAPE_Q_T_1 in KERNEL_SRC_256
    assert KERNEL_SHAPE_K_T_1 in KERNEL_SRC_256
    assert KERNEL_SHAPE_S_1 in KERNEL_SRC_256
    assert KERNEL_SHAPE_PSUM_S_1 in KERNEL_SRC_256
    assert KERNEL_SHAPE_DMA_Q_1 in KERNEL_SRC_256
    assert KERNEL_SHAPE_DMA_V_1 in KERNEL_SRC_256


def test_kernel_dim_comment() -> None:
    """Kernel docstring includes dimension info in tile_size x num_blocks format."""
    assert KERNEL_DIM_COMMENT_1 in KERNEL_SRC_256


def test_kernel_simulates_correctly() -> None:
    """Eager kernel produces correct output when simulated at float64."""
    simulated = simulate_kernel(
        KERNEL_SRC_256, "attention_kernel", {"Q": Q_256, "K": K_256, "V": V_256, "scale": np.array(SCALE)}
    )
    np.testing.assert_allclose(simulated, REFERENCE_256, rtol=1e-6, atol=1e-6)
