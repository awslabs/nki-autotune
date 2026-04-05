"""Tests for matmul eager mode kernel generation (design doc section 2).

Verifies tracing, dimension unification, and kernel generation
for a single nc_matmul op with 2048x2048 inputs.
"""

import numpy as np
from golden.matmul_eager_data import (
    KERNEL_CONTAINS_1,
    KERNEL_DIM_COMMENT_1,
    KERNEL_DMA_LOAD_A_1,
    KERNEL_DMA_LOAD_B_1,
    KERNEL_DMA_STORE_1,
    KERNEL_ISA_CALL_1,
    KERNEL_OUTPUT_LOOP_D1_1,
    KERNEL_OUTPUT_LOOP_D3_1,
    KERNEL_REDUCTION_LOOP_D0_1,
    KERNEL_SHAPE_DMA_A_1,
    KERNEL_SHAPE_DMA_B_1,
    KERNEL_SHAPE_PSUM_OUTPUT_1,
    KERNEL_SHAPE_SBUF_OUTPUT_1,
    KERNEL_TENSOR_COPY_1,
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


def _trace_matmul(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, EagerTracer]:
    """Trace a single matmul and return result + tracer."""
    tracer = EagerTracer()
    nkigym.reset_array_names()
    nkigym.set_tracer(tracer)
    try:
        tracer.register_input("a", a.shape)
        tracer.register_input("b", b.shape)
        nkigym._register_array(a, "a")
        nkigym._register_array(b, "b")
        output = nkigym.nc_matmul(a, b)
        return output, tracer
    finally:
        nkigym.set_tracer(None)


A_2048 = np.random.RandomState(42).randn(2048, 2048).astype(np.float16)
B_2048 = np.random.RandomState(43).randn(2048, 2048).astype(np.float16)
REFERENCE_2048, TRACER_2048 = _trace_matmul(A_2048, B_2048)
KERNEL_SRC_2048 = generate_eager_kernel(TRACER_2048, "matmul_kernel", scale_param="")


def test_trace_captures_single_op() -> None:
    """Tracing matmul captures exactly 1 op."""
    assert len(TRACER_2048.ops) == TRACE_NUM_OPS_1


def test_trace_op_names() -> None:
    """Traced op is nc_matmul."""
    actual = [op.op.NAME for op in TRACER_2048.ops]
    assert actual == TRACE_OP_NAMES_1


def test_dim_unification_ids() -> None:
    """Dimension unification produces d0, d1, d3."""
    assert set(TRACER_2048.dims.keys()) == TRACE_DIM_IDS_1


def test_dim_total_sizes() -> None:
    """Each dimension has total size 2048."""
    actual = {d: info.total_size for d, info in TRACER_2048.dims.items()}
    assert actual == TRACE_DIM_TOTAL_SIZES_1


def test_dim_tile_sizes() -> None:
    """K and M tile at 128, N tiles at 512."""
    actual = {d: info.tile_size for d, info in TRACER_2048.dims.items()}
    assert actual == TRACE_DIM_TILE_SIZES_1


def test_dim_num_blocks() -> None:
    """num_blocks = total_size / tile_size for each dimension."""
    actual = {d: info.num_blocks for d, info in TRACER_2048.dims.items()}
    assert actual == TRACE_DIM_NUM_BLOCKS_1


def test_kernel_contains_required_constructs() -> None:
    """Generated kernel includes all expected NKI constructs."""
    for fragment in KERNEL_CONTAINS_1:
        assert fragment in KERNEL_SRC_2048, f"Missing: {fragment!r}"


def test_kernel_6d_shapes() -> None:
    """Tensor allocations use 6D shapes with correct sizes."""
    assert KERNEL_SHAPE_SBUF_OUTPUT_1 in KERNEL_SRC_2048, "sbuf_output shape mismatch"
    assert KERNEL_SHAPE_PSUM_OUTPUT_1 in KERNEL_SRC_2048, "psum_output shape mismatch"
    assert KERNEL_SHAPE_DMA_A_1 in KERNEL_SRC_2048, "sbuf_a shape mismatch"
    assert KERNEL_SHAPE_DMA_B_1 in KERNEL_SRC_2048, "sbuf_b shape mismatch"


def test_kernel_dim_comment() -> None:
    """Kernel docstring includes dimension info."""
    assert KERNEL_DIM_COMMENT_1 in KERNEL_SRC_2048


def test_kernel_output_loops() -> None:
    """Output loops iterate over d1 and d3 blocks."""
    assert KERNEL_OUTPUT_LOOP_D1_1 in KERNEL_SRC_2048
    assert KERNEL_OUTPUT_LOOP_D3_1 in KERNEL_SRC_2048


def test_kernel_reduction_loop() -> None:
    """Reduction loop iterates over d0 blocks."""
    assert KERNEL_REDUCTION_LOOP_D0_1 in KERNEL_SRC_2048


def test_kernel_isa_call() -> None:
    """ISA call uses 6D slicing on PSUM destination."""
    assert KERNEL_ISA_CALL_1 in KERNEL_SRC_2048


def test_kernel_tensor_copy() -> None:
    """tensor_copy moves PSUM to SBUF with block indices."""
    assert KERNEL_TENSOR_COPY_1 in KERNEL_SRC_2048


def test_kernel_dma_store() -> None:
    """DMA store writes SBUF to HBM output with block offsets."""
    assert KERNEL_DMA_STORE_1 in KERNEL_SRC_2048


def test_kernel_dma_load_a() -> None:
    """DMA load for stationary operand with correct slicing."""
    assert KERNEL_DMA_LOAD_A_1 in KERNEL_SRC_2048


def test_kernel_dma_load_b() -> None:
    """DMA load for moving operand with correct slicing."""
    assert KERNEL_DMA_LOAD_B_1 in KERNEL_SRC_2048


def test_kernel_simulates_correctly() -> None:
    """Eager kernel produces correct output when simulated."""
    simulated = simulate_kernel(KERNEL_SRC_2048, "matmul_kernel", {"a": A_2048, "b": B_2048})
    np.testing.assert_allclose(simulated, REFERENCE_2048, rtol=1e-2, atol=1e-2)
