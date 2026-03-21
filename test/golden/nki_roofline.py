"""Golden test data for roofline analysis: NKIKernel objects with known metrics."""

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul

"""
ROOFLINE_MATMUL_KERNEL: reuses MATMUL_TANH_KERNEL structure from nki_codegen.py.
256x256 BF16 matmul with tanh activation, 4 blocks.

Per block: 2 matmuls (128x128x128 each) = 2 * 128^3 = 4,194,304 MACs
4 blocks total = 16,777,216 MACs
total_flops = 16,777,216 * 2 = 33,554,432

HBM DMA analysis per block:
  - 4 loads from "a"/"b" (HBM->SBUF): each 128x128 * 2 bytes = 32,768 bytes
  - 1 store to "output" (SBUF->HBM): 128x128 * 2 bytes = 32,768 bytes
  - tensor_copy and activation are SBUF-internal, not HBM

  But DMA copies touch HBM on ONE side only per copy:
  - Load from "a": src is HBM (32,768 bytes counted)
  - Load from "b": src is HBM (32,768 bytes counted)
  - Store to "output": dst is HBM (32,768 bytes counted)
  - dst of loads (tensor_N) is SBUF: not HBM
  - src of store (tensor_N) is SBUF: not HBM

  Per block: 4 loads * 32,768 + 1 store * 32,768 = 163,840 bytes
  4 blocks: 4 * 163,840 = 655,360 bytes

arithmetic_intensity = 33,554,432 / 655,360 = 51.2 FLOP/byte
BF16 ridge_point = 79e12 / 375e9 = 210.667 FLOP/byte
51.2 < 210.667 -> bound = "memory"
roofline_peak_tflops = 51.2 * 375e9 / 1e12 = 19.2 TFLOPS
"""

OUTPUT_SHAPE = (256, 256)
_PARAMS = ("a", "b", "output")


def _matmul_block(
    name: str,
    a_slices: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]],
    b_slices: tuple[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]],
    out_slices: tuple[tuple[int, int], tuple[int, int]],
    base_id: int,
) -> NKIBlock:
    """Build a single matmul block with 2 reduction steps."""
    t = lambda i: f"tensor_{base_id + i}"
    s128 = ((0, 128), (0, 128))
    return NKIBlock(
        name=name,
        params=_PARAMS,
        body=(
            NKIAlloc(dst=t(0), shape=(128, 128), dtype="nl.float32", buffer="psum"),
            NKIAlloc(dst=t(1), shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
            NKIDmaCopy(dst=TensorRef(t(1), (128, 128), s128), src=TensorRef("a", (128, 128), a_slices[0])),
            NKIAlloc(dst=t(2), shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
            NKIDmaCopy(dst=TensorRef(t(2), (128, 128), s128), src=TensorRef("b", (128, 128), b_slices[0])),
            NKIMatmul(
                dst=TensorRef(t(0), (128, 128), s128),
                stationary=TensorRef(t(1), (128, 128), s128),
                moving=TensorRef(t(2), (128, 128), s128),
            ),
            NKIAlloc(dst=t(3), shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
            NKIDmaCopy(dst=TensorRef(t(3), (128, 128), s128), src=TensorRef("a", (128, 128), a_slices[1])),
            NKIAlloc(dst=t(4), shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
            NKIDmaCopy(dst=TensorRef(t(4), (128, 128), s128), src=TensorRef("b", (128, 128), b_slices[1])),
            NKIMatmul(
                dst=TensorRef(t(0), (128, 128), s128),
                stationary=TensorRef(t(3), (128, 128), s128),
                moving=TensorRef(t(4), (128, 128), s128),
            ),
            NKIAlloc(dst=t(5), shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
            NKIDmaCopy(dst=TensorRef("output", OUTPUT_SHAPE, out_slices), src=TensorRef(t(5), (128, 128), s128)),
        ),
    )


ROOFLINE_MATMUL_KERNEL = NKIKernel(
    name="matmul_bf16",
    params=("a", "b"),
    input_shapes=(OUTPUT_SHAPE, OUTPUT_SHAPE),
    dtype="nl.bfloat16",
    output_shape=OUTPUT_SHAPE,
    blocks=(
        _matmul_block(
            "_block_0",
            (((0, 128), (0, 128)), ((128, 256), (0, 128))),
            (((0, 128), (0, 128)), ((128, 256), (0, 128))),
            ((0, 128), (0, 128)),
            0,
        ),
        _matmul_block(
            "_block_1",
            (((0, 128), (0, 128)), ((128, 256), (0, 128))),
            (((0, 128), (128, 256)), ((128, 256), (128, 256))),
            ((0, 128), (128, 256)),
            10,
        ),
        _matmul_block(
            "_block_2",
            (((0, 128), (128, 256)), ((128, 256), (128, 256))),
            (((0, 128), (0, 128)), ((128, 256), (0, 128))),
            ((128, 256), (0, 128)),
            20,
        ),
        _matmul_block(
            "_block_3",
            (((0, 128), (128, 256)), ((128, 256), (128, 256))),
            (((0, 128), (128, 256)), ((128, 256), (128, 256))),
            ((128, 256), (128, 256)),
            30,
        ),
    ),
)

"""
ROOFLINE_COMPUTE_KERNEL: compute-bound kernel with high arithmetic intensity.
Single block with 16 matmuls sharing the same loaded data.
Only 2 DMA loads + 1 DMA store = 3 * 32,768 = 98,304 bytes HBM traffic.
16 matmuls * 128^3 MACs each = 33,554,432 MACs
total_flops = 33,554,432 * 2 = 67,108,864
arithmetic_intensity = 67,108,864 / 98,304 = 682.67 FLOP/byte
682.67 > 210.67 -> bound = "compute"
"""

_S128 = ((0, 128), (0, 128))

_COMPUTE_MATMULS: tuple[NKIMatmul, ...] = tuple(
    NKIMatmul(
        dst=TensorRef("psum_0", (128, 128), _S128),
        stationary=TensorRef("sbuf_a", (128, 128), _S128),
        moving=TensorRef("sbuf_b", (128, 128), _S128),
    )
    for _ in range(16)
)

ROOFLINE_COMPUTE_KERNEL = NKIKernel(
    name="compute_heavy",
    params=("a", "b"),
    input_shapes=((128, 128), (128, 128)),
    dtype="nl.bfloat16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="psum_0", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="sbuf_a", shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("sbuf_a", (128, 128), _S128), src=TensorRef("a", (128, 128), _S128)),
                NKIAlloc(dst="sbuf_b", shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("sbuf_b", (128, 128), _S128), src=TensorRef("b", (128, 128), _S128)),
                *_COMPUTE_MATMULS,
                NKIAlloc(dst="sbuf_out", shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("output", (128, 128), _S128), src=TensorRef("sbuf_out", (128, 128), _S128)),
            ),
        ),
    ),
)

"""
ROOFLINE_NO_MATMUL_KERNEL: DMA-only kernel, no matmuls.
total_flops = 0, arithmetic_intensity = 0.0, bound = "memory"
1 load (128x128 * 2 = 32,768 bytes) + 1 store (32,768 bytes) = 65,536 bytes
"""

ROOFLINE_NO_MATMUL_KERNEL = NKIKernel(
    name="copy_only",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.bfloat16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="sbuf_0", shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("sbuf_0", (128, 128), _S128), src=TensorRef("a", (128, 128), _S128)),
                NKIDmaCopy(dst=TensorRef("output", (128, 128), _S128), src=TensorRef("sbuf_0", (128, 128), _S128)),
            ),
        ),
    ),
)

"""
ROOFLINE_ZERO_BYTES_KERNEL: only SBUF-internal ops, no HBM DMA at all.
total_flops > 0 but total_hbm_bytes = 0 -> arithmetic_intensity = inf, bound = "compute"
"""

ROOFLINE_ZERO_BYTES_KERNEL = NKIKernel(
    name="sbuf_only",
    params=(),
    input_shapes=(),
    dtype="nl.bfloat16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("output",),
            body=(
                NKIAlloc(dst="psum_0", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="sbuf_a", shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
                NKIAlloc(dst="sbuf_b", shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
                NKIMatmul(
                    dst=TensorRef("psum_0", (128, 128), _S128),
                    stationary=TensorRef("sbuf_a", (128, 128), _S128),
                    moving=TensorRef("sbuf_b", (128, 128), _S128),
                ),
            ),
        ),
    ),
)
