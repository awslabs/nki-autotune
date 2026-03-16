"""Golden test data for DCE: before/after kernels for dead code elimination."""

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy

DCE_LIVE_KERNEL = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

DCE_DEAD_BEFORE = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_dead", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_dead", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

DCE_DEAD_AFTER = DCE_LIVE_KERNEL

DCE_EMPTY_BLOCK_BEFORE = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
        NKIBlock(
            name="_block_1",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_dead", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_dead", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

DCE_EMPTY_BLOCK_AFTER = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)
