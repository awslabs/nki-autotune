"""Golden test data for DCE: before/after kernels for dead code elimination."""

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy

_TENSOR_0_REF = TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))
_A_INPUT_REF = TensorRef("a", (128, 128), ((0, 128), (0, 128)))
_OUTPUT_REF = TensorRef("output", (128, 128), ((0, 128), (0, 128)))

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
                NKIDmaCopy(dst=_TENSOR_0_REF, src=_A_INPUT_REF),
                NKIDmaCopy(dst=_OUTPUT_REF, src=_TENSOR_0_REF),
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
                NKIDmaCopy(dst=_TENSOR_0_REF, src=_A_INPUT_REF),
                NKIAlloc(dst="tensor_dead", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_dead", (128, 128), ((0, 128), (0, 128))), src=_A_INPUT_REF),
                NKIDmaCopy(dst=_OUTPUT_REF, src=_TENSOR_0_REF),
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
        NKIBlock(name="_block_0", params=("a", "output"), body=(NKIDmaCopy(dst=_OUTPUT_REF, src=_A_INPUT_REF),)),
        NKIBlock(
            name="_block_1",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_dead", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_dead", (128, 128), ((0, 128), (0, 128))), src=_A_INPUT_REF),
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
    blocks=(NKIBlock(name="_block_0", params=("a", "output"), body=(NKIDmaCopy(dst=_OUTPUT_REF, src=_A_INPUT_REF),)),),
)

_TENSOR_1_REF = TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))
_TENSOR_2_REF = TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))
_TENSOR_3_REF = TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))
_TENSOR_4_REF = TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))

DCE_DEAD_CHAIN_BEFORE = NKIKernel(
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
                NKIDmaCopy(dst=_TENSOR_0_REF, src=_A_INPUT_REF),
                NKIDmaCopy(dst=_OUTPUT_REF, src=_TENSOR_0_REF),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_TENSOR_2_REF, src=_A_INPUT_REF),
                NKIAlloc(dst="tensor_3", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_TENSOR_3_REF, src=_A_INPUT_REF),
                NKIMatmul(dst=_TENSOR_1_REF, stationary=_TENSOR_2_REF, moving=_TENSOR_3_REF),
                NKIAlloc(dst="tensor_4", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKITensorCopy(dst=_TENSOR_4_REF, src=_TENSOR_1_REF),
            ),
        ),
    ),
)

DCE_DEAD_CHAIN_AFTER = DCE_LIVE_KERNEL

DCE_TRANSITIVE_BEFORE = NKIKernel(
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
                NKIDmaCopy(dst=_OUTPUT_REF, src=_A_INPUT_REF),
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_TENSOR_0_REF, src=_A_INPUT_REF),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_TENSOR_2_REF, src=_A_INPUT_REF),
                NKIMatmul(dst=_TENSOR_1_REF, stationary=_TENSOR_0_REF, moving=_TENSOR_2_REF),
            ),
        ),
    ),
)

DCE_TRANSITIVE_AFTER = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(NKIBlock(name="_block_0", params=("a", "output"), body=(NKIDmaCopy(dst=_OUTPUT_REF, src=_A_INPUT_REF),)),),
)
