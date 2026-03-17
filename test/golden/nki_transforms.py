"""Golden test data for transforms: before/after/options for data reuse and operand merge."""

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.activation import NKIActivation
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.transforms.base import StmtRef, TransformOption

DATA_REUSE_BEFORE = NKIKernel(
    name="test",
    params=("a", "b"),
    input_shapes=((128, 128), (128, 128)),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
        NKIBlock(
            name="_block_1",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_3", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

DATA_REUSE_OPTIONS = [
    TransformOption(ref_a=StmtRef(block_name="_block_0", stmt_idx=1), ref_b=StmtRef(block_name="_block_1", stmt_idx=1)),
    TransformOption(ref_a=StmtRef(block_name="_block_0", stmt_idx=3), ref_b=StmtRef(block_name="_block_1", stmt_idx=3)),
]

DATA_REUSE_AFTER = NKIKernel(
    name="test",
    params=("a", "b"),
    input_shapes=((128, 128), (128, 128)),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0_1",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIAlloc(dst="tensor_3", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

DATA_REUSE_NO_MATCH = NKIKernel(
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

DMA_MERGE_BEFORE = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 256),),
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
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (128, 256))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

DMA_MERGE_OPTIONS = [
    TransformOption(ref_a=StmtRef(block_name="_block_0", stmt_idx=1), ref_b=StmtRef(block_name="_block_0", stmt_idx=3))
]

DMA_MERGE_AFTER = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 256),),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("a", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

COMPUTE_MERGE_BEFORE = NKIKernel(
    name="test",
    params=("a", "b"),
    input_shapes=((128, 128), (128, 256)),
    dtype="nl.float16",
    output_shape=(128, 256),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_2", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("b", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_3", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIMatmul(
                    dst=TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_2", (128, 128), ((0, 128), (128, 256))),
                ),
            ),
        ),
    ),
)

COMPUTE_MERGE_OPTIONS = [
    TransformOption(ref_a=StmtRef(block_name="_block_0", stmt_idx=5), ref_b=StmtRef(block_name="_block_0", stmt_idx=7))
]

COMPUTE_MERGE_AFTER = NKIKernel(
    name="test",
    params=("a", "b"),
    input_shapes=((128, 128), (128, 256)),
    dtype="nl.float16",
    output_shape=(128, 256),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 256), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_2", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("b", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
                    stationary=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_2", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIAlloc(dst="tensor_3", shape=(128, 128), dtype="nl.float32", buffer="psum"),
            ),
        ),
    ),
)

M_AXIS_KERNEL = NKIKernel(
    name="test",
    params=("a", "b"),
    input_shapes=((128, 256), (128, 128)),
    dtype="nl.float16",
    output_shape=(256, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_1", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("a", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_3", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIMatmul(
                    dst=TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256))),
                    moving=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

ACCUMULATION_KERNEL = NKIKernel(
    name="test",
    params=("a", "b"),
    input_shapes=((128, 128), (128, 128)),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "b", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_1", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("a", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256))),
                    moving=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

ACTIVATION_MERGE_BEFORE = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 256),),
    dtype="nl.float16",
    output_shape=(128, 256),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("a", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    op="nl.tanh",
                ),
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (128, 256))),
                    op="nl.tanh",
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (128, 256))),
                    src=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

ACTIVATION_MERGE_OPTIONS = [
    TransformOption(ref_a=StmtRef(block_name="_block_0", stmt_idx=3), ref_b=StmtRef(block_name="_block_0", stmt_idx=5))
]

ACTIVATION_MERGE_AFTER = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 256),),
    dtype="nl.float16",
    output_shape=(128, 256),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("a", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIAlloc(dst="tensor_1", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_1", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
                    op="nl.tanh",
                ),
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (128, 256))),
                    src=TensorRef("tensor_1", (128, 128), ((0, 128), (128, 256))),
                ),
            ),
        ),
    ),
)

DMA_PARTITION_MERGE_KERNEL = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((256, 128),),
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
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((128, 256), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

ACTIVATION_DIFFERENT_OPS_KERNEL = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 256),),
    dtype="nl.float16",
    output_shape=(128, 256),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", (128, 256), ((0, 128), (0, 256))),
                    src=TensorRef("a", (128, 256), ((0, 128), (0, 256))),
                ),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    op="nl.tanh",
                ),
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (128, 256))),
                    op="nl.exp",
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (128, 256))),
                    src=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)
