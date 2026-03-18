"""Golden test data for DataReuseTransform: two-block, three-block, within-block, and no-match cases."""

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.transforms.base import StmtRef, TransformOption

_SBUF_SHAPE = (128, 128)
_SBUF_SLICES = ((0, 128), (0, 128))
_OUTPUT_SHAPE = (256, 256)
_A_SRC = TensorRef("a", (128, 128), ((0, 128), (0, 128)))

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
    TransformOption(StmtRef("_block_0", 1), StmtRef("_block_1", 1)),
    TransformOption(StmtRef("_block_0", 3), StmtRef("_block_1", 3)),
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

THREE_BLOCK_BEFORE = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=_OUTPUT_SHAPE,
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES), src=_A_SRC),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
        NKIBlock(
            name="_block_1",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_1", _SBUF_SHAPE, _SBUF_SLICES), src=_A_SRC),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((0, 128), (128, 256))),
                    src=TensorRef("tensor_1", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
        NKIBlock(
            name="_block_2",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_2", _SBUF_SHAPE, _SBUF_SLICES), src=_A_SRC),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((128, 256), (0, 128))),
                    src=TensorRef("tensor_2", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
    ),
)

THREE_BLOCK_OPTIONS = [
    TransformOption(StmtRef("_block_0", 1), StmtRef("_block_1", 1)),
    TransformOption(StmtRef("_block_0", 1), StmtRef("_block_2", 1)),
    TransformOption(StmtRef("_block_1", 1), StmtRef("_block_2", 1)),
]

THREE_BLOCK_AFTER = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=_OUTPUT_SHAPE,
    blocks=(
        NKIBlock(
            name="_block_0_1",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES), src=_A_SRC),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES),
                ),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((0, 128), (128, 256))),
                    src=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
        NKIBlock(
            name="_block_2",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_2", _SBUF_SHAPE, _SBUF_SLICES), src=_A_SRC),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((128, 256), (0, 128))),
                    src=TensorRef("tensor_2", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
    ),
)


DIFFERENT_SLICES_KERNEL = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((256, 256),),
    dtype="nl.float16",
    output_shape=_OUTPUT_SHAPE,
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
        NKIBlock(
            name="_block_1",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", _SBUF_SHAPE, _SBUF_SLICES),
                    src=TensorRef("a", (128, 128), ((128, 256), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((0, 128), (128, 256))),
                    src=TensorRef("tensor_1", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
        NKIBlock(
            name="_block_2",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_2", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_2", _SBUF_SHAPE, _SBUF_SLICES),
                    src=TensorRef("a", (128, 128), ((0, 128), (128, 256))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", _OUTPUT_SHAPE, ((128, 256), (0, 128))),
                    src=TensorRef("tensor_2", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
    ),
)

WITHIN_BLOCK_BEFORE = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=(128, 256),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES), src=_A_SRC),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_1", _SBUF_SHAPE, _SBUF_SLICES), src=_A_SRC),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (128, 256))),
                    src=TensorRef("tensor_1", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
    ),
)

WITHIN_BLOCK_OPTIONS = [TransformOption(StmtRef("_block_0", 1), StmtRef("_block_0", 3))]

WITHIN_BLOCK_AFTER = NKIKernel(
    name="test",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=(128, 256),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES), src=_A_SRC),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 256), ((0, 128), (128, 256))),
                    src=TensorRef("tensor_0", _SBUF_SHAPE, _SBUF_SLICES),
                ),
            ),
        ),
    ),
)
