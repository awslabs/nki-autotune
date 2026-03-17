"""Golden test data for codegen package: NKIKernel objects and expected render strings."""

import pytest

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.activation import NKIActivation
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy

OUTPUT_SHAPE = (256, 256)
_PARAMS = ("a", "b", "output")

MATMUL_TANH_KERNEL = NKIKernel(
    name="matmul_tanh",
    params=("a", "b"),
    input_shapes=(OUTPUT_SHAPE, OUTPUT_SHAPE),
    dtype="nl.float16",
    output_shape=OUTPUT_SHAPE,
    blocks=(
        NKIBlock(
            name="_block_0",
            params=_PARAMS,
            body=(
                NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
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
                NKIAlloc(dst="tensor_3", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((128, 256), (0, 128))),
                ),
                NKIAlloc(dst="tensor_4", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((128, 256), (0, 128))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_5", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKITensorCopy(
                    dst=TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_6", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
                    op="nl.tanh",
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", OUTPUT_SHAPE, ((0, 128), (0, 128))),
                    src=TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
        NKIBlock(
            name="_block_1",
            params=_PARAMS,
            body=(
                NKIAlloc(dst="tensor_7", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_8", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_9", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (128, 256))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_10", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((128, 256), (0, 128))),
                ),
                NKIAlloc(dst="tensor_11", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((128, 256), (128, 256))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_12", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKITensorCopy(
                    dst=TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_13", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128))),
                    op="nl.tanh",
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", OUTPUT_SHAPE, ((0, 128), (128, 256))),
                    src=TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
        NKIBlock(
            name="_block_2",
            params=_PARAMS,
            body=(
                NKIAlloc(dst="tensor_14", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_15", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (128, 256))),
                ),
                NKIAlloc(dst="tensor_16", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_17", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((128, 256), (128, 256))),
                ),
                NKIAlloc(dst="tensor_18", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((128, 256), (0, 128))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_19", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKITensorCopy(
                    dst=TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_20", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128))),
                    op="nl.tanh",
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", OUTPUT_SHAPE, ((128, 256), (0, 128))),
                    src=TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
        NKIBlock(
            name="_block_3",
            params=_PARAMS,
            body=(
                NKIAlloc(dst="tensor_21", shape=(128, 128), dtype="nl.float32", buffer="psum"),
                NKIAlloc(dst="tensor_22", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (128, 256))),
                ),
                NKIAlloc(dst="tensor_23", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((0, 128), (128, 256))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_24", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_24", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((128, 256), (128, 256))),
                ),
                NKIAlloc(dst="tensor_25", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_25", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("b", (128, 128), ((128, 256), (128, 256))),
                ),
                NKIMatmul(
                    dst=TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128))),
                    stationary=TensorRef("tensor_24", (128, 128), ((0, 128), (0, 128))),
                    moving=TensorRef("tensor_25", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_26", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKITensorCopy(
                    dst=TensorRef("tensor_26", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIAlloc(dst="tensor_27", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIActivation(
                    dst=TensorRef("tensor_27", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_26", (128, 128), ((0, 128), (0, 128))),
                    op="nl.tanh",
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", OUTPUT_SHAPE, ((128, 256), (128, 256))),
                    src=TensorRef("tensor_27", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

MATMUL_TANH_RENDERED = (
    "import nki\n"
    "import nki.language as nl\n"
    "import nki.isa as nisa\n"
    "\n"
    "\n"
    "@nki.jit\n"
    "def matmul_tanh(a, b):\n"
    "    assert a.shape == (256, 256)\n"
    "    assert a.dtype == nl.float16\n"
    "    assert b.shape == (256, 256)\n"
    "    assert b.dtype == nl.float16\n"
    "    output = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)\n"
    "    for i_p0 in nl.affine_range(2):\n"
    "        for i_p1 in nl.affine_range(2):\n"
    "            _rolled_block_0_1_2_3(a, b, output, i_p0, i_p1)\n"
    "    return output\n"
    "\n"
    "\n"
    "def _rolled_block_0_1_2_3(a, b, output, i_p0, i_p1):\n"
    "    tensor_0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)\n"
    "    tensor_1 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_1[0:128, 0:128], src=a[0:128, i_p0 * 128:i_p0 * 128 + 128])\n"
    "    tensor_2 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_2[0:128, 0:128], src=b[0:128, i_p1 * 128:i_p1 * 128 + 128])\n"
    "    nisa.nc_matmul(dst=tensor_0[0:128, 0:128], stationary=tensor_1[0:128, 0:128],"
    " moving=tensor_2[0:128, 0:128])\n"
    "    tensor_3 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_3[0:128, 0:128], src=a[128:256, i_p0 * 128:i_p0 * 128 + 128])\n"
    "    tensor_4 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_4[0:128, 0:128], src=b[128:256, i_p1 * 128:i_p1 * 128 + 128])\n"
    "    nisa.nc_matmul(dst=tensor_0[0:128, 0:128], stationary=tensor_3[0:128, 0:128],"
    " moving=tensor_4[0:128, 0:128])\n"
    "    tensor_5 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.tensor_copy(dst=tensor_5[0:128, 0:128], src=tensor_0[0:128, 0:128])\n"
    "    tensor_6 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.activation(dst=tensor_6[0:128, 0:128], data=tensor_5[0:128, 0:128], op=nl.tanh)\n"
    "    nisa.dma_copy(dst=output[i_p0 * 128:i_p0 * 128 + 128, i_p1 * 128:i_p1 * 128 + 128],"
    " src=tensor_6[0:128, 0:128])\n"
)

NORMALIZE_BEFORE = NKIKernel(
    name="small",
    params=("a",),
    input_shapes=((128, 128),),
    dtype="nl.float16",
    output_shape=(128, 128),
    blocks=(
        NKIBlock(
            name="_block_0",
            params=("a", "output"),
            body=(
                NKIAlloc(dst="tensor_5", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(
                    dst=TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
                ),
                NKIDmaCopy(
                    dst=TensorRef("output", (128, 128), ((0, 128), (0, 128))),
                    src=TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
                ),
            ),
        ),
    ),
)

NORMALIZE_AFTER = NKIKernel(
    name="small",
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

STMT_RENDER_CASES = [
    pytest.param(
        NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.float32", buffer="psum"),
        "tensor_0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)",
        id="alloc_psum",
    ),
    pytest.param(
        NKIAlloc(dst="tensor_1", shape=(128, 128), dtype="nl.float16", buffer="sbuf"),
        "tensor_1 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)",
        id="alloc_sbuf",
    ),
    pytest.param(
        NKIDmaCopy(
            dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            src=TensorRef("a", (128, 128), ((0, 128), (0, 128))),
        ),
        "nisa.dma_copy(dst=tensor_1[0:128, 0:128], src=a[0:128, 0:128])",
        id="dma_copy",
    ),
    pytest.param(
        NKIMatmul(
            dst=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            stationary=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            moving=TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
        ),
        "nisa.nc_matmul(dst=tensor_0[0:128, 0:128], stationary=tensor_1[0:128, 0:128], moving=tensor_2[0:128, 0:128])",
        id="matmul",
    ),
    pytest.param(
        NKITensorCopy(
            dst=TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            src=TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
        ),
        "nisa.tensor_copy(dst=tensor_5[0:128, 0:128], src=tensor_0[0:128, 0:128])",
        id="tensor_copy",
    ),
    pytest.param(
        NKIActivation(
            dst=TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            src=TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            op="nl.tanh",
        ),
        "nisa.activation(dst=tensor_6[0:128, 0:128], data=tensor_5[0:128, 0:128], op=nl.tanh)",
        id="activation",
    ),
    pytest.param(
        NKIAlloc(dst="tensor_0", shape=(128, 256), dtype="nl.float16", buffer="sbuf"),
        "tensor_0 = nl.ndarray((128, 256), dtype=nl.float16, buffer=nl.sbuf)",
        id="alloc_sbuf_wide",
    ),
    pytest.param(
        NKIAlloc(dst="tensor_0", shape=(128, 256), dtype="nl.float32", buffer="psum"),
        "tensor_0 = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.psum)",
        id="alloc_psum_wide",
    ),
    pytest.param(
        NKIAlloc(dst="tensor_0", shape=(128, 128), dtype="nl.bfloat16", buffer="sbuf"),
        "tensor_0 = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)",
        id="alloc_bfloat16",
    ),
    pytest.param(
        NKIDmaCopy(
            dst=TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            src=TensorRef("a", (128, 128), ((128, 256), (0, 128))),
        ),
        "nisa.dma_copy(dst=tensor_1[0:128, 0:128], src=a[128:256, 0:128])",
        id="dma_offset_src",
    ),
    pytest.param(
        NKIActivation(
            dst=TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            src=TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            op="nl.exp",
        ),
        "nisa.activation(dst=tensor_6[0:128, 0:128], data=tensor_5[0:128, 0:128], op=nl.exp)",
        id="activation_exp",
    ),
    pytest.param(
        NKIActivation(
            dst=TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            src=TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            op="nl.identity",
        ),
        "nisa.activation(dst=tensor_6[0:128, 0:128], data=tensor_5[0:128, 0:128], op=nl.identity)",
        id="activation_identity",
    ),
]
