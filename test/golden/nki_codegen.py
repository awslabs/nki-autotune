"""Golden test data for codegen package: NKIKernel objects and expected render strings."""

import pytest

from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.activation import NKIActivation
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy

_T = (128, 128)
_S = ((0, 128), (0, 128))
_O = (256, 256)
_PARAMS = ("a", "b", "output")


def _r(name: str, shape: tuple[int, ...], slices: tuple[tuple[int, int], ...]) -> TensorRef:
    """Create a TensorRef shorthand.

    Args:
        name: Variable name.
        shape: Tensor shape.
        slices: Per-axis (start, stop) bounds.

    Returns:
        TensorRef instance.
    """
    return TensorRef(name, shape, slices)


def _sbuf_alloc(idx: int) -> NKIAlloc:
    """Create an SBUF alloc for tensor_<idx>.

    Args:
        idx: Tensor index.

    Returns:
        NKIAlloc for SBUF.
    """
    return NKIAlloc(dst=f"tensor_{idx}", shape=_T, dtype="nl.float16", buffer="sbuf")


def _block_body(
    base: int, a_slices_0: tuple, b_slices_0: tuple, a_slices_1: tuple, b_slices_1: tuple, out_slices: tuple
) -> tuple:
    """Build the 16-statement body for one matmul+tanh block.

    Args:
        base: Starting tensor index for this block.
        a_slices_0: HBM slices for 'a' at reduction position 0.
        b_slices_0: HBM slices for 'b' at reduction position 0.
        a_slices_1: HBM slices for 'a' at reduction position 1.
        b_slices_1: HBM slices for 'b' at reduction position 1.
        out_slices: Output HBM slices.

    Returns:
        Tuple of 16 NKIStmt instances.
    """
    psum = _r(f"tensor_{base}", _T, _S)
    return (
        NKIAlloc(dst=f"tensor_{base}", shape=_T, dtype="nl.float32", buffer="psum"),
        _sbuf_alloc(base + 1),
        NKIDmaCopy(dst=_r(f"tensor_{base + 1}", _T, _S), src=_r("a", _T, a_slices_0)),
        _sbuf_alloc(base + 2),
        NKIDmaCopy(dst=_r(f"tensor_{base + 2}", _T, _S), src=_r("b", _T, b_slices_0)),
        NKIMatmul(dst=psum, stationary=_r(f"tensor_{base + 1}", _T, _S), moving=_r(f"tensor_{base + 2}", _T, _S)),
        _sbuf_alloc(base + 3),
        NKIDmaCopy(dst=_r(f"tensor_{base + 3}", _T, _S), src=_r("a", _T, a_slices_1)),
        _sbuf_alloc(base + 4),
        NKIDmaCopy(dst=_r(f"tensor_{base + 4}", _T, _S), src=_r("b", _T, b_slices_1)),
        NKIMatmul(dst=psum, stationary=_r(f"tensor_{base + 3}", _T, _S), moving=_r(f"tensor_{base + 4}", _T, _S)),
        _sbuf_alloc(base + 5),
        NKITensorCopy(dst=_r(f"tensor_{base + 5}", _T, _S), src=psum),
        _sbuf_alloc(base + 6),
        NKIActivation(dst=_r(f"tensor_{base + 6}", _T, _S), src=_r(f"tensor_{base + 5}", _T, _S), op="nl.tanh"),
        NKIDmaCopy(dst=_r("output", _O, out_slices), src=_r(f"tensor_{base + 6}", _T, _S)),
    )


_HI = (128, 256)


MATMUL_TANH_BLOCK_0 = NKIBlock(
    name="_block_0", params=_PARAMS, body=_block_body(0, _S, _S, ((128, 256), (0, 128)), ((128, 256), (0, 128)), _S)
)

MATMUL_TANH_KERNEL = NKIKernel(
    name="matmul_tanh",
    params=("a", "b"),
    input_shapes=(_O, _O),
    dtype="nl.float16",
    output_shape=_O,
    blocks=(
        MATMUL_TANH_BLOCK_0,
        NKIBlock(
            name="_block_1",
            params=_PARAMS,
            body=_block_body(7, _S, ((0, 128), _HI), ((128, 256), (0, 128)), ((128, 256), _HI), ((0, 128), _HI)),
        ),
        NKIBlock(
            name="_block_2",
            params=_PARAMS,
            body=_block_body(14, ((0, 128), _HI), _S, ((128, 256), _HI), ((128, 256), (0, 128)), (_HI, (0, 128))),
        ),
        NKIBlock(
            name="_block_3",
            params=_PARAMS,
            body=_block_body(21, ((0, 128), _HI), ((0, 128), _HI), ((128, 256), _HI), ((128, 256), _HI), (_HI, _HI)),
        ),
    ),
)

MATMUL_TANH_RENDERED = (
    "import nki\n"
    "import nki.language as nl\n"
    "import nki.isa as nisa\n"
    "import numpy as np\n"
    "\n"
    "\n"
    "@nki.jit\n"
    "def matmul_tanh(a, b):\n"
    "    assert a.shape == (256, 256)\n"
    "    assert a.dtype == np.float16\n"
    "    assert b.shape == (256, 256)\n"
    "    assert b.dtype == np.float16\n"
    "    output = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)\n"
    "    _block_0(a, b, output)\n"
    "    _block_1(a, b, output)\n"
    "    _block_2(a, b, output)\n"
    "    _block_3(a, b, output)\n"
    "    return output\n"
    "\n"
    "\n"
    "def _block_0(a, b, output):\n"
    "    tensor_0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)\n"
    "    tensor_1 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_1[0:128, 0:128], src=a[0:128, 0:128])\n"
    "    tensor_2 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_2[0:128, 0:128], src=b[0:128, 0:128])\n"
    "    nisa.nc_matmul(dst=tensor_0[0:128, 0:128], stationary=tensor_1[0:128, 0:128], moving=tensor_2[0:128, 0:128])\n"
    "    tensor_3 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_3[0:128, 0:128], src=a[128:256, 0:128])\n"
    "    tensor_4 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_4[0:128, 0:128], src=b[128:256, 0:128])\n"
    "    nisa.nc_matmul(dst=tensor_0[0:128, 0:128], stationary=tensor_3[0:128, 0:128], moving=tensor_4[0:128, 0:128])\n"
    "    tensor_5 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.tensor_copy(dst=tensor_5[0:128, 0:128], src=tensor_0[0:128, 0:128])\n"
    "    tensor_6 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.activation(dst=tensor_6[0:128, 0:128], data=tensor_5[0:128, 0:128], op=nl.tanh)\n"
    "    nisa.dma_copy(dst=output[0:128, 0:128], src=tensor_6[0:128, 0:128])\n"
    "\n"
    "\n"
    "def _block_1(a, b, output):\n"
    "    tensor_7 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)\n"
    "    tensor_8 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_8[0:128, 0:128], src=a[0:128, 0:128])\n"
    "    tensor_9 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_9[0:128, 0:128], src=b[0:128, 128:256])\n"
    "    nisa.nc_matmul(dst=tensor_7[0:128, 0:128], stationary=tensor_8[0:128, 0:128], moving=tensor_9[0:128, 0:128])\n"
    "    tensor_10 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_10[0:128, 0:128], src=a[128:256, 0:128])\n"
    "    tensor_11 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_11[0:128, 0:128], src=b[128:256, 128:256])\n"
    "    nisa.nc_matmul(dst=tensor_7[0:128, 0:128], stationary=tensor_10[0:128, 0:128], moving=tensor_11[0:128, 0:128])\n"
    "    tensor_12 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.tensor_copy(dst=tensor_12[0:128, 0:128], src=tensor_7[0:128, 0:128])\n"
    "    tensor_13 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.activation(dst=tensor_13[0:128, 0:128], data=tensor_12[0:128, 0:128], op=nl.tanh)\n"
    "    nisa.dma_copy(dst=output[0:128, 128:256], src=tensor_13[0:128, 0:128])\n"
    "\n"
    "\n"
    "def _block_2(a, b, output):\n"
    "    tensor_14 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)\n"
    "    tensor_15 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_15[0:128, 0:128], src=a[0:128, 128:256])\n"
    "    tensor_16 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_16[0:128, 0:128], src=b[0:128, 0:128])\n"
    "    nisa.nc_matmul(dst=tensor_14[0:128, 0:128], stationary=tensor_15[0:128, 0:128], moving=tensor_16[0:128, 0:128])\n"
    "    tensor_17 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_17[0:128, 0:128], src=a[128:256, 128:256])\n"
    "    tensor_18 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_18[0:128, 0:128], src=b[128:256, 0:128])\n"
    "    nisa.nc_matmul(dst=tensor_14[0:128, 0:128], stationary=tensor_17[0:128, 0:128], moving=tensor_18[0:128, 0:128])\n"
    "    tensor_19 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.tensor_copy(dst=tensor_19[0:128, 0:128], src=tensor_14[0:128, 0:128])\n"
    "    tensor_20 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.activation(dst=tensor_20[0:128, 0:128], data=tensor_19[0:128, 0:128], op=nl.tanh)\n"
    "    nisa.dma_copy(dst=output[128:256, 0:128], src=tensor_20[0:128, 0:128])\n"
    "\n"
    "\n"
    "def _block_3(a, b, output):\n"
    "    tensor_21 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)\n"
    "    tensor_22 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_22[0:128, 0:128], src=a[0:128, 128:256])\n"
    "    tensor_23 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_23[0:128, 0:128], src=b[0:128, 128:256])\n"
    "    nisa.nc_matmul(dst=tensor_21[0:128, 0:128], stationary=tensor_22[0:128, 0:128], moving=tensor_23[0:128, 0:128])\n"
    "    tensor_24 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_24[0:128, 0:128], src=a[128:256, 128:256])\n"
    "    tensor_25 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.dma_copy(dst=tensor_25[0:128, 0:128], src=b[128:256, 128:256])\n"
    "    nisa.nc_matmul(dst=tensor_21[0:128, 0:128], stationary=tensor_24[0:128, 0:128], moving=tensor_25[0:128, 0:128])\n"
    "    tensor_26 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.tensor_copy(dst=tensor_26[0:128, 0:128], src=tensor_21[0:128, 0:128])\n"
    "    tensor_27 = nl.ndarray((128, 128), dtype=nl.float16, buffer=nl.sbuf)\n"
    "    nisa.activation(dst=tensor_27[0:128, 0:128], data=tensor_26[0:128, 0:128], op=nl.tanh)\n"
    "    nisa.dma_copy(dst=output[128:256, 128:256], src=tensor_27[0:128, 0:128])\n"
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
                NKIAlloc(dst="tensor_5", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_5", _T, _S), src=_r("a", _T, _S)),
                NKIDmaCopy(dst=_r("output", _T, _S), src=_r("tensor_5", _T, _S)),
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
                NKIAlloc(dst="tensor_0", shape=_T, dtype="nl.float16", buffer="sbuf"),
                NKIDmaCopy(dst=_r("tensor_0", _T, _S), src=_r("a", _T, _S)),
                NKIDmaCopy(dst=_r("output", _T, _S), src=_r("tensor_0", _T, _S)),
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
        NKIDmaCopy(dst=_r("tensor_1", _T, _S), src=_r("a", _T, _S)),
        "nisa.dma_copy(dst=tensor_1[0:128, 0:128], src=a[0:128, 0:128])",
        id="dma_copy",
    ),
    pytest.param(
        NKIMatmul(dst=_r("tensor_0", _T, _S), stationary=_r("tensor_1", _T, _S), moving=_r("tensor_2", _T, _S)),
        "nisa.nc_matmul(dst=tensor_0[0:128, 0:128], stationary=tensor_1[0:128, 0:128], moving=tensor_2[0:128, 0:128])",
        id="matmul",
    ),
    pytest.param(
        NKITensorCopy(dst=_r("tensor_5", _T, _S), src=_r("tensor_0", _T, _S)),
        "nisa.tensor_copy(dst=tensor_5[0:128, 0:128], src=tensor_0[0:128, 0:128])",
        id="tensor_copy",
    ),
    pytest.param(
        NKIActivation(dst=_r("tensor_6", _T, _S), src=_r("tensor_5", _T, _S), op="nl.tanh"),
        "nisa.activation(dst=tensor_6[0:128, 0:128], data=tensor_5[0:128, 0:128], op=nl.tanh)",
        id="activation",
    ),
]
