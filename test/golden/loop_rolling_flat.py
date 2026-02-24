"""Non-reduction unrolled golden functions for loop rolling tests."""

import numpy as np

import nkigym


def flat_1x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 1x1 tiled matmul (1 row, 1 col)."""
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    return output


def flat_2x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 2x1 tiled matmul (1 row, 2 cols)."""
    output = np.empty((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 128:256] = tensor_5[0:128, 0:128]
    return output


def flat_3x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 3x1 tiled matmul (1 row, 3 cols)."""
    output = np.empty((128, 384), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 128:256] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 0:128]
    tensor_7 = b[0:128, 256:384]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    output[0:128, 256:384] = tensor_8[0:128, 0:128]
    return output


def flat_4x1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 4x1 tiled matmul (1 row, 4 cols)."""
    output = np.empty((128, 512), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 128:256] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 0:128]
    tensor_7 = b[0:128, 256:384]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    output[0:128, 256:384] = tensor_8[0:128, 0:128]
    tensor_9 = a[0:128, 0:128]
    tensor_10 = b[0:128, 384:512]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    output[0:128, 384:512] = tensor_11[0:128, 0:128]
    return output


def flat_1x4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 1x4 tiled matmul (4 rows, 1 col)."""
    output = np.empty((512, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 128:256]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[128:256, 0:128] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 256:384]
    tensor_7 = b[0:128, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    output[256:384, 0:128] = tensor_8[0:128, 0:128]
    tensor_9 = a[0:128, 384:512]
    tensor_10 = b[0:128, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    output[384:512, 0:128] = tensor_11[0:128, 0:128]
    return output


def flat_1x5(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 1x5 tiled matmul (5 rows, 1 col)."""
    output = np.empty((640, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 128:256]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[128:256, 0:128] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 256:384]
    tensor_7 = b[0:128, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    output[256:384, 0:128] = tensor_8[0:128, 0:128]
    tensor_9 = a[0:128, 384:512]
    tensor_10 = b[0:128, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    output[384:512, 0:128] = tensor_11[0:128, 0:128]
    tensor_12 = a[0:128, 512:640]
    tensor_13 = b[0:128, 0:128]
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    output[512:640, 0:128] = tensor_14[0:128, 0:128]
    return output


def flat_2x2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 2x2 tiled matmul (2 rows, 2 cols)."""
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 128:256] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 128:256]
    tensor_7 = b[0:128, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    output[128:256, 0:128] = tensor_8[0:128, 0:128]
    tensor_9 = a[0:128, 128:256]
    tensor_10 = b[0:128, 128:256]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    output[128:256, 128:256] = tensor_11[0:128, 0:128]
    return output


_SOURCE_FLAT_3X5 = """\
def flat_3x5(a, b):
    output = np.empty((640, 384), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 128:256] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 0:128]
    tensor_7 = b[0:128, 256:384]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    output[0:128, 256:384] = tensor_8[0:128, 0:128]
    tensor_9 = a[0:128, 128:256]
    tensor_10 = b[0:128, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    output[128:256, 0:128] = tensor_11[0:128, 0:128]
    tensor_12 = a[0:128, 128:256]
    tensor_13 = b[0:128, 128:256]
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    output[128:256, 128:256] = tensor_14[0:128, 0:128]
    tensor_15 = a[0:128, 128:256]
    tensor_16 = b[0:128, 256:384]
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    output[128:256, 256:384] = tensor_17[0:128, 0:128]
    tensor_18 = a[0:128, 256:384]
    tensor_19 = b[0:128, 0:128]
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    output[256:384, 0:128] = tensor_20[0:128, 0:128]
    tensor_21 = a[0:128, 256:384]
    tensor_22 = b[0:128, 128:256]
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128])
    output[256:384, 128:256] = tensor_23[0:128, 0:128]
    tensor_24 = a[0:128, 256:384]
    tensor_25 = b[0:128, 256:384]
    tensor_26 = nkigym.nc_matmul(tensor_24[0:128, 0:128], tensor_25[0:128, 0:128])
    output[256:384, 256:384] = tensor_26[0:128, 0:128]
    tensor_27 = a[0:128, 384:512]
    tensor_28 = b[0:128, 0:128]
    tensor_29 = nkigym.nc_matmul(tensor_27[0:128, 0:128], tensor_28[0:128, 0:128])
    output[384:512, 0:128] = tensor_29[0:128, 0:128]
    tensor_30 = a[0:128, 384:512]
    tensor_31 = b[0:128, 128:256]
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128])
    output[384:512, 128:256] = tensor_32[0:128, 0:128]
    tensor_33 = a[0:128, 384:512]
    tensor_34 = b[0:128, 256:384]
    tensor_35 = nkigym.nc_matmul(tensor_33[0:128, 0:128], tensor_34[0:128, 0:128])
    output[384:512, 256:384] = tensor_35[0:128, 0:128]
    tensor_36 = a[0:128, 512:640]
    tensor_37 = b[0:128, 0:128]
    tensor_38 = nkigym.nc_matmul(tensor_36[0:128, 0:128], tensor_37[0:128, 0:128])
    output[512:640, 0:128] = tensor_38[0:128, 0:128]
    tensor_39 = a[0:128, 512:640]
    tensor_40 = b[0:128, 128:256]
    tensor_41 = nkigym.nc_matmul(tensor_39[0:128, 0:128], tensor_40[0:128, 0:128])
    output[512:640, 128:256] = tensor_41[0:128, 0:128]
    tensor_42 = a[0:128, 512:640]
    tensor_43 = b[0:128, 256:384]
    tensor_44 = nkigym.nc_matmul(tensor_42[0:128, 0:128], tensor_43[0:128, 0:128])
    output[512:640, 256:384] = tensor_44[0:128, 0:128]
    return output
"""


def flat_3x5(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 3x5 tiled matmul (5 rows, 3 cols)."""
    output = np.empty((640, 384), dtype=np.float32)
    for i in range(5):
        for j in range(3):
            tensor_0 = a[0:128, i * 128 : (i + 1) * 128]
            tensor_1 = b[0:128, j * 128 : (j + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            output[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = tensor_2[0:128, 0:128]
    return output


flat_3x5.__source__ = _SOURCE_FLAT_3X5

_SOURCE_FLAT_4X4 = """\
def flat_4x4(a, b):
    output = np.empty((512, 512), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 128:256] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 0:128]
    tensor_7 = b[0:128, 256:384]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    output[0:128, 256:384] = tensor_8[0:128, 0:128]
    tensor_9 = a[0:128, 0:128]
    tensor_10 = b[0:128, 384:512]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    output[0:128, 384:512] = tensor_11[0:128, 0:128]
    tensor_12 = a[0:128, 128:256]
    tensor_13 = b[0:128, 0:128]
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    output[128:256, 0:128] = tensor_14[0:128, 0:128]
    tensor_15 = a[0:128, 128:256]
    tensor_16 = b[0:128, 128:256]
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    output[128:256, 128:256] = tensor_17[0:128, 0:128]
    tensor_18 = a[0:128, 128:256]
    tensor_19 = b[0:128, 256:384]
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    output[128:256, 256:384] = tensor_20[0:128, 0:128]
    tensor_21 = a[0:128, 128:256]
    tensor_22 = b[0:128, 384:512]
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128])
    output[128:256, 384:512] = tensor_23[0:128, 0:128]
    tensor_24 = a[0:128, 256:384]
    tensor_25 = b[0:128, 0:128]
    tensor_26 = nkigym.nc_matmul(tensor_24[0:128, 0:128], tensor_25[0:128, 0:128])
    output[256:384, 0:128] = tensor_26[0:128, 0:128]
    tensor_27 = a[0:128, 256:384]
    tensor_28 = b[0:128, 128:256]
    tensor_29 = nkigym.nc_matmul(tensor_27[0:128, 0:128], tensor_28[0:128, 0:128])
    output[256:384, 128:256] = tensor_29[0:128, 0:128]
    tensor_30 = a[0:128, 256:384]
    tensor_31 = b[0:128, 256:384]
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128])
    output[256:384, 256:384] = tensor_32[0:128, 0:128]
    tensor_33 = a[0:128, 256:384]
    tensor_34 = b[0:128, 384:512]
    tensor_35 = nkigym.nc_matmul(tensor_33[0:128, 0:128], tensor_34[0:128, 0:128])
    output[256:384, 384:512] = tensor_35[0:128, 0:128]
    tensor_36 = a[0:128, 384:512]
    tensor_37 = b[0:128, 0:128]
    tensor_38 = nkigym.nc_matmul(tensor_36[0:128, 0:128], tensor_37[0:128, 0:128])
    output[384:512, 0:128] = tensor_38[0:128, 0:128]
    tensor_39 = a[0:128, 384:512]
    tensor_40 = b[0:128, 128:256]
    tensor_41 = nkigym.nc_matmul(tensor_39[0:128, 0:128], tensor_40[0:128, 0:128])
    output[384:512, 128:256] = tensor_41[0:128, 0:128]
    tensor_42 = a[0:128, 384:512]
    tensor_43 = b[0:128, 256:384]
    tensor_44 = nkigym.nc_matmul(tensor_42[0:128, 0:128], tensor_43[0:128, 0:128])
    output[384:512, 256:384] = tensor_44[0:128, 0:128]
    tensor_45 = a[0:128, 384:512]
    tensor_46 = b[0:128, 384:512]
    tensor_47 = nkigym.nc_matmul(tensor_45[0:128, 0:128], tensor_46[0:128, 0:128])
    output[384:512, 384:512] = tensor_47[0:128, 0:128]
    return output
"""


def flat_4x4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 4x4 tiled matmul (4 rows, 4 cols)."""
    output = np.empty((512, 512), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            tensor_0 = a[0:128, i * 128 : (i + 1) * 128]
            tensor_1 = b[0:128, j * 128 : (j + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            output[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = tensor_2[0:128, 0:128]
    return output


flat_4x4.__source__ = _SOURCE_FLAT_4X4
