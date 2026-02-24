"""Reduction unrolled golden functions for loop rolling tests."""

import numpy as np

import nkigym


def flat_red2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat reduction-2 tiled matmul (1 row, 1 col, 2 reductions)."""
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    output[0:128, 0:128] = tensor_5[0:128, 0:128]
    return output


def flat_red4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat reduction-4 tiled matmul (1 row, 1 col, 4 reductions)."""
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    tensor_6 = a[256:384, 0:128]
    tensor_7 = b[256:384, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128], acc=tensor_5[0:128, 0:128])
    tensor_9 = a[384:512, 0:128]
    tensor_10 = b[384:512, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
    output[0:128, 0:128] = tensor_11[0:128, 0:128]
    return output


def flat_red8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat reduction-8 tiled matmul (1 row, 1 col, 8 reductions)."""
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    tensor_6 = a[256:384, 0:128]
    tensor_7 = b[256:384, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128], acc=tensor_5[0:128, 0:128])
    tensor_9 = a[384:512, 0:128]
    tensor_10 = b[384:512, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
    tensor_12 = a[512:640, 0:128]
    tensor_13 = b[512:640, 0:128]
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128], acc=tensor_11[0:128, 0:128])
    tensor_15 = a[640:768, 0:128]
    tensor_16 = b[640:768, 0:128]
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    tensor_18 = a[768:896, 0:128]
    tensor_19 = b[768:896, 0:128]
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128], acc=tensor_17[0:128, 0:128])
    tensor_21 = a[896:1024, 0:128]
    tensor_22 = b[896:1024, 0:128]
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128], acc=tensor_20[0:128, 0:128])
    output[0:128, 0:128] = tensor_23[0:128, 0:128]
    return output


def flat_2x2_red2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 2x2 reduction-2 tiled matmul (2 rows, 2 cols, 2 reductions)."""
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    output[0:128, 0:128] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 0:128]
    tensor_7 = b[0:128, 128:256]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    tensor_9 = a[128:256, 0:128]
    tensor_10 = b[128:256, 128:256]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
    output[0:128, 128:256] = tensor_11[0:128, 0:128]
    tensor_12 = a[0:128, 128:256]
    tensor_13 = b[0:128, 0:128]
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    tensor_15 = a[128:256, 128:256]
    tensor_16 = b[128:256, 0:128]
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    output[128:256, 0:128] = tensor_17[0:128, 0:128]
    tensor_18 = a[0:128, 128:256]
    tensor_19 = b[0:128, 128:256]
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    tensor_21 = a[128:256, 128:256]
    tensor_22 = b[128:256, 128:256]
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128], acc=tensor_20[0:128, 0:128])
    output[128:256, 128:256] = tensor_23[0:128, 0:128]
    return output


_SOURCE_FLAT_3X5_RED2 = """\
def flat_3x5_red2(a, b):
    output = np.empty((640, 384), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    output[0:128, 0:128] = tensor_5[0:128, 0:128]
    tensor_6 = a[0:128, 0:128]
    tensor_7 = b[0:128, 128:256]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    tensor_9 = a[128:256, 0:128]
    tensor_10 = b[128:256, 128:256]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
    output[0:128, 128:256] = tensor_11[0:128, 0:128]
    tensor_12 = a[0:128, 0:128]
    tensor_13 = b[0:128, 256:384]
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    tensor_15 = a[128:256, 0:128]
    tensor_16 = b[128:256, 256:384]
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    output[0:128, 256:384] = tensor_17[0:128, 0:128]
    tensor_18 = a[0:128, 128:256]
    tensor_19 = b[0:128, 0:128]
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    tensor_21 = a[128:256, 128:256]
    tensor_22 = b[128:256, 0:128]
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128], acc=tensor_20[0:128, 0:128])
    output[128:256, 0:128] = tensor_23[0:128, 0:128]
    tensor_24 = a[0:128, 128:256]
    tensor_25 = b[0:128, 128:256]
    tensor_26 = nkigym.nc_matmul(tensor_24[0:128, 0:128], tensor_25[0:128, 0:128])
    tensor_27 = a[128:256, 128:256]
    tensor_28 = b[128:256, 128:256]
    tensor_29 = nkigym.nc_matmul(tensor_27[0:128, 0:128], tensor_28[0:128, 0:128], acc=tensor_26[0:128, 0:128])
    output[128:256, 128:256] = tensor_29[0:128, 0:128]
    tensor_30 = a[0:128, 128:256]
    tensor_31 = b[0:128, 256:384]
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128])
    tensor_33 = a[128:256, 128:256]
    tensor_34 = b[128:256, 256:384]
    tensor_35 = nkigym.nc_matmul(tensor_33[0:128, 0:128], tensor_34[0:128, 0:128], acc=tensor_32[0:128, 0:128])
    output[128:256, 256:384] = tensor_35[0:128, 0:128]
    tensor_36 = a[0:128, 256:384]
    tensor_37 = b[0:128, 0:128]
    tensor_38 = nkigym.nc_matmul(tensor_36[0:128, 0:128], tensor_37[0:128, 0:128])
    tensor_39 = a[128:256, 256:384]
    tensor_40 = b[128:256, 0:128]
    tensor_41 = nkigym.nc_matmul(tensor_39[0:128, 0:128], tensor_40[0:128, 0:128], acc=tensor_38[0:128, 0:128])
    output[256:384, 0:128] = tensor_41[0:128, 0:128]
    tensor_42 = a[0:128, 256:384]
    tensor_43 = b[0:128, 128:256]
    tensor_44 = nkigym.nc_matmul(tensor_42[0:128, 0:128], tensor_43[0:128, 0:128])
    tensor_45 = a[128:256, 256:384]
    tensor_46 = b[128:256, 128:256]
    tensor_47 = nkigym.nc_matmul(tensor_45[0:128, 0:128], tensor_46[0:128, 0:128], acc=tensor_44[0:128, 0:128])
    output[256:384, 128:256] = tensor_47[0:128, 0:128]
    tensor_48 = a[0:128, 256:384]
    tensor_49 = b[0:128, 256:384]
    tensor_50 = nkigym.nc_matmul(tensor_48[0:128, 0:128], tensor_49[0:128, 0:128])
    tensor_51 = a[128:256, 256:384]
    tensor_52 = b[128:256, 256:384]
    tensor_53 = nkigym.nc_matmul(tensor_51[0:128, 0:128], tensor_52[0:128, 0:128], acc=tensor_50[0:128, 0:128])
    output[256:384, 256:384] = tensor_53[0:128, 0:128]
    tensor_54 = a[0:128, 384:512]
    tensor_55 = b[0:128, 0:128]
    tensor_56 = nkigym.nc_matmul(tensor_54[0:128, 0:128], tensor_55[0:128, 0:128])
    tensor_57 = a[128:256, 384:512]
    tensor_58 = b[128:256, 0:128]
    tensor_59 = nkigym.nc_matmul(tensor_57[0:128, 0:128], tensor_58[0:128, 0:128], acc=tensor_56[0:128, 0:128])
    output[384:512, 0:128] = tensor_59[0:128, 0:128]
    tensor_60 = a[0:128, 384:512]
    tensor_61 = b[0:128, 128:256]
    tensor_62 = nkigym.nc_matmul(tensor_60[0:128, 0:128], tensor_61[0:128, 0:128])
    tensor_63 = a[128:256, 384:512]
    tensor_64 = b[128:256, 128:256]
    tensor_65 = nkigym.nc_matmul(tensor_63[0:128, 0:128], tensor_64[0:128, 0:128], acc=tensor_62[0:128, 0:128])
    output[384:512, 128:256] = tensor_65[0:128, 0:128]
    tensor_66 = a[0:128, 384:512]
    tensor_67 = b[0:128, 256:384]
    tensor_68 = nkigym.nc_matmul(tensor_66[0:128, 0:128], tensor_67[0:128, 0:128])
    tensor_69 = a[128:256, 384:512]
    tensor_70 = b[128:256, 256:384]
    tensor_71 = nkigym.nc_matmul(tensor_69[0:128, 0:128], tensor_70[0:128, 0:128], acc=tensor_68[0:128, 0:128])
    output[384:512, 256:384] = tensor_71[0:128, 0:128]
    tensor_72 = a[0:128, 512:640]
    tensor_73 = b[0:128, 0:128]
    tensor_74 = nkigym.nc_matmul(tensor_72[0:128, 0:128], tensor_73[0:128, 0:128])
    tensor_75 = a[128:256, 512:640]
    tensor_76 = b[128:256, 0:128]
    tensor_77 = nkigym.nc_matmul(tensor_75[0:128, 0:128], tensor_76[0:128, 0:128], acc=tensor_74[0:128, 0:128])
    output[512:640, 0:128] = tensor_77[0:128, 0:128]
    tensor_78 = a[0:128, 512:640]
    tensor_79 = b[0:128, 128:256]
    tensor_80 = nkigym.nc_matmul(tensor_78[0:128, 0:128], tensor_79[0:128, 0:128])
    tensor_81 = a[128:256, 512:640]
    tensor_82 = b[128:256, 128:256]
    tensor_83 = nkigym.nc_matmul(tensor_81[0:128, 0:128], tensor_82[0:128, 0:128], acc=tensor_80[0:128, 0:128])
    output[512:640, 128:256] = tensor_83[0:128, 0:128]
    tensor_84 = a[0:128, 512:640]
    tensor_85 = b[0:128, 256:384]
    tensor_86 = nkigym.nc_matmul(tensor_84[0:128, 0:128], tensor_85[0:128, 0:128])
    tensor_87 = a[128:256, 512:640]
    tensor_88 = b[128:256, 256:384]
    tensor_89 = nkigym.nc_matmul(tensor_87[0:128, 0:128], tensor_88[0:128, 0:128], acc=tensor_86[0:128, 0:128])
    output[512:640, 256:384] = tensor_89[0:128, 0:128]
    return output
"""


def flat_3x5_red2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 3x5 reduction-2 tiled matmul (5 rows, 3 cols, 2 reductions)."""
    output = np.empty((640, 384), dtype=np.float32)
    for i in range(5):
        for j in range(3):
            tensor_0 = a[0:128, i * 128 : (i + 1) * 128]
            tensor_1 = b[0:128, j * 128 : (j + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            tensor_3 = a[128:256, i * 128 : (i + 1) * 128]
            tensor_4 = b[128:256, j * 128 : (j + 1) * 128]
            tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
            output[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = tensor_5[0:128, 0:128]
    return output


flat_3x5_red2.__source__ = _SOURCE_FLAT_3X5_RED2

_SOURCE_FLAT_2X3_RED3 = """\
def flat_2x3_red3(a, b):
    output = np.empty((256, 384), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    tensor_6 = a[256:384, 0:128]
    tensor_7 = b[256:384, 0:128]
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128], acc=tensor_5[0:128, 0:128])
    output[0:128, 0:128] = tensor_8[0:128, 0:128]
    tensor_9 = a[0:128, 0:128]
    tensor_10 = b[0:128, 128:256]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    tensor_12 = a[128:256, 0:128]
    tensor_13 = b[128:256, 128:256]
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128], acc=tensor_11[0:128, 0:128])
    tensor_15 = a[256:384, 0:128]
    tensor_16 = b[256:384, 128:256]
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    output[0:128, 128:256] = tensor_17[0:128, 0:128]
    tensor_18 = a[0:128, 0:128]
    tensor_19 = b[0:128, 256:384]
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    tensor_21 = a[128:256, 0:128]
    tensor_22 = b[128:256, 256:384]
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128], acc=tensor_20[0:128, 0:128])
    tensor_24 = a[256:384, 0:128]
    tensor_25 = b[256:384, 256:384]
    tensor_26 = nkigym.nc_matmul(tensor_24[0:128, 0:128], tensor_25[0:128, 0:128], acc=tensor_23[0:128, 0:128])
    output[0:128, 256:384] = tensor_26[0:128, 0:128]
    tensor_27 = a[0:128, 128:256]
    tensor_28 = b[0:128, 0:128]
    tensor_29 = nkigym.nc_matmul(tensor_27[0:128, 0:128], tensor_28[0:128, 0:128])
    tensor_30 = a[128:256, 128:256]
    tensor_31 = b[128:256, 0:128]
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128], acc=tensor_29[0:128, 0:128])
    tensor_33 = a[256:384, 128:256]
    tensor_34 = b[256:384, 0:128]
    tensor_35 = nkigym.nc_matmul(tensor_33[0:128, 0:128], tensor_34[0:128, 0:128], acc=tensor_32[0:128, 0:128])
    output[128:256, 0:128] = tensor_35[0:128, 0:128]
    tensor_36 = a[0:128, 128:256]
    tensor_37 = b[0:128, 128:256]
    tensor_38 = nkigym.nc_matmul(tensor_36[0:128, 0:128], tensor_37[0:128, 0:128])
    tensor_39 = a[128:256, 128:256]
    tensor_40 = b[128:256, 128:256]
    tensor_41 = nkigym.nc_matmul(tensor_39[0:128, 0:128], tensor_40[0:128, 0:128], acc=tensor_38[0:128, 0:128])
    tensor_42 = a[256:384, 128:256]
    tensor_43 = b[256:384, 128:256]
    tensor_44 = nkigym.nc_matmul(tensor_42[0:128, 0:128], tensor_43[0:128, 0:128], acc=tensor_41[0:128, 0:128])
    output[128:256, 128:256] = tensor_44[0:128, 0:128]
    tensor_45 = a[0:128, 128:256]
    tensor_46 = b[0:128, 256:384]
    tensor_47 = nkigym.nc_matmul(tensor_45[0:128, 0:128], tensor_46[0:128, 0:128])
    tensor_48 = a[128:256, 128:256]
    tensor_49 = b[128:256, 256:384]
    tensor_50 = nkigym.nc_matmul(tensor_48[0:128, 0:128], tensor_49[0:128, 0:128], acc=tensor_47[0:128, 0:128])
    tensor_51 = a[256:384, 128:256]
    tensor_52 = b[256:384, 256:384]
    tensor_53 = nkigym.nc_matmul(tensor_51[0:128, 0:128], tensor_52[0:128, 0:128], acc=tensor_50[0:128, 0:128])
    output[128:256, 256:384] = tensor_53[0:128, 0:128]
    return output
"""


def flat_2x3_red3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Flat 2x3 reduction-3 tiled matmul (2 rows, 3 cols, 3 reductions)."""
    output = np.empty((256, 384), dtype=np.float32)
    for i in range(2):
        for j in range(3):
            tensor_0 = a[0:128, i * 128 : (i + 1) * 128]
            tensor_1 = b[0:128, j * 128 : (j + 1) * 128]
            tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
            tensor_3 = a[128:256, i * 128 : (i + 1) * 128]
            tensor_4 = b[128:256, j * 128 : (j + 1) * 128]
            tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
            tensor_6 = a[256:384, i * 128 : (i + 1) * 128]
            tensor_7 = b[256:384, j * 128 : (j + 1) * 128]
            tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128], acc=tensor_5[0:128, 0:128])
            output[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = tensor_8[0:128, 0:128]
    return output


flat_2x3_red3.__source__ = _SOURCE_FLAT_2X3_RED3
