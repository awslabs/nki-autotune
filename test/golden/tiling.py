"""Golden source strings for tiling tests.

This module contains expected generated source code strings for various matmul
configurations. Used by test_tiling.py for validation.

Each golden string is a standalone variable named SINGLE_<a_shape>_<b_shape> or
DOUBLE_<a_shape>_<b_shape>_<c_shape>. The lookup dicts at the bottom map shape
tuples to these variables.
"""

import numpy as np

from nkigym.ir import GymProgram, GymStatement, TensorRef
from nkigym.ops import AllocateOp, LoadOp, MatmulOp, StoreOp


def _kw(shapes: dict) -> dict:
    """Create zero-filled kwargs from shape dict."""
    return {k: np.zeros(v, dtype=np.float32) for k, v in shapes.items()}


SINGLE_128x128_128x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
    return output

"""

SINGLE_128x256_128x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((256, 128), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
    tensor_3 = nkigym.load(a[0:128, 128:256])
    tensor_4 = nkigym.load(b[0:128, 0:128])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    nkigym.store(tensor_5[0:128, 0:128], output[128:256, 0:128])
    return output

"""

SINGLE_128x128_128x256 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 256), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
    tensor_3 = nkigym.load(a[0:128, 0:128])
    tensor_4 = nkigym.load(b[0:128, 128:256])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    nkigym.store(tensor_5[0:128, 0:128], output[0:128, 128:256])
    return output

"""

SINGLE_128x256_128x256 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
    tensor_3 = nkigym.load(a[0:128, 0:128])
    tensor_4 = nkigym.load(b[0:128, 128:256])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    nkigym.store(tensor_5[0:128, 0:128], output[0:128, 128:256])
    tensor_6 = nkigym.load(a[0:128, 128:256])
    tensor_7 = nkigym.load(b[0:128, 0:128])
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    nkigym.store(tensor_8[0:128, 0:128], output[128:256, 0:128])
    tensor_9 = nkigym.load(a[0:128, 128:256])
    tensor_10 = nkigym.load(b[0:128, 128:256])
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    nkigym.store(tensor_11[0:128, 0:128], output[128:256, 128:256])
    return output

"""

SINGLE_128x512_128x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((512, 128), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
    tensor_3 = nkigym.load(a[0:128, 128:256])
    tensor_4 = nkigym.load(b[0:128, 0:128])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    nkigym.store(tensor_5[0:128, 0:128], output[128:256, 0:128])
    tensor_6 = nkigym.load(a[0:128, 256:384])
    tensor_7 = nkigym.load(b[0:128, 0:128])
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    nkigym.store(tensor_8[0:128, 0:128], output[256:384, 0:128])
    tensor_9 = nkigym.load(a[0:128, 384:512])
    tensor_10 = nkigym.load(b[0:128, 0:128])
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    nkigym.store(tensor_11[0:128, 0:128], output[384:512, 0:128])
    return output

"""

SINGLE_256x128_256x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(a[128:256, 0:128])
    tensor_4 = nkigym.load(b[128:256, 0:128])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    nkigym.store(tensor_5[0:128, 0:128], output[0:128, 0:128])
    return output

"""

SINGLE_256x256_256x256 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(a[128:256, 0:128])
    tensor_4 = nkigym.load(b[128:256, 0:128])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    nkigym.store(tensor_5[0:128, 0:128], output[0:128, 0:128])
    tensor_6 = nkigym.load(a[0:128, 0:128])
    tensor_7 = nkigym.load(b[0:128, 128:256])
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    tensor_9 = nkigym.load(a[128:256, 0:128])
    tensor_10 = nkigym.load(b[128:256, 128:256])
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
    nkigym.store(tensor_11[0:128, 0:128], output[0:128, 128:256])
    tensor_12 = nkigym.load(a[0:128, 128:256])
    tensor_13 = nkigym.load(b[0:128, 0:128])
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    tensor_15 = nkigym.load(a[128:256, 128:256])
    tensor_16 = nkigym.load(b[128:256, 0:128])
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    nkigym.store(tensor_17[0:128, 0:128], output[128:256, 0:128])
    tensor_18 = nkigym.load(a[0:128, 128:256])
    tensor_19 = nkigym.load(b[0:128, 128:256])
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    tensor_21 = nkigym.load(a[128:256, 128:256])
    tensor_22 = nkigym.load(b[128:256, 128:256])
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128], acc=tensor_20[0:128, 0:128])
    nkigym.store(tensor_23[0:128, 0:128], output[128:256, 128:256])
    return output

"""

SINGLE_256x512_256x512 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((512, 512), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(a[128:256, 0:128])
    tensor_4 = nkigym.load(b[128:256, 0:128])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    nkigym.store(tensor_5[0:128, 0:128], output[0:128, 0:128])
    tensor_6 = nkigym.load(a[0:128, 0:128])
    tensor_7 = nkigym.load(b[0:128, 128:256])
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    tensor_9 = nkigym.load(a[128:256, 0:128])
    tensor_10 = nkigym.load(b[128:256, 128:256])
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
    nkigym.store(tensor_11[0:128, 0:128], output[0:128, 128:256])
    tensor_12 = nkigym.load(a[0:128, 0:128])
    tensor_13 = nkigym.load(b[0:128, 256:384])
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    tensor_15 = nkigym.load(a[128:256, 0:128])
    tensor_16 = nkigym.load(b[128:256, 256:384])
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    nkigym.store(tensor_17[0:128, 0:128], output[0:128, 256:384])
    tensor_18 = nkigym.load(a[0:128, 0:128])
    tensor_19 = nkigym.load(b[0:128, 384:512])
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    tensor_21 = nkigym.load(a[128:256, 0:128])
    tensor_22 = nkigym.load(b[128:256, 384:512])
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128], acc=tensor_20[0:128, 0:128])
    nkigym.store(tensor_23[0:128, 0:128], output[0:128, 384:512])
    tensor_24 = nkigym.load(a[0:128, 128:256])
    tensor_25 = nkigym.load(b[0:128, 0:128])
    tensor_26 = nkigym.nc_matmul(tensor_24[0:128, 0:128], tensor_25[0:128, 0:128])
    tensor_27 = nkigym.load(a[128:256, 128:256])
    tensor_28 = nkigym.load(b[128:256, 0:128])
    tensor_29 = nkigym.nc_matmul(tensor_27[0:128, 0:128], tensor_28[0:128, 0:128], acc=tensor_26[0:128, 0:128])
    nkigym.store(tensor_29[0:128, 0:128], output[128:256, 0:128])
    tensor_30 = nkigym.load(a[0:128, 128:256])
    tensor_31 = nkigym.load(b[0:128, 128:256])
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128])
    tensor_33 = nkigym.load(a[128:256, 128:256])
    tensor_34 = nkigym.load(b[128:256, 128:256])
    tensor_35 = nkigym.nc_matmul(tensor_33[0:128, 0:128], tensor_34[0:128, 0:128], acc=tensor_32[0:128, 0:128])
    nkigym.store(tensor_35[0:128, 0:128], output[128:256, 128:256])
    tensor_36 = nkigym.load(a[0:128, 128:256])
    tensor_37 = nkigym.load(b[0:128, 256:384])
    tensor_38 = nkigym.nc_matmul(tensor_36[0:128, 0:128], tensor_37[0:128, 0:128])
    tensor_39 = nkigym.load(a[128:256, 128:256])
    tensor_40 = nkigym.load(b[128:256, 256:384])
    tensor_41 = nkigym.nc_matmul(tensor_39[0:128, 0:128], tensor_40[0:128, 0:128], acc=tensor_38[0:128, 0:128])
    nkigym.store(tensor_41[0:128, 0:128], output[128:256, 256:384])
    tensor_42 = nkigym.load(a[0:128, 128:256])
    tensor_43 = nkigym.load(b[0:128, 384:512])
    tensor_44 = nkigym.nc_matmul(tensor_42[0:128, 0:128], tensor_43[0:128, 0:128])
    tensor_45 = nkigym.load(a[128:256, 128:256])
    tensor_46 = nkigym.load(b[128:256, 384:512])
    tensor_47 = nkigym.nc_matmul(tensor_45[0:128, 0:128], tensor_46[0:128, 0:128], acc=tensor_44[0:128, 0:128])
    nkigym.store(tensor_47[0:128, 0:128], output[128:256, 384:512])
    tensor_48 = nkigym.load(a[0:128, 256:384])
    tensor_49 = nkigym.load(b[0:128, 0:128])
    tensor_50 = nkigym.nc_matmul(tensor_48[0:128, 0:128], tensor_49[0:128, 0:128])
    tensor_51 = nkigym.load(a[128:256, 256:384])
    tensor_52 = nkigym.load(b[128:256, 0:128])
    tensor_53 = nkigym.nc_matmul(tensor_51[0:128, 0:128], tensor_52[0:128, 0:128], acc=tensor_50[0:128, 0:128])
    nkigym.store(tensor_53[0:128, 0:128], output[256:384, 0:128])
    tensor_54 = nkigym.load(a[0:128, 256:384])
    tensor_55 = nkigym.load(b[0:128, 128:256])
    tensor_56 = nkigym.nc_matmul(tensor_54[0:128, 0:128], tensor_55[0:128, 0:128])
    tensor_57 = nkigym.load(a[128:256, 256:384])
    tensor_58 = nkigym.load(b[128:256, 128:256])
    tensor_59 = nkigym.nc_matmul(tensor_57[0:128, 0:128], tensor_58[0:128, 0:128], acc=tensor_56[0:128, 0:128])
    nkigym.store(tensor_59[0:128, 0:128], output[256:384, 128:256])
    tensor_60 = nkigym.load(a[0:128, 256:384])
    tensor_61 = nkigym.load(b[0:128, 256:384])
    tensor_62 = nkigym.nc_matmul(tensor_60[0:128, 0:128], tensor_61[0:128, 0:128])
    tensor_63 = nkigym.load(a[128:256, 256:384])
    tensor_64 = nkigym.load(b[128:256, 256:384])
    tensor_65 = nkigym.nc_matmul(tensor_63[0:128, 0:128], tensor_64[0:128, 0:128], acc=tensor_62[0:128, 0:128])
    nkigym.store(tensor_65[0:128, 0:128], output[256:384, 256:384])
    tensor_66 = nkigym.load(a[0:128, 256:384])
    tensor_67 = nkigym.load(b[0:128, 384:512])
    tensor_68 = nkigym.nc_matmul(tensor_66[0:128, 0:128], tensor_67[0:128, 0:128])
    tensor_69 = nkigym.load(a[128:256, 256:384])
    tensor_70 = nkigym.load(b[128:256, 384:512])
    tensor_71 = nkigym.nc_matmul(tensor_69[0:128, 0:128], tensor_70[0:128, 0:128], acc=tensor_68[0:128, 0:128])
    nkigym.store(tensor_71[0:128, 0:128], output[256:384, 384:512])
    tensor_72 = nkigym.load(a[0:128, 384:512])
    tensor_73 = nkigym.load(b[0:128, 0:128])
    tensor_74 = nkigym.nc_matmul(tensor_72[0:128, 0:128], tensor_73[0:128, 0:128])
    tensor_75 = nkigym.load(a[128:256, 384:512])
    tensor_76 = nkigym.load(b[128:256, 0:128])
    tensor_77 = nkigym.nc_matmul(tensor_75[0:128, 0:128], tensor_76[0:128, 0:128], acc=tensor_74[0:128, 0:128])
    nkigym.store(tensor_77[0:128, 0:128], output[384:512, 0:128])
    tensor_78 = nkigym.load(a[0:128, 384:512])
    tensor_79 = nkigym.load(b[0:128, 128:256])
    tensor_80 = nkigym.nc_matmul(tensor_78[0:128, 0:128], tensor_79[0:128, 0:128])
    tensor_81 = nkigym.load(a[128:256, 384:512])
    tensor_82 = nkigym.load(b[128:256, 128:256])
    tensor_83 = nkigym.nc_matmul(tensor_81[0:128, 0:128], tensor_82[0:128, 0:128], acc=tensor_80[0:128, 0:128])
    nkigym.store(tensor_83[0:128, 0:128], output[384:512, 128:256])
    tensor_84 = nkigym.load(a[0:128, 384:512])
    tensor_85 = nkigym.load(b[0:128, 256:384])
    tensor_86 = nkigym.nc_matmul(tensor_84[0:128, 0:128], tensor_85[0:128, 0:128])
    tensor_87 = nkigym.load(a[128:256, 384:512])
    tensor_88 = nkigym.load(b[128:256, 256:384])
    tensor_89 = nkigym.nc_matmul(tensor_87[0:128, 0:128], tensor_88[0:128, 0:128], acc=tensor_86[0:128, 0:128])
    nkigym.store(tensor_89[0:128, 0:128], output[384:512, 256:384])
    tensor_90 = nkigym.load(a[0:128, 384:512])
    tensor_91 = nkigym.load(b[0:128, 384:512])
    tensor_92 = nkigym.nc_matmul(tensor_90[0:128, 0:128], tensor_91[0:128, 0:128])
    tensor_93 = nkigym.load(a[128:256, 384:512])
    tensor_94 = nkigym.load(b[128:256, 384:512])
    tensor_95 = nkigym.nc_matmul(tensor_93[0:128, 0:128], tensor_94[0:128, 0:128], acc=tensor_92[0:128, 0:128])
    nkigym.store(tensor_95[0:128, 0:128], output[384:512, 384:512])
    return output

"""

SINGLE_512x128_512x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(a[128:256, 0:128])
    tensor_4 = nkigym.load(b[128:256, 0:128])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    tensor_6 = nkigym.load(a[256:384, 0:128])
    tensor_7 = nkigym.load(b[256:384, 0:128])
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128], acc=tensor_5[0:128, 0:128])
    tensor_9 = nkigym.load(a[384:512, 0:128])
    tensor_10 = nkigym.load(b[384:512, 0:128])
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
    nkigym.store(tensor_11[0:128, 0:128], output[0:128, 0:128])
    return output

"""

SINGLE_128x512_128x512 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((512, 512), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
    tensor_3 = nkigym.load(a[0:128, 0:128])
    tensor_4 = nkigym.load(b[0:128, 128:256])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    nkigym.store(tensor_5[0:128, 0:128], output[0:128, 128:256])
    tensor_6 = nkigym.load(a[0:128, 0:128])
    tensor_7 = nkigym.load(b[0:128, 256:384])
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128])
    nkigym.store(tensor_8[0:128, 0:128], output[0:128, 256:384])
    tensor_9 = nkigym.load(a[0:128, 0:128])
    tensor_10 = nkigym.load(b[0:128, 384:512])
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    nkigym.store(tensor_11[0:128, 0:128], output[0:128, 384:512])
    tensor_12 = nkigym.load(a[0:128, 128:256])
    tensor_13 = nkigym.load(b[0:128, 0:128])
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    nkigym.store(tensor_14[0:128, 0:128], output[128:256, 0:128])
    tensor_15 = nkigym.load(a[0:128, 128:256])
    tensor_16 = nkigym.load(b[0:128, 128:256])
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    nkigym.store(tensor_17[0:128, 0:128], output[128:256, 128:256])
    tensor_18 = nkigym.load(a[0:128, 128:256])
    tensor_19 = nkigym.load(b[0:128, 256:384])
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    nkigym.store(tensor_20[0:128, 0:128], output[128:256, 256:384])
    tensor_21 = nkigym.load(a[0:128, 128:256])
    tensor_22 = nkigym.load(b[0:128, 384:512])
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128])
    nkigym.store(tensor_23[0:128, 0:128], output[128:256, 384:512])
    tensor_24 = nkigym.load(a[0:128, 256:384])
    tensor_25 = nkigym.load(b[0:128, 0:128])
    tensor_26 = nkigym.nc_matmul(tensor_24[0:128, 0:128], tensor_25[0:128, 0:128])
    nkigym.store(tensor_26[0:128, 0:128], output[256:384, 0:128])
    tensor_27 = nkigym.load(a[0:128, 256:384])
    tensor_28 = nkigym.load(b[0:128, 128:256])
    tensor_29 = nkigym.nc_matmul(tensor_27[0:128, 0:128], tensor_28[0:128, 0:128])
    nkigym.store(tensor_29[0:128, 0:128], output[256:384, 128:256])
    tensor_30 = nkigym.load(a[0:128, 256:384])
    tensor_31 = nkigym.load(b[0:128, 256:384])
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128])
    nkigym.store(tensor_32[0:128, 0:128], output[256:384, 256:384])
    tensor_33 = nkigym.load(a[0:128, 256:384])
    tensor_34 = nkigym.load(b[0:128, 384:512])
    tensor_35 = nkigym.nc_matmul(tensor_33[0:128, 0:128], tensor_34[0:128, 0:128])
    nkigym.store(tensor_35[0:128, 0:128], output[256:384, 384:512])
    tensor_36 = nkigym.load(a[0:128, 384:512])
    tensor_37 = nkigym.load(b[0:128, 0:128])
    tensor_38 = nkigym.nc_matmul(tensor_36[0:128, 0:128], tensor_37[0:128, 0:128])
    nkigym.store(tensor_38[0:128, 0:128], output[384:512, 0:128])
    tensor_39 = nkigym.load(a[0:128, 384:512])
    tensor_40 = nkigym.load(b[0:128, 128:256])
    tensor_41 = nkigym.nc_matmul(tensor_39[0:128, 0:128], tensor_40[0:128, 0:128])
    nkigym.store(tensor_41[0:128, 0:128], output[384:512, 128:256])
    tensor_42 = nkigym.load(a[0:128, 384:512])
    tensor_43 = nkigym.load(b[0:128, 256:384])
    tensor_44 = nkigym.nc_matmul(tensor_42[0:128, 0:128], tensor_43[0:128, 0:128])
    nkigym.store(tensor_44[0:128, 0:128], output[384:512, 256:384])
    tensor_45 = nkigym.load(a[0:128, 384:512])
    tensor_46 = nkigym.load(b[0:128, 384:512])
    tensor_47 = nkigym.nc_matmul(tensor_45[0:128, 0:128], tensor_46[0:128, 0:128])
    nkigym.store(tensor_47[0:128, 0:128], output[384:512, 384:512])
    return output

"""

SINGLE_1024x128_1024x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(a[128:256, 0:128])
    tensor_4 = nkigym.load(b[128:256, 0:128])
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128], acc=tensor_2[0:128, 0:128])
    tensor_6 = nkigym.load(a[256:384, 0:128])
    tensor_7 = nkigym.load(b[256:384, 0:128])
    tensor_8 = nkigym.nc_matmul(tensor_6[0:128, 0:128], tensor_7[0:128, 0:128], acc=tensor_5[0:128, 0:128])
    tensor_9 = nkigym.load(a[384:512, 0:128])
    tensor_10 = nkigym.load(b[384:512, 0:128])
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128], acc=tensor_8[0:128, 0:128])
    tensor_12 = nkigym.load(a[512:640, 0:128])
    tensor_13 = nkigym.load(b[512:640, 0:128])
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128], acc=tensor_11[0:128, 0:128])
    tensor_15 = nkigym.load(a[640:768, 0:128])
    tensor_16 = nkigym.load(b[640:768, 0:128])
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    tensor_18 = nkigym.load(a[768:896, 0:128])
    tensor_19 = nkigym.load(b[768:896, 0:128])
    tensor_20 = nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128], acc=tensor_17[0:128, 0:128])
    tensor_21 = nkigym.load(a[896:1024, 0:128])
    tensor_22 = nkigym.load(b[896:1024, 0:128])
    tensor_23 = nkigym.nc_matmul(tensor_21[0:128, 0:128], tensor_22[0:128, 0:128], acc=tensor_20[0:128, 0:128])
    nkigym.store(tensor_23[0:128, 0:128], output[0:128, 0:128])
    return output

"""

DOUBLE_128x128_128x128_128x128 = """\
import numpy as np
import nkigym
def double_matmul(a, b, c):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(c[0:128, 0:128])
    tensor_4 = nkigym.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128])
    nkigym.store(tensor_4[0:128, 0:128], output[0:128, 0:128])
    return output

"""

DOUBLE_128x256_128x128_256x128 = """\
import numpy as np
import nkigym
def double_matmul(a, b, c):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(c[0:128, 0:128])
    tensor_4 = nkigym.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128])
    tensor_5 = nkigym.load(a[0:128, 128:256])
    tensor_6 = nkigym.load(b[0:128, 0:128])
    tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_8 = nkigym.load(c[128:256, 0:128])
    tensor_9 = nkigym.nc_matmul(tensor_7[0:128, 0:128], tensor_8[0:128, 0:128], acc=tensor_4[0:128, 0:128])
    nkigym.store(tensor_9[0:128, 0:128], output[0:128, 0:128])
    return output

"""

DOUBLE_128x256_128x128_256x256 = """\
import numpy as np
import nkigym
def double_matmul(a, b, c):
    output = np.empty((128, 256), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(c[0:128, 0:128])
    tensor_4 = nkigym.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128])
    tensor_5 = nkigym.load(a[0:128, 128:256])
    tensor_6 = nkigym.load(b[0:128, 0:128])
    tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_8 = nkigym.load(c[128:256, 0:128])
    tensor_9 = nkigym.nc_matmul(tensor_7[0:128, 0:128], tensor_8[0:128, 0:128], acc=tensor_4[0:128, 0:128])
    nkigym.store(tensor_9[0:128, 0:128], output[0:128, 0:128])
    tensor_10 = nkigym.load(a[0:128, 0:128])
    tensor_11 = nkigym.load(b[0:128, 0:128])
    tensor_12 = nkigym.nc_matmul(tensor_10[0:128, 0:128], tensor_11[0:128, 0:128])
    tensor_13 = nkigym.load(c[0:128, 128:256])
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128])
    tensor_15 = nkigym.load(a[0:128, 128:256])
    tensor_16 = nkigym.load(b[0:128, 0:128])
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    tensor_18 = nkigym.load(c[128:256, 128:256])
    tensor_19 = nkigym.nc_matmul(tensor_17[0:128, 0:128], tensor_18[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    nkigym.store(tensor_19[0:128, 0:128], output[0:128, 128:256])
    return output

"""

DOUBLE_256x256_256x256_256x256 = """\
import numpy as np
import nkigym
def double_matmul(a, b, c):
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = nkigym.load(a[0:128, 0:128])
    tensor_1 = nkigym.load(b[0:128, 0:128])
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = nkigym.load(c[0:128, 0:128])
    tensor_4 = nkigym.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128])
    tensor_5 = nkigym.load(a[0:128, 128:256])
    tensor_6 = nkigym.load(b[0:128, 0:128])
    tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_8 = nkigym.load(c[128:256, 0:128])
    tensor_9 = nkigym.nc_matmul(tensor_7[0:128, 0:128], tensor_8[0:128, 0:128], acc=tensor_4[0:128, 0:128])
    tensor_10 = nkigym.load(a[128:256, 0:128])
    tensor_11 = nkigym.load(b[128:256, 0:128])
    tensor_12 = nkigym.nc_matmul(tensor_10[0:128, 0:128], tensor_11[0:128, 0:128])
    tensor_13 = nkigym.load(c[0:128, 0:128])
    tensor_14 = nkigym.nc_matmul(tensor_12[0:128, 0:128], tensor_13[0:128, 0:128], acc=tensor_9[0:128, 0:128])
    tensor_15 = nkigym.load(a[128:256, 128:256])
    tensor_16 = nkigym.load(b[128:256, 0:128])
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    tensor_18 = nkigym.load(c[128:256, 0:128])
    tensor_19 = nkigym.nc_matmul(tensor_17[0:128, 0:128], tensor_18[0:128, 0:128], acc=tensor_14[0:128, 0:128])
    nkigym.store(tensor_19[0:128, 0:128], output[0:128, 0:128])
    tensor_20 = nkigym.load(a[0:128, 0:128])
    tensor_21 = nkigym.load(b[0:128, 0:128])
    tensor_22 = nkigym.nc_matmul(tensor_20[0:128, 0:128], tensor_21[0:128, 0:128])
    tensor_23 = nkigym.load(c[0:128, 128:256])
    tensor_24 = nkigym.nc_matmul(tensor_22[0:128, 0:128], tensor_23[0:128, 0:128])
    tensor_25 = nkigym.load(a[0:128, 128:256])
    tensor_26 = nkigym.load(b[0:128, 0:128])
    tensor_27 = nkigym.nc_matmul(tensor_25[0:128, 0:128], tensor_26[0:128, 0:128])
    tensor_28 = nkigym.load(c[128:256, 128:256])
    tensor_29 = nkigym.nc_matmul(tensor_27[0:128, 0:128], tensor_28[0:128, 0:128], acc=tensor_24[0:128, 0:128])
    tensor_30 = nkigym.load(a[128:256, 0:128])
    tensor_31 = nkigym.load(b[128:256, 0:128])
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128])
    tensor_33 = nkigym.load(c[0:128, 128:256])
    tensor_34 = nkigym.nc_matmul(tensor_32[0:128, 0:128], tensor_33[0:128, 0:128], acc=tensor_29[0:128, 0:128])
    tensor_35 = nkigym.load(a[128:256, 128:256])
    tensor_36 = nkigym.load(b[128:256, 0:128])
    tensor_37 = nkigym.nc_matmul(tensor_35[0:128, 0:128], tensor_36[0:128, 0:128])
    tensor_38 = nkigym.load(c[128:256, 128:256])
    tensor_39 = nkigym.nc_matmul(tensor_37[0:128, 0:128], tensor_38[0:128, 0:128], acc=tensor_34[0:128, 0:128])
    nkigym.store(tensor_39[0:128, 0:128], output[0:128, 128:256])
    tensor_40 = nkigym.load(a[0:128, 0:128])
    tensor_41 = nkigym.load(b[0:128, 128:256])
    tensor_42 = nkigym.nc_matmul(tensor_40[0:128, 0:128], tensor_41[0:128, 0:128])
    tensor_43 = nkigym.load(c[0:128, 0:128])
    tensor_44 = nkigym.nc_matmul(tensor_42[0:128, 0:128], tensor_43[0:128, 0:128])
    tensor_45 = nkigym.load(a[0:128, 128:256])
    tensor_46 = nkigym.load(b[0:128, 128:256])
    tensor_47 = nkigym.nc_matmul(tensor_45[0:128, 0:128], tensor_46[0:128, 0:128])
    tensor_48 = nkigym.load(c[128:256, 0:128])
    tensor_49 = nkigym.nc_matmul(tensor_47[0:128, 0:128], tensor_48[0:128, 0:128], acc=tensor_44[0:128, 0:128])
    tensor_50 = nkigym.load(a[128:256, 0:128])
    tensor_51 = nkigym.load(b[128:256, 128:256])
    tensor_52 = nkigym.nc_matmul(tensor_50[0:128, 0:128], tensor_51[0:128, 0:128])
    tensor_53 = nkigym.load(c[0:128, 0:128])
    tensor_54 = nkigym.nc_matmul(tensor_52[0:128, 0:128], tensor_53[0:128, 0:128], acc=tensor_49[0:128, 0:128])
    tensor_55 = nkigym.load(a[128:256, 128:256])
    tensor_56 = nkigym.load(b[128:256, 128:256])
    tensor_57 = nkigym.nc_matmul(tensor_55[0:128, 0:128], tensor_56[0:128, 0:128])
    tensor_58 = nkigym.load(c[128:256, 0:128])
    tensor_59 = nkigym.nc_matmul(tensor_57[0:128, 0:128], tensor_58[0:128, 0:128], acc=tensor_54[0:128, 0:128])
    nkigym.store(tensor_59[0:128, 0:128], output[128:256, 0:128])
    tensor_60 = nkigym.load(a[0:128, 0:128])
    tensor_61 = nkigym.load(b[0:128, 128:256])
    tensor_62 = nkigym.nc_matmul(tensor_60[0:128, 0:128], tensor_61[0:128, 0:128])
    tensor_63 = nkigym.load(c[0:128, 128:256])
    tensor_64 = nkigym.nc_matmul(tensor_62[0:128, 0:128], tensor_63[0:128, 0:128])
    tensor_65 = nkigym.load(a[0:128, 128:256])
    tensor_66 = nkigym.load(b[0:128, 128:256])
    tensor_67 = nkigym.nc_matmul(tensor_65[0:128, 0:128], tensor_66[0:128, 0:128])
    tensor_68 = nkigym.load(c[128:256, 128:256])
    tensor_69 = nkigym.nc_matmul(tensor_67[0:128, 0:128], tensor_68[0:128, 0:128], acc=tensor_64[0:128, 0:128])
    tensor_70 = nkigym.load(a[128:256, 0:128])
    tensor_71 = nkigym.load(b[128:256, 128:256])
    tensor_72 = nkigym.nc_matmul(tensor_70[0:128, 0:128], tensor_71[0:128, 0:128])
    tensor_73 = nkigym.load(c[0:128, 128:256])
    tensor_74 = nkigym.nc_matmul(tensor_72[0:128, 0:128], tensor_73[0:128, 0:128], acc=tensor_69[0:128, 0:128])
    tensor_75 = nkigym.load(a[128:256, 128:256])
    tensor_76 = nkigym.load(b[128:256, 128:256])
    tensor_77 = nkigym.nc_matmul(tensor_75[0:128, 0:128], tensor_76[0:128, 0:128])
    tensor_78 = nkigym.load(c[128:256, 128:256])
    tensor_79 = nkigym.nc_matmul(tensor_77[0:128, 0:128], tensor_78[0:128, 0:128], acc=tensor_74[0:128, 0:128])
    nkigym.store(tensor_79[0:128, 0:128], output[128:256, 128:256])
    return output

"""

GOLDEN_SINGLE_MATMUL_SOURCE = {
    ((128, 128), (128, 128)): SINGLE_128x128_128x128,
    ((128, 256), (128, 128)): SINGLE_128x256_128x128,
    ((128, 128), (128, 256)): SINGLE_128x128_128x256,
    ((128, 256), (128, 256)): SINGLE_128x256_128x256,
    ((128, 512), (128, 128)): SINGLE_128x512_128x128,
    ((256, 128), (256, 128)): SINGLE_256x128_256x128,
    ((256, 256), (256, 256)): SINGLE_256x256_256x256,
    ((256, 512), (256, 512)): SINGLE_256x512_256x512,
    ((512, 128), (512, 128)): SINGLE_512x128_512x128,
    ((128, 512), (128, 512)): SINGLE_128x512_128x512,
    ((1024, 128), (1024, 128)): SINGLE_1024x128_1024x128,
}

GOLDEN_DOUBLE_MATMUL_SOURCE = {
    ((128, 128), (128, 128), (128, 128)): DOUBLE_128x128_128x128_128x128,
    ((128, 256), (128, 128), (256, 128)): DOUBLE_128x256_128x128_256x128,
    ((128, 256), (128, 128), (256, 256)): DOUBLE_128x256_128x128_256x256,
    ((256, 256), (256, 256), (256, 256)): DOUBLE_256x256_256x256_256x256,
}

GOLDEN_SINGLE_MATMUL_PROGRAM = {
    ((128, 128), (128, 128)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (128, 128), "b": (128, 128)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (128, 128), ((0, 128), (0, 128))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((128, 256), (128, 128)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (128, 256), "b": (128, 128)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 128), ((0, 256), (0, 128)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (256, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 128), ((128, 256), (0, 128)))),
                ),
                TensorRef("output", (256, 128), ((128, 256), (0, 128))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((128, 128), (128, 256)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (128, 128), "b": (128, 256)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (128, 256), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
                ),
                TensorRef("output", (128, 256), ((0, 128), (128, 256))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((128, 256), (128, 256)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (128, 256), "b": (128, 256)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (256, 256), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((0, 128), (128, 256)))),
                ),
                TensorRef("output", (256, 256), ((0, 128), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((128, 256), (0, 128)))),
                ),
                TensorRef("output", (256, 256), ((128, 256), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((128, 256), (128, 256)))),
                ),
                TensorRef("output", (256, 256), ((128, 256), (128, 256))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((128, 512), (128, 128)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (128, 512), "b": (128, 128)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (512, 128), ((0, 512), (0, 128)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (512, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 128), ((128, 256), (0, 128)))),
                ),
                TensorRef("output", (512, 128), ((128, 256), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 128), ((256, 384), (0, 128)))),
                ),
                TensorRef("output", (512, 128), ((256, 384), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 128), ((384, 512), (0, 128)))),
                ),
                TensorRef("output", (512, 128), ((384, 512), (0, 128))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((256, 128), (256, 128)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (256, 128), "b": (256, 128)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 128), ((128, 256), (0, 128)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (128, 128), ((0, 128), (0, 128))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((256, 256), (256, 256)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (256, 256), "b": (256, 256)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (256, 256), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((0, 128), (128, 256)))),
                ),
                TensorRef("output", (256, 256), ((0, 128), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((128, 256), (0, 128)))),
                ),
                TensorRef("output", (256, 256), ((128, 256), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((128, 256), (128, 256)))),
                ),
                TensorRef("output", (256, 256), ((128, 256), (128, 256))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((256, 512), (256, 512)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (256, 512), "b": (256, 512)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (512, 512), ((0, 512), (0, 512)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (0, 128)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (512, 512), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (0, 128)))),),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (128, 256)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((0, 128), (128, 256)))),
                ),
                TensorRef("output", (512, 512), ((0, 128), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (0, 128)))),),
                TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (256, 384)))),),
                TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((0, 128), (256, 384)))),
                ),
                TensorRef("output", (512, 512), ((0, 128), (256, 384))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (0, 128)))),),
                TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (384, 512)))),),
                TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((0, 128), (384, 512)))),
                ),
                TensorRef("output", (512, 512), ((0, 128), (384, 512))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_24", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_25", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_24", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_25", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_26", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (128, 256)))),),
                TensorRef("tensor_27", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (0, 128)))),),
                TensorRef("tensor_28", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_27", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_28", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_26", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_29", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_29", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((128, 256), (0, 128)))),
                ),
                TensorRef("output", (512, 512), ((128, 256), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_30", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_31", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_30", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_31", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_32", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (128, 256)))),),
                TensorRef("tensor_33", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (128, 256)))),),
                TensorRef("tensor_34", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_33", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_34", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_32", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_35", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_35", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((128, 256), (128, 256)))),
                ),
                TensorRef("output", (512, 512), ((128, 256), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_36", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_37", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_36", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_37", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_38", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (128, 256)))),),
                TensorRef("tensor_39", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (256, 384)))),),
                TensorRef("tensor_40", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_39", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_40", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_38", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_41", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_41", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((128, 256), (256, 384)))),
                ),
                TensorRef("output", (512, 512), ((128, 256), (256, 384))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_42", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_43", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_42", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_43", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_44", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (128, 256)))),),
                TensorRef("tensor_45", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (384, 512)))),),
                TensorRef("tensor_46", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_45", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_46", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_44", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_47", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_47", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((128, 256), (384, 512)))),
                ),
                TensorRef("output", (512, 512), ((128, 256), (384, 512))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_48", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_49", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_48", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_49", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_50", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (256, 384)))),),
                TensorRef("tensor_51", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (0, 128)))),),
                TensorRef("tensor_52", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_51", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_52", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_50", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_53", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_53", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((256, 384), (0, 128)))),
                ),
                TensorRef("output", (512, 512), ((256, 384), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_54", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_55", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_54", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_55", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_56", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (256, 384)))),),
                TensorRef("tensor_57", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (128, 256)))),),
                TensorRef("tensor_58", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_57", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_58", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_56", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_59", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_59", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((256, 384), (128, 256)))),
                ),
                TensorRef("output", (512, 512), ((256, 384), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_60", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_61", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_60", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_61", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_62", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (256, 384)))),),
                TensorRef("tensor_63", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (256, 384)))),),
                TensorRef("tensor_64", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_63", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_64", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_62", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_65", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_65", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((256, 384), (256, 384)))),
                ),
                TensorRef("output", (512, 512), ((256, 384), (256, 384))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_66", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_67", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_66", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_67", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_68", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (256, 384)))),),
                TensorRef("tensor_69", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (384, 512)))),),
                TensorRef("tensor_70", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_69", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_70", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_68", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_71", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_71", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((256, 384), (384, 512)))),
                ),
                TensorRef("output", (512, 512), ((256, 384), (384, 512))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_72", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_73", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_72", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_73", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_74", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (384, 512)))),),
                TensorRef("tensor_75", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (0, 128)))),),
                TensorRef("tensor_76", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_75", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_76", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_74", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_77", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_77", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((384, 512), (0, 128)))),
                ),
                TensorRef("output", (512, 512), ((384, 512), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_78", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_79", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_78", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_79", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_80", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (384, 512)))),),
                TensorRef("tensor_81", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (128, 256)))),),
                TensorRef("tensor_82", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_81", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_82", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_80", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_83", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_83", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((384, 512), (128, 256)))),
                ),
                TensorRef("output", (512, 512), ((384, 512), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_84", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_85", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_84", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_85", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_86", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (384, 512)))),),
                TensorRef("tensor_87", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (256, 384)))),),
                TensorRef("tensor_88", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_87", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_88", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_86", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_89", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_89", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((384, 512), (256, 384)))),
                ),
                TensorRef("output", (512, 512), ((384, 512), (256, 384))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_90", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_91", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_90", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_91", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_92", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 512), ((128, 256), (384, 512)))),),
                TensorRef("tensor_93", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 512), ((128, 256), (384, 512)))),),
                TensorRef("tensor_94", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_93", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_94", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_92", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_95", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_95", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((384, 512), (384, 512)))),
                ),
                TensorRef("output", (512, 512), ((384, 512), (384, 512))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((512, 128), (512, 128)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (512, 128), "b": (512, 128)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (512, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (512, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (512, 128), ((128, 256), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (512, 128), ((128, 256), (0, 128)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (512, 128), ((256, 384), (0, 128)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (512, 128), ((256, 384), (0, 128)))),),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (512, 128), ((384, 512), (0, 128)))),),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (512, 128), ((384, 512), (0, 128)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (128, 128), ((0, 128), (0, 128))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((128, 512), (128, 512)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (128, 512), "b": (128, 512)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (512, 512), ((0, 512), (0, 512)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (512, 512), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((0, 128), (128, 256)))),
                ),
                TensorRef("output", (512, 512), ((0, 128), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((0, 128), (256, 384)))),
                ),
                TensorRef("output", (512, 512), ((0, 128), (256, 384))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((0, 128), (384, 512)))),
                ),
                TensorRef("output", (512, 512), ((0, 128), (384, 512))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((128, 256), (0, 128)))),
                ),
                TensorRef("output", (512, 512), ((128, 256), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((128, 256), (128, 256)))),
                ),
                TensorRef("output", (512, 512), ((128, 256), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((128, 256), (256, 384)))),
                ),
                TensorRef("output", (512, 512), ((128, 256), (256, 384))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((128, 256), (384, 512)))),
                ),
                TensorRef("output", (512, 512), ((128, 256), (384, 512))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_24", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_25", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_24", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_25", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_26", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_26", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((256, 384), (0, 128)))),
                ),
                TensorRef("output", (512, 512), ((256, 384), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_27", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_28", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_27", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_28", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_29", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_29", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((256, 384), (128, 256)))),
                ),
                TensorRef("output", (512, 512), ((256, 384), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_30", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_31", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_30", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_31", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_32", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_32", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((256, 384), (256, 384)))),
                ),
                TensorRef("output", (512, 512), ((256, 384), (256, 384))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_33", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_34", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_33", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_34", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_35", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_35", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((256, 384), (384, 512)))),
                ),
                TensorRef("output", (512, 512), ((256, 384), (384, 512))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_36", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (0, 128)))),),
                TensorRef("tensor_37", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_36", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_37", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_38", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_38", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((384, 512), (0, 128)))),
                ),
                TensorRef("output", (512, 512), ((384, 512), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_39", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (128, 256)))),),
                TensorRef("tensor_40", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_39", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_40", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_41", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_41", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((384, 512), (128, 256)))),
                ),
                TensorRef("output", (512, 512), ((384, 512), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_42", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (256, 384)))),),
                TensorRef("tensor_43", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_42", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_43", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_44", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_44", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((384, 512), (256, 384)))),
                ),
                TensorRef("output", (512, 512), ((384, 512), (256, 384))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_45", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 512), ((0, 128), (384, 512)))),),
                TensorRef("tensor_46", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_45", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_46", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_47", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_47", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (512, 512), ((384, 512), (384, 512)))),
                ),
                TensorRef("output", (512, 512), ((384, 512), (384, 512))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((1024, 128), (1024, 128)): GymProgram(
        name="matmul",
        kwargs=_kw({"a": (1024, 128), "b": (1024, 128)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (1024, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (1024, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (1024, 128), ((128, 256), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (1024, 128), ((128, 256), (0, 128)))),),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (1024, 128), ((256, 384), (0, 128)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (1024, 128), ((256, 384), (0, 128)))),),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (1024, 128), ((384, 512), (0, 128)))),),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (1024, 128), ((384, 512), (0, 128)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (1024, 128), ((512, 640), (0, 128)))),),
                TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (1024, 128), ((512, 640), (0, 128)))),),
                TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (1024, 128), ((640, 768), (0, 128)))),),
                TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (1024, 128), ((640, 768), (0, 128)))),),
                TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (1024, 128), ((768, 896), (0, 128)))),),
                TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (1024, 128), ((768, 896), (0, 128)))),),
                TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (1024, 128), ((896, 1024), (0, 128)))),),
                TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (1024, 128), ((896, 1024), (0, 128)))),),
                TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (128, 128), ((0, 128), (0, 128))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
}

GOLDEN_DOUBLE_MATMUL_PROGRAM = {
    ((128, 128), (128, 128), (128, 128)): GymProgram(
        name="double_matmul",
        kwargs=_kw({"a": (128, 128), "b": (128, 128), "c": (128, 128)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (128, 128), ((0, 128), (0, 128))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((128, 256), (128, 128), (256, 128)): GymProgram(
        name="double_matmul",
        kwargs=_kw({"a": (128, 256), "b": (128, 128), "c": (256, 128)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 128), ((128, 256), (0, 128)))),),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (128, 128), ((0, 128), (0, 128))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((128, 256), (128, 128), (256, 256)): GymProgram(
        name="double_matmul",
        kwargs=_kw({"a": (128, 256), "b": (128, 128), "c": (256, 256)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (128, 256), ((0, 128), (0, 256)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 256), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (128, 256), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (128, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (128, 128), ((0, 128), (0, 128)))),),
                TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (128, 256), ((0, 128), (128, 256)))),
                ),
                TensorRef("output", (128, 256), ((0, 128), (128, 256))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
    ((256, 256), (256, 256), (256, 256)): GymProgram(
        name="double_matmul",
        kwargs=_kw({"a": (256, 256), "b": (256, 256), "c": (256, 256)}),
        stmts=(
            GymStatement(AllocateOp, (("dtype", np.float32),), TensorRef("output", (256, 256), ((0, 256), (0, 256)))),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_6", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_7", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_8", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_10", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_11", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_12", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_13", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_9", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_15", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_16", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_17", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_18", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_14", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_19", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((0, 128), (0, 128)))),
                ),
                TensorRef("output", (256, 256), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_20", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_21", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_22", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_23", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_24", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_25", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_26", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_25", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_26", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_27", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_28", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_27", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_28", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_24", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_29", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_30", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_31", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_30", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_31", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_32", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_33", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_32", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_33", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_29", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_34", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_35", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_36", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_35", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_36", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_37", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_38", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_37", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_38", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_34", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_39", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_39", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((0, 128), (128, 256)))),
                ),
                TensorRef("output", (256, 256), ((0, 128), (128, 256))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_40", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_41", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_40", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_41", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_42", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_43", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_42", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_43", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_44", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_45", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_46", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_45", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_46", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_47", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_48", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_47", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_48", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_44", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_49", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_50", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_51", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_50", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_51", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_52", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_53", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_52", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_53", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_49", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_54", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_55", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_56", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_55", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_56", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_57", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_58", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_57", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_58", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_54", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_59", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_59", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((128, 256), (0, 128)))),
                ),
                TensorRef("output", (256, 256), ((128, 256), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (0, 128)))),),
                TensorRef("tensor_60", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_61", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_60", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_61", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_62", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_63", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_62", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_63", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_64", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_65", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_66", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_65", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_66", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_67", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_68", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_67", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_68", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_64", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_69", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (0, 128)))),),
                TensorRef("tensor_70", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_71", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_70", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_71", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_72", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((0, 128), (128, 256)))),),
                TensorRef("tensor_73", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_72", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_73", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_69", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_74", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("a", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_75", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("b", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_76", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_75", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_76", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_77", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                LoadOp,
                (("src", TensorRef("c", (256, 256), ((128, 256), (128, 256)))),),
                TensorRef("tensor_78", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                MatmulOp,
                (
                    ("stationary", TensorRef("tensor_77", (128, 128), ((0, 128), (0, 128)))),
                    ("moving", TensorRef("tensor_78", (128, 128), ((0, 128), (0, 128)))),
                    ("acc", TensorRef("tensor_74", (128, 128), ((0, 128), (0, 128)))),
                ),
                TensorRef("tensor_79", (128, 128), ((0, 128), (0, 128))),
            ),
            GymStatement(
                StoreOp,
                (
                    ("src", TensorRef("tensor_79", (128, 128), ((0, 128), (0, 128)))),
                    ("dst", TensorRef("output", (256, 256), ((128, 256), (128, 256)))),
                ),
                TensorRef("output", (256, 256), ((128, 256), (128, 256))),
            ),
        ),
        return_var="output",
        output_dtype=np.float32,
    ),
}
