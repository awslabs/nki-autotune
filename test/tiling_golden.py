"""Golden source strings for tiling tests.

This module contains expected generated source code strings for various matmul
configurations. Used by test_tiling.py for validation.

Each golden string is a standalone variable named SINGLE_<a_shape>_<b_shape> or
DOUBLE_<a_shape>_<b_shape>_<c_shape>. The lookup dicts at the bottom map shape
tuples to these variables.
"""

SINGLE_128x128_128x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]

    return output
"""

SINGLE_128x256_128x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((256, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]

    tensor_3 = a[0:128, 128:256]
    tensor_4 = b[0:128, 0:128]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[128:256, 0:128] = tensor_5[0:128, 0:128]

    return output
"""

SINGLE_128x128_128x256 = """\
import numpy as np
import nkigym
def matmul(a, b):
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
"""

SINGLE_128x256_128x256 = """\
import numpy as np
import nkigym
def matmul(a, b):
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
"""

SINGLE_128x512_128x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
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
"""

SINGLE_256x128_256x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]

    return output
"""

SINGLE_256x256_256x256 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]

    tensor_5 = a[0:128, 0:128]
    tensor_6 = b[0:128, 128:256]
    tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_8 = a[128:256, 0:128]
    tensor_9 = b[128:256, 128:256]
    tensor_7[0:128, 0:128] += nkigym.nc_matmul(tensor_8[0:128, 0:128], tensor_9[0:128, 0:128])
    output[0:128, 128:256] = tensor_7[0:128, 0:128]

    tensor_10 = a[0:128, 128:256]
    tensor_11 = b[0:128, 0:128]
    tensor_12 = nkigym.nc_matmul(tensor_10[0:128, 0:128], tensor_11[0:128, 0:128])
    tensor_13 = a[128:256, 128:256]
    tensor_14 = b[128:256, 0:128]
    tensor_12[0:128, 0:128] += nkigym.nc_matmul(tensor_13[0:128, 0:128], tensor_14[0:128, 0:128])
    output[128:256, 0:128] = tensor_12[0:128, 0:128]

    tensor_15 = a[0:128, 128:256]
    tensor_16 = b[0:128, 128:256]
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    tensor_18 = a[128:256, 128:256]
    tensor_19 = b[128:256, 128:256]
    tensor_17[0:128, 0:128] += nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    output[128:256, 128:256] = tensor_17[0:128, 0:128]

    return output
"""

SINGLE_256x512_256x512 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((512, 512), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_5 = a[0:128, 0:128]
    tensor_6 = b[0:128, 128:256]
    tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_8 = a[128:256, 0:128]
    tensor_9 = b[128:256, 128:256]
    tensor_7[0:128, 0:128] += nkigym.nc_matmul(tensor_8[0:128, 0:128], tensor_9[0:128, 0:128])
    output[0:128, 128:256] = tensor_7[0:128, 0:128]
    tensor_10 = a[0:128, 0:128]
    tensor_11 = b[0:128, 256:384]
    tensor_12 = nkigym.nc_matmul(tensor_10[0:128, 0:128], tensor_11[0:128, 0:128])
    tensor_13 = a[128:256, 0:128]
    tensor_14 = b[128:256, 256:384]
    tensor_12[0:128, 0:128] += nkigym.nc_matmul(tensor_13[0:128, 0:128], tensor_14[0:128, 0:128])
    output[0:128, 256:384] = tensor_12[0:128, 0:128]
    tensor_15 = a[0:128, 0:128]
    tensor_16 = b[0:128, 384:512]
    tensor_17 = nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    tensor_18 = a[128:256, 0:128]
    tensor_19 = b[128:256, 384:512]
    tensor_17[0:128, 0:128] += nkigym.nc_matmul(tensor_18[0:128, 0:128], tensor_19[0:128, 0:128])
    output[0:128, 384:512] = tensor_17[0:128, 0:128]
    tensor_20 = a[0:128, 128:256]
    tensor_21 = b[0:128, 0:128]
    tensor_22 = nkigym.nc_matmul(tensor_20[0:128, 0:128], tensor_21[0:128, 0:128])
    tensor_23 = a[128:256, 128:256]
    tensor_24 = b[128:256, 0:128]
    tensor_22[0:128, 0:128] += nkigym.nc_matmul(tensor_23[0:128, 0:128], tensor_24[0:128, 0:128])
    output[128:256, 0:128] = tensor_22[0:128, 0:128]
    tensor_25 = a[0:128, 128:256]
    tensor_26 = b[0:128, 128:256]
    tensor_27 = nkigym.nc_matmul(tensor_25[0:128, 0:128], tensor_26[0:128, 0:128])
    tensor_28 = a[128:256, 128:256]
    tensor_29 = b[128:256, 128:256]
    tensor_27[0:128, 0:128] += nkigym.nc_matmul(tensor_28[0:128, 0:128], tensor_29[0:128, 0:128])
    output[128:256, 128:256] = tensor_27[0:128, 0:128]
    tensor_30 = a[0:128, 128:256]
    tensor_31 = b[0:128, 256:384]
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128])
    tensor_33 = a[128:256, 128:256]
    tensor_34 = b[128:256, 256:384]
    tensor_32[0:128, 0:128] += nkigym.nc_matmul(tensor_33[0:128, 0:128], tensor_34[0:128, 0:128])
    output[128:256, 256:384] = tensor_32[0:128, 0:128]
    tensor_35 = a[0:128, 128:256]
    tensor_36 = b[0:128, 384:512]
    tensor_37 = nkigym.nc_matmul(tensor_35[0:128, 0:128], tensor_36[0:128, 0:128])
    tensor_38 = a[128:256, 128:256]
    tensor_39 = b[128:256, 384:512]
    tensor_37[0:128, 0:128] += nkigym.nc_matmul(tensor_38[0:128, 0:128], tensor_39[0:128, 0:128])
    output[128:256, 384:512] = tensor_37[0:128, 0:128]
    tensor_40 = a[0:128, 256:384]
    tensor_41 = b[0:128, 0:128]
    tensor_42 = nkigym.nc_matmul(tensor_40[0:128, 0:128], tensor_41[0:128, 0:128])
    tensor_43 = a[128:256, 256:384]
    tensor_44 = b[128:256, 0:128]
    tensor_42[0:128, 0:128] += nkigym.nc_matmul(tensor_43[0:128, 0:128], tensor_44[0:128, 0:128])
    output[256:384, 0:128] = tensor_42[0:128, 0:128]
    tensor_45 = a[0:128, 256:384]
    tensor_46 = b[0:128, 128:256]
    tensor_47 = nkigym.nc_matmul(tensor_45[0:128, 0:128], tensor_46[0:128, 0:128])
    tensor_48 = a[128:256, 256:384]
    tensor_49 = b[128:256, 128:256]
    tensor_47[0:128, 0:128] += nkigym.nc_matmul(tensor_48[0:128, 0:128], tensor_49[0:128, 0:128])
    output[256:384, 128:256] = tensor_47[0:128, 0:128]
    tensor_50 = a[0:128, 256:384]
    tensor_51 = b[0:128, 256:384]
    tensor_52 = nkigym.nc_matmul(tensor_50[0:128, 0:128], tensor_51[0:128, 0:128])
    tensor_53 = a[128:256, 256:384]
    tensor_54 = b[128:256, 256:384]
    tensor_52[0:128, 0:128] += nkigym.nc_matmul(tensor_53[0:128, 0:128], tensor_54[0:128, 0:128])
    output[256:384, 256:384] = tensor_52[0:128, 0:128]
    tensor_55 = a[0:128, 256:384]
    tensor_56 = b[0:128, 384:512]
    tensor_57 = nkigym.nc_matmul(tensor_55[0:128, 0:128], tensor_56[0:128, 0:128])
    tensor_58 = a[128:256, 256:384]
    tensor_59 = b[128:256, 384:512]
    tensor_57[0:128, 0:128] += nkigym.nc_matmul(tensor_58[0:128, 0:128], tensor_59[0:128, 0:128])
    output[256:384, 384:512] = tensor_57[0:128, 0:128]
    tensor_60 = a[0:128, 384:512]
    tensor_61 = b[0:128, 0:128]
    tensor_62 = nkigym.nc_matmul(tensor_60[0:128, 0:128], tensor_61[0:128, 0:128])
    tensor_63 = a[128:256, 384:512]
    tensor_64 = b[128:256, 0:128]
    tensor_62[0:128, 0:128] += nkigym.nc_matmul(tensor_63[0:128, 0:128], tensor_64[0:128, 0:128])
    output[384:512, 0:128] = tensor_62[0:128, 0:128]
    tensor_65 = a[0:128, 384:512]
    tensor_66 = b[0:128, 128:256]
    tensor_67 = nkigym.nc_matmul(tensor_65[0:128, 0:128], tensor_66[0:128, 0:128])
    tensor_68 = a[128:256, 384:512]
    tensor_69 = b[128:256, 128:256]
    tensor_67[0:128, 0:128] += nkigym.nc_matmul(tensor_68[0:128, 0:128], tensor_69[0:128, 0:128])
    output[384:512, 128:256] = tensor_67[0:128, 0:128]
    tensor_70 = a[0:128, 384:512]
    tensor_71 = b[0:128, 256:384]
    tensor_72 = nkigym.nc_matmul(tensor_70[0:128, 0:128], tensor_71[0:128, 0:128])
    tensor_73 = a[128:256, 384:512]
    tensor_74 = b[128:256, 256:384]
    tensor_72[0:128, 0:128] += nkigym.nc_matmul(tensor_73[0:128, 0:128], tensor_74[0:128, 0:128])
    output[384:512, 256:384] = tensor_72[0:128, 0:128]
    tensor_75 = a[0:128, 384:512]
    tensor_76 = b[0:128, 384:512]
    tensor_77 = nkigym.nc_matmul(tensor_75[0:128, 0:128], tensor_76[0:128, 0:128])
    tensor_78 = a[128:256, 384:512]
    tensor_79 = b[128:256, 384:512]
    tensor_77[0:128, 0:128] += nkigym.nc_matmul(tensor_78[0:128, 0:128], tensor_79[0:128, 0:128])
    output[384:512, 384:512] = tensor_77[0:128, 0:128]
    return output
"""

SINGLE_512x128_512x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    tensor_5 = a[256:384, 0:128]
    tensor_6 = b[256:384, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_7 = a[384:512, 0:128]
    tensor_8 = b[384:512, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_7[0:128, 0:128], tensor_8[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    return output
"""

SINGLE_128x512_128x512 = """\
import numpy as np
import nkigym
def matmul(a, b):
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

SINGLE_1024x128_1024x128 = """\
import numpy as np
import nkigym
def matmul(a, b):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = a[128:256, 0:128]
    tensor_4 = b[128:256, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    tensor_5 = a[256:384, 0:128]
    tensor_6 = b[256:384, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_7 = a[384:512, 0:128]
    tensor_8 = b[384:512, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_7[0:128, 0:128], tensor_8[0:128, 0:128])
    tensor_9 = a[512:640, 0:128]
    tensor_10 = b[512:640, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    tensor_11 = a[640:768, 0:128]
    tensor_12 = b[640:768, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_11[0:128, 0:128], tensor_12[0:128, 0:128])
    tensor_13 = a[768:896, 0:128]
    tensor_14 = b[768:896, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_13[0:128, 0:128], tensor_14[0:128, 0:128])
    tensor_15 = a[896:1024, 0:128]
    tensor_16 = b[896:1024, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]

    return output
"""

DOUBLE_128x128_128x128_128x128 = """\
import numpy as np
import nkigym
def double_matmul(a, b, c):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = c[0:128, 0:128]
    tensor_4 = nkigym.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128])
    output[0:128, 0:128] = tensor_4[0:128, 0:128]

    return output
"""

DOUBLE_128x256_128x128_256x128 = """\
import numpy as np
import nkigym
def double_matmul(a, b, c):
    output = np.empty((128, 128), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = c[0:128, 0:128]
    tensor_4 = nkigym.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128])
    tensor_5 = a[0:128, 128:256]
    tensor_6 = b[0:128, 0:128]
    tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_8 = c[128:256, 0:128]
    tensor_4[0:128, 0:128] += nkigym.nc_matmul(tensor_7[0:128, 0:128], tensor_8[0:128, 0:128])
    output[0:128, 0:128] = tensor_4[0:128, 0:128]

    return output
"""

DOUBLE_128x256_128x128_256x256 = """\
import numpy as np
import nkigym
def double_matmul(a, b, c):
    output = np.empty((128, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = c[0:128, 0:128]
    tensor_4 = nkigym.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128])
    tensor_5 = a[0:128, 128:256]
    tensor_6 = b[0:128, 0:128]
    tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_8 = c[128:256, 0:128]
    tensor_4[0:128, 0:128] += nkigym.nc_matmul(tensor_7[0:128, 0:128], tensor_8[0:128, 0:128])
    output[0:128, 0:128] = tensor_4[0:128, 0:128]

    tensor_9 = a[0:128, 0:128]
    tensor_10 = b[0:128, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    tensor_12 = c[0:128, 128:256]
    tensor_13 = nkigym.nc_matmul(tensor_11[0:128, 0:128], tensor_12[0:128, 0:128])
    tensor_14 = a[0:128, 128:256]
    tensor_15 = b[0:128, 0:128]
    tensor_16 = nkigym.nc_matmul(tensor_14[0:128, 0:128], tensor_15[0:128, 0:128])
    tensor_17 = c[128:256, 128:256]
    tensor_13[0:128, 0:128] += nkigym.nc_matmul(tensor_16[0:128, 0:128], tensor_17[0:128, 0:128])
    output[0:128, 128:256] = tensor_13[0:128, 0:128]

    return output
"""

DOUBLE_256x256_256x256_256x256 = """\
import numpy as np
import nkigym
def double_matmul(a, b, c):
    output = np.empty((256, 256), dtype=np.float32)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    tensor_3 = c[0:128, 0:128]
    tensor_4 = nkigym.nc_matmul(tensor_2[0:128, 0:128], tensor_3[0:128, 0:128])
    tensor_5 = a[0:128, 128:256]
    tensor_6 = b[0:128, 0:128]
    tensor_7 = nkigym.nc_matmul(tensor_5[0:128, 0:128], tensor_6[0:128, 0:128])
    tensor_8 = c[128:256, 0:128]
    tensor_4[0:128, 0:128] += nkigym.nc_matmul(tensor_7[0:128, 0:128], tensor_8[0:128, 0:128])
    tensor_9 = a[128:256, 0:128]
    tensor_10 = b[128:256, 0:128]
    tensor_11 = nkigym.nc_matmul(tensor_9[0:128, 0:128], tensor_10[0:128, 0:128])
    tensor_12 = c[0:128, 0:128]
    tensor_4[0:128, 0:128] += nkigym.nc_matmul(tensor_11[0:128, 0:128], tensor_12[0:128, 0:128])
    tensor_13 = a[128:256, 128:256]
    tensor_14 = b[128:256, 0:128]
    tensor_15 = nkigym.nc_matmul(tensor_13[0:128, 0:128], tensor_14[0:128, 0:128])
    tensor_16 = c[128:256, 0:128]
    tensor_4[0:128, 0:128] += nkigym.nc_matmul(tensor_15[0:128, 0:128], tensor_16[0:128, 0:128])
    output[0:128, 0:128] = tensor_4[0:128, 0:128]
    tensor_17 = a[0:128, 0:128]
    tensor_18 = b[0:128, 0:128]
    tensor_19 = nkigym.nc_matmul(tensor_17[0:128, 0:128], tensor_18[0:128, 0:128])
    tensor_20 = c[0:128, 128:256]
    tensor_21 = nkigym.nc_matmul(tensor_19[0:128, 0:128], tensor_20[0:128, 0:128])
    tensor_22 = a[0:128, 128:256]
    tensor_23 = b[0:128, 0:128]
    tensor_24 = nkigym.nc_matmul(tensor_22[0:128, 0:128], tensor_23[0:128, 0:128])
    tensor_25 = c[128:256, 128:256]
    tensor_21[0:128, 0:128] += nkigym.nc_matmul(tensor_24[0:128, 0:128], tensor_25[0:128, 0:128])
    tensor_26 = a[128:256, 0:128]
    tensor_27 = b[128:256, 0:128]
    tensor_28 = nkigym.nc_matmul(tensor_26[0:128, 0:128], tensor_27[0:128, 0:128])
    tensor_29 = c[0:128, 128:256]
    tensor_21[0:128, 0:128] += nkigym.nc_matmul(tensor_28[0:128, 0:128], tensor_29[0:128, 0:128])
    tensor_30 = a[128:256, 128:256]
    tensor_31 = b[128:256, 0:128]
    tensor_32 = nkigym.nc_matmul(tensor_30[0:128, 0:128], tensor_31[0:128, 0:128])
    tensor_33 = c[128:256, 128:256]
    tensor_21[0:128, 0:128] += nkigym.nc_matmul(tensor_32[0:128, 0:128], tensor_33[0:128, 0:128])
    output[0:128, 128:256] = tensor_21[0:128, 0:128]
    tensor_34 = a[0:128, 0:128]
    tensor_35 = b[0:128, 128:256]
    tensor_36 = nkigym.nc_matmul(tensor_34[0:128, 0:128], tensor_35[0:128, 0:128])
    tensor_37 = c[0:128, 0:128]
    tensor_38 = nkigym.nc_matmul(tensor_36[0:128, 0:128], tensor_37[0:128, 0:128])
    tensor_39 = a[0:128, 128:256]
    tensor_40 = b[0:128, 128:256]
    tensor_41 = nkigym.nc_matmul(tensor_39[0:128, 0:128], tensor_40[0:128, 0:128])
    tensor_42 = c[128:256, 0:128]
    tensor_38[0:128, 0:128] += nkigym.nc_matmul(tensor_41[0:128, 0:128], tensor_42[0:128, 0:128])
    tensor_43 = a[128:256, 0:128]
    tensor_44 = b[128:256, 128:256]
    tensor_45 = nkigym.nc_matmul(tensor_43[0:128, 0:128], tensor_44[0:128, 0:128])
    tensor_46 = c[0:128, 0:128]
    tensor_38[0:128, 0:128] += nkigym.nc_matmul(tensor_45[0:128, 0:128], tensor_46[0:128, 0:128])
    tensor_47 = a[128:256, 128:256]
    tensor_48 = b[128:256, 128:256]
    tensor_49 = nkigym.nc_matmul(tensor_47[0:128, 0:128], tensor_48[0:128, 0:128])
    tensor_50 = c[128:256, 0:128]
    tensor_38[0:128, 0:128] += nkigym.nc_matmul(tensor_49[0:128, 0:128], tensor_50[0:128, 0:128])
    output[128:256, 0:128] = tensor_38[0:128, 0:128]
    tensor_51 = a[0:128, 0:128]
    tensor_52 = b[0:128, 128:256]
    tensor_53 = nkigym.nc_matmul(tensor_51[0:128, 0:128], tensor_52[0:128, 0:128])
    tensor_54 = c[0:128, 128:256]
    tensor_55 = nkigym.nc_matmul(tensor_53[0:128, 0:128], tensor_54[0:128, 0:128])
    tensor_56 = a[0:128, 128:256]
    tensor_57 = b[0:128, 128:256]
    tensor_58 = nkigym.nc_matmul(tensor_56[0:128, 0:128], tensor_57[0:128, 0:128])
    tensor_59 = c[128:256, 128:256]
    tensor_55[0:128, 0:128] += nkigym.nc_matmul(tensor_58[0:128, 0:128], tensor_59[0:128, 0:128])
    tensor_60 = a[128:256, 0:128]
    tensor_61 = b[128:256, 128:256]
    tensor_62 = nkigym.nc_matmul(tensor_60[0:128, 0:128], tensor_61[0:128, 0:128])
    tensor_63 = c[0:128, 128:256]
    tensor_55[0:128, 0:128] += nkigym.nc_matmul(tensor_62[0:128, 0:128], tensor_63[0:128, 0:128])
    tensor_64 = a[128:256, 128:256]
    tensor_65 = b[128:256, 128:256]
    tensor_66 = nkigym.nc_matmul(tensor_64[0:128, 0:128], tensor_65[0:128, 0:128])
    tensor_67 = c[128:256, 128:256]
    tensor_55[0:128, 0:128] += nkigym.nc_matmul(tensor_66[0:128, 0:128], tensor_67[0:128, 0:128])
    output[128:256, 128:256] = tensor_55[0:128, 0:128]
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
