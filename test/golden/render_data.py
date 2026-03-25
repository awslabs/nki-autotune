"""Golden rendered NKI kernel strings for codegen render tests."""

from nkigym.schedule.types import DimSchedule, Schedule

RENDER_SCHEDULE_4 = Schedule(
    loop_order=(("d3", 0), ("d1", 0), ("d0", 0)),
    dim_schedules=(DimSchedule("d3", 256, 1), DimSchedule("d1", 128, 1), DimSchedule("d0", 128, 1)),
    op_placements=(2, 2),
)

RENDER_1 = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_kernel(a, b):
    hbm_tensor_0 = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(2):
        for i_1 in nl.affine_range(1):
            psum_tensor_0 = nl.ndarray((128, 1, 1, 256), dtype=nl.float32, buffer=nl.psum)
            for i_2 in nl.affine_range(2):
                sbuf_tensor_0 = nl.ndarray((128, 1, 1, 128), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], src=a[i_2 * 128:i_2 * 128 + 128, i_0 * 128:i_0 * 128 + 128])
                sbuf_tensor_1 = nl.ndarray((128, 1, 1, 256), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_1[0:128, 0:1, 0:1, 0:256], src=b[i_2 * 128:i_2 * 128 + 128, i_1 * 256:i_1 * 256 + 256])
                nisa.nc_matmul(dst=psum_tensor_0[0:128, 0:1, 0:1, 0:256], stationary=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], moving=sbuf_tensor_1[0:128, 0:1, 0:1, 0:256])
            sbuf_tensor_2 = nl.ndarray((128, 1, 1, 256), dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256], src=psum_tensor_0[0:128, 0:1, 0:1, 0:256])
            nisa.dma_copy(dst=hbm_tensor_0[i_0 * 128:i_0 * 128 + 128, i_1 * 256:i_1 * 256 + 256], src=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256])
    return hbm_tensor_0
"""

RENDER_2 = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_tanh_kernel(a, b):
    hbm_tensor_0 = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(2):
        for i_1 in nl.affine_range(1):
            psum_tensor_0 = nl.ndarray((128, 1, 1, 256), dtype=nl.float32, buffer=nl.psum)
            for i_2 in nl.affine_range(2):
                sbuf_tensor_0 = nl.ndarray((128, 1, 1, 128), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], src=a[i_2 * 128:i_2 * 128 + 128, i_0 * 128:i_0 * 128 + 128])
                sbuf_tensor_1 = nl.ndarray((128, 1, 1, 256), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_1[0:128, 0:1, 0:1, 0:256], src=b[i_2 * 128:i_2 * 128 + 128, i_1 * 256:i_1 * 256 + 256])
                nisa.nc_matmul(dst=psum_tensor_0[0:128, 0:1, 0:1, 0:256], stationary=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], moving=sbuf_tensor_1[0:128, 0:1, 0:1, 0:256])
            sbuf_tensor_2 = nl.ndarray((128, 1, 1, 256), dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256], src=psum_tensor_0[0:128, 0:1, 0:1, 0:256])
            sbuf_tensor_3 = nl.ndarray((128, 1, 1, 256), dtype=a.dtype, buffer=nl.sbuf)
            nisa.activation(dst=sbuf_tensor_3[0:128, 0:1, 0:1, 0:256], data=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256], op=nl.tanh)
            nisa.dma_copy(dst=hbm_tensor_0[i_0 * 128:i_0 * 128 + 128, i_1 * 256:i_1 * 256 + 256], src=sbuf_tensor_3[0:128, 0:1, 0:1, 0:256])
    return hbm_tensor_0
"""

RENDER_3 = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def add_kernel(x, y):
    hbm_tensor_0 = nl.ndarray((256, 256), dtype=x.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(2):
        for i_1 in nl.affine_range(2):
            sbuf_tensor_0 = nl.ndarray((128, 1, 1, 128), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], src=x[i_0 * 128:i_0 * 128 + 128, i_1 * 128:i_1 * 128 + 128])
            sbuf_tensor_1 = nl.ndarray((128, 1, 1, 128), dtype=y.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sbuf_tensor_1[0:128, 0:1, 0:1, 0:128], src=y[i_0 * 128:i_0 * 128 + 128, i_1 * 128:i_1 * 128 + 128])
            sbuf_tensor_2 = nl.ndarray((128, 1, 1, 128), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sbuf_tensor_2[0:128, 0:1, 0:1, 0:128], data1=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], data2=sbuf_tensor_1[0:128, 0:1, 0:1, 0:128], op=nl.add)
            nisa.dma_copy(dst=hbm_tensor_0[i_0 * 128:i_0 * 128 + 128, i_1 * 128:i_1 * 128 + 128], src=sbuf_tensor_2[0:128, 0:1, 0:1, 0:128])
    return hbm_tensor_0
"""

RENDER_4 = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_swapped_kernel(a, b):
    hbm_tensor_0 = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(1):
        for i_1 in nl.affine_range(2):
            psum_tensor_0 = nl.ndarray((128, 1, 1, 256), dtype=nl.float32, buffer=nl.psum)
            for i_2 in nl.affine_range(2):
                sbuf_tensor_0 = nl.ndarray((128, 1, 1, 128), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], src=a[i_2 * 128:i_2 * 128 + 128, i_1 * 128:i_1 * 128 + 128])
                sbuf_tensor_1 = nl.ndarray((128, 1, 1, 256), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_1[0:128, 0:1, 0:1, 0:256], src=b[i_2 * 128:i_2 * 128 + 128, i_0 * 256:i_0 * 256 + 256])
                nisa.nc_matmul(dst=psum_tensor_0[0:128, 0:1, 0:1, 0:256], stationary=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], moving=sbuf_tensor_1[0:128, 0:1, 0:1, 0:256])
            sbuf_tensor_2 = nl.ndarray((128, 1, 1, 256), dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256], src=psum_tensor_0[0:128, 0:1, 0:1, 0:256])
            nisa.dma_copy(dst=hbm_tensor_0[i_1 * 128:i_1 * 128 + 128, i_0 * 256:i_0 * 256 + 256], src=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256])
    return hbm_tensor_0
"""

RENDER_5 = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_add_kernel(a, b, bias):
    hbm_tensor_0 = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(2):
        for i_1 in nl.affine_range(1):
            psum_tensor_0 = nl.ndarray((128, 1, 1, 256), dtype=nl.float32, buffer=nl.psum)
            sbuf_tensor_0 = nl.ndarray((128, 1, 1, 256), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sbuf_tensor_0[0:128, 0:1, 0:1, 0:256], src=bias[i_0 * 128:i_0 * 128 + 128, i_1 * 256:i_1 * 256 + 256])
            for i_2 in nl.affine_range(2):
                sbuf_tensor_1 = nl.ndarray((128, 1, 1, 128), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_1[0:128, 0:1, 0:1, 0:128], src=a[i_2 * 128:i_2 * 128 + 128, i_0 * 128:i_0 * 128 + 128])
                sbuf_tensor_2 = nl.ndarray((128, 1, 1, 256), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256], src=b[i_2 * 128:i_2 * 128 + 128, i_1 * 256:i_1 * 256 + 256])
                nisa.nc_matmul(dst=psum_tensor_0[0:128, 0:1, 0:1, 0:256], stationary=sbuf_tensor_1[0:128, 0:1, 0:1, 0:128], moving=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256])
            sbuf_tensor_3 = nl.ndarray((128, 1, 1, 256), dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sbuf_tensor_3[0:128, 0:1, 0:1, 0:256], src=psum_tensor_0[0:128, 0:1, 0:1, 0:256])
            sbuf_tensor_4 = nl.ndarray((128, 1, 1, 256), dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=sbuf_tensor_4[0:128, 0:1, 0:1, 0:256], data1=sbuf_tensor_3[0:128, 0:1, 0:1, 0:256], data2=sbuf_tensor_0[0:128, 0:1, 0:1, 0:256], op=nl.add)
            nisa.dma_copy(dst=hbm_tensor_0[i_0 * 128:i_0 * 128 + 128, i_1 * 256:i_1 * 256 + 256], src=sbuf_tensor_4[0:128, 0:1, 0:1, 0:256])
    return hbm_tensor_0
"""

RENDER_SCHEDULE_6 = Schedule(
    loop_order=(("d1", 0), ("d0", 0), ("d3", 0)),
    dim_schedules=(DimSchedule("d1", 128, 1), DimSchedule("d0", 128, 1), DimSchedule("d3", 256, 1)),
    op_placements=(2, 2),
)

RENDER_6 = """\
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_red_middle_kernel(a, b):
    hbm_tensor_0 = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(2):
        psum_tensor_0 = nl.ndarray((128, 1, 1, 256), dtype=nl.float32, buffer=nl.psum)
        for i_1 in nl.affine_range(2):
            sbuf_tensor_0 = nl.ndarray((128, 1, 1, 128), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], src=a[i_1 * 128:i_1 * 128 + 128, i_0 * 128:i_0 * 128 + 128])
            for i_2 in nl.affine_range(1):
                sbuf_tensor_1 = nl.ndarray((128, 1, 1, 256), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_1[0:128, 0:1, 0:1, 0:256], src=b[i_1 * 128:i_1 * 128 + 128, i_2 * 256:i_2 * 256 + 256])
                nisa.nc_matmul(dst=psum_tensor_0[0:128, 0:1, i_2:i_2 + 1, 0:256], stationary=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], moving=sbuf_tensor_1[0:128, 0:1, 0:1, 0:256])
        sbuf_tensor_2 = nl.ndarray((128, 1, 1, 256), dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256], src=psum_tensor_0[0:128, 0:1, 0:1, 0:256])
        nisa.dma_copy(dst=hbm_tensor_0[i_0 * 128:i_0 * 128 + 128, 0:256], src=sbuf_tensor_2[0:128, 0:1, 0:1, 0:256])
    return hbm_tensor_0
"""

RENDER_7 = """\
import numpy as np
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def rmsnorm_matmul(a, b):
    hbm_tensor_0 = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(2):
        for i_1 in nl.affine_range(1):
            psum_tensor_0 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
            for i_2 in nl.affine_range(2):
                sbuf_tensor_0 = nl.ndarray((128, 1, 1, 128), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], src=a[i_0 * 128:i_0 * 128 + 128, i_2 * 128:i_2 * 128 + 128])
                sbuf_tensor_1 = nl.ndarray((128, 1, 1, 128), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sbuf_tensor_1[0:128, 0:1, 0:1, 0:128], op=nl.square, data=sbuf_tensor_0[0:128, 0:1, 0:1, 0:128], reduce_op=np.add, reduce_res=psum_tensor_0[0:128, 0:1])
            sbuf_tensor_2 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sbuf_tensor_2[0:128, 0:1], src=psum_tensor_0[0:128, 0:1])
            nisa.tensor_scalar(dst=sbuf_tensor_2[0:128, 0:1], data=sbuf_tensor_2[0:128, 0:1], operand0=0.00048828125, op0=nl.multiply, op1=nl.add, operand1=1e-06)
            nisa.activation(dst=sbuf_tensor_2[0:128, 0:1], data=sbuf_tensor_2[0:128, 0:1], op=nl.rsqrt)
            psum_tensor_1 = nl.ndarray((128, 1, 1, 256), dtype=nl.float32, buffer=nl.psum)
            for i_3 in nl.affine_range(2):
                sbuf_tensor_3 = nl.ndarray((128, 1, 1, 128), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_3[0:128, 0:1, 0:1, 0:128], src=a[i_0 * 128:i_0 * 128 + 128, i_3 * 128:i_3 * 128 + 128])
                sbuf_tensor_4 = nl.ndarray((128, 1, 1, 256), dtype=b.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=sbuf_tensor_4[0:128, 0:1, 0:1, 0:256], src=b[i_3 * 128:i_3 * 128 + 128, i_1 * 256:i_1 * 256 + 256])
                nisa.tensor_scalar(dst=sbuf_tensor_3[0:128, 0:1, 0:1, 0:128], data=sbuf_tensor_3[0:128, 0:1, 0:1, 0:128], operand0=sbuf_tensor_2[0:128, 0:1], op0=nl.multiply)
                psum_tensor_2 = nl.ndarray((128, 1, 1, 128), dtype=a.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=psum_tensor_2[0:128, 0:1, 0:1, 0:128], data=sbuf_tensor_3[0:128, 0:1, 0:1, 0:128])
                nisa.tensor_copy(dst=sbuf_tensor_3[0:128, 0:1, 0:1, 0:128], src=psum_tensor_2[0:128, 0:1, 0:1, 0:128])
                nisa.nc_matmul(dst=psum_tensor_1[0:128, 0:1, 0:1, 0:256], stationary=sbuf_tensor_3[0:128, 0:1, 0:1, 0:128], moving=sbuf_tensor_4[0:128, 0:1, 0:1, 0:256])
            sbuf_tensor_5 = nl.ndarray((128, 1, 1, 256), dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sbuf_tensor_5[0:128, 0:1, 0:1, 0:256], src=psum_tensor_1[0:128, 0:1, 0:1, 0:256])
            nisa.dma_copy(dst=hbm_tensor_0[i_0 * 128:i_0 * 128 + 128, i_1 * 256:i_1 * 256 + 256], src=sbuf_tensor_5[0:128, 0:1, 0:1, 0:256])
    return hbm_tensor_0
"""
