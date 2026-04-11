"""
NKI CPU simulator correctness tests for design doc code examples.

Group A (§6 transforms, tpb=1): load_placement before/after + 6 loop orderings
  C = lhs_T.T @ rhs, float32
  K=256 (2 tiles), M=256 (2 tiles), N=1024 (2 tiles)

Group B (§6 transforms, tpb=2): tiles_per_block before/after + interleave before/after
  C = lhs_T.T @ rhs, float32
  K=512 (4 tiles, tpb=2, 2 blocks), M=256 (2 tiles), N=1024 (2 tiles)

Group C (§5.4 naive lowering): transpose + matmul, one loop nest per op
  result = lhs_T.T @ rhs_T.T, float16
  d0=128 (1 tile), d1=2048 (16 tiles), d2=8192 (16 unified tiles, ig=4)
"""

import sys
from collections.abc import Callable
from typing import Any

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from numpy.typing import NDArray

from nkigym.gadgets import load_tensor_block, save_tensor_block

"""
======== Group A: load placement + loop reordering ========
K=256 (KT=2 tiles), M=256 (MT=2), N=1024 (NT=2)
"""

KT_A = 2
MT_A = 2
NT_A = 2


@nki.jit
def lp_before(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Load placement BEFORE: single on d2. Order (d0, d1, d2)."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, MT_A, NT_A, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=psum_out[0:128, 0:MT_A, 0:NT_A, 0:512], value=0.0)
    for i_d0 in range(KT_A):
        for i_d1 in range(MT_A):
            load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * 128, free_ofs=i_d1 * 128)
            for i_d2 in range(NT_A):
                load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * 128, free_ofs=i_d2 * 512)
                nisa.nc_matmul(
                    dst=psum_out[0:128, i_d1, i_d2, 0:512],
                    stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0, 0:512],
                )
    save_tensor_block(output, psum_out, par_ofs=0, free_ofs=0)


@nki.jit
def lp_after(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Load placement AFTER: rhs full on d2, hoisted past irrelevant d1."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, NT_A, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, MT_A, NT_A, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=psum_out[0:128, 0:MT_A, 0:NT_A, 0:512], value=0.0)
    for i_d0 in range(KT_A):
        load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * 128, free_ofs=0)
        for i_d1 in range(MT_A):
            load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * 128, free_ofs=i_d1 * 128)
            for i_d2 in range(NT_A):
                nisa.nc_matmul(
                    dst=psum_out[0:128, i_d1, i_d2, 0:512],
                    stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, i_d2, 0:512],
                )
    save_tensor_block(output, psum_out, par_ofs=0, free_ofs=0)


"""
======== Loop Reordering: 6 orderings ========
"""


@nki.jit
def lr_d0d2d1(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Order (d0, d2, d1). rhs reused 2x across d1."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, MT_A, NT_A, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=psum_out[0:128, 0:MT_A, 0:NT_A, 0:512], value=0.0)
    for i_d0 in range(KT_A):
        for i_d2 in range(NT_A):
            load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * 128, free_ofs=i_d2 * 512)
            for i_d1 in range(MT_A):
                load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * 128, free_ofs=i_d1 * 128)
                nisa.nc_matmul(
                    dst=psum_out[0:128, i_d1, i_d2, 0:512],
                    stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0, 0:512],
                )
    save_tensor_block(output, psum_out, par_ofs=0, free_ofs=0)


@nki.jit
def lr_d1d0d2(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Order (d1, d0, d2). d1 outside K, writeback per d1."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, 1, NT_A, 512), dtype=nl.float32, buffer=nl.psum)
    for i_d1 in range(MT_A):
        nisa.memset(dst=psum_out[0:128, 0, 0:NT_A, 0:512], value=0.0)
        for i_d0 in range(KT_A):
            load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * 128, free_ofs=i_d1 * 128)
            for i_d2 in range(NT_A):
                load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * 128, free_ofs=i_d2 * 512)
                nisa.nc_matmul(
                    dst=psum_out[0:128, 0, i_d2, 0:512],
                    stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0, 0:512],
                )
        save_tensor_block(output, psum_out, par_ofs=i_d1 * 128, free_ofs=0)


@nki.jit
def lr_d1d2d0(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Order (d1, d2, d0). Both output dims outside K, writeback per (d1,d2)."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
    for i_d1 in range(MT_A):
        for i_d2 in range(NT_A):
            nisa.memset(dst=psum_out[0:128, 0, 0, 0:512], value=0.0)
            for i_d0 in range(KT_A):
                load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * 128, free_ofs=i_d1 * 128)
                load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * 128, free_ofs=i_d2 * 512)
                nisa.nc_matmul(
                    dst=psum_out[0:128, 0, 0, 0:512],
                    stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0, 0:512],
                )
            save_tensor_block(output, psum_out, par_ofs=i_d1 * 128, free_ofs=i_d2 * 512)


@nki.jit
def lr_d2d0d1(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Order (d2, d0, d1). d2 outside K, writeback per d2."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, MT_A, 1, 512), dtype=nl.float32, buffer=nl.psum)
    for i_d2 in range(NT_A):
        nisa.memset(dst=psum_out[0:128, 0:MT_A, 0, 0:512], value=0.0)
        for i_d0 in range(KT_A):
            load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * 128, free_ofs=i_d2 * 512)
            for i_d1 in range(MT_A):
                load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * 128, free_ofs=i_d1 * 128)
                nisa.nc_matmul(
                    dst=psum_out[0:128, i_d1, 0, 0:512],
                    stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0, 0:512],
                )
        save_tensor_block(output, psum_out, par_ofs=0, free_ofs=i_d2 * 512)


@nki.jit
def lr_d2d1d0(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Order (d2, d1, d0). Both output dims outside K, writeback per (d2,d1)."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
    for i_d2 in range(NT_A):
        for i_d1 in range(MT_A):
            nisa.memset(dst=psum_out[0:128, 0, 0, 0:512], value=0.0)
            for i_d0 in range(KT_A):
                load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * 128, free_ofs=i_d1 * 128)
                load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * 128, free_ofs=i_d2 * 512)
                nisa.nc_matmul(
                    dst=psum_out[0:128, 0, 0, 0:512],
                    stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0, 0:512],
                )
            save_tensor_block(output, psum_out, par_ofs=i_d1 * 128, free_ofs=i_d2 * 512)


"""
======== Group B: tiles_per_block + interleave ========
K=512 (KT_B=4 tiles, TPB=2, NUM_BLK=2), M=256 (MT_B=2), N=1024 (NT_B=2)
"""

KT_B = 4
MT_B = 2
NT_B = 2
TPB = 2
NUM_BLK = 2


@nki.jit
def tpb_before(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Tiles-per-block BEFORE: tpb=1. 4 K tiles, order (d0, d1, d2)."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, MT_B, NT_B, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=psum_out[0:128, 0:MT_B, 0:NT_B, 0:512], value=0.0)
    for i_d0 in range(KT_B):
        for i_d1 in range(MT_B):
            load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * 128, free_ofs=i_d1 * 128)
            for i_d2 in range(NT_B):
                load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * 128, free_ofs=i_d2 * 512)
                nisa.nc_matmul(
                    dst=psum_out[0:128, i_d1, i_d2, 0:512],
                    stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                    moving=sbuf_rhs[0:128, 0, 0, 0:512],
                )
    save_tensor_block(output, psum_out, par_ofs=0, free_ofs=0)


@nki.jit
def tpb_after(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Tiles-per-block AFTER: tpb=2. 2 blocks of 2 tiles, order (d0, d1, d2)."""
    sbuf_lhs = nl.ndarray((128, TPB, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, TPB, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, MT_B, NT_B, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=psum_out[0:128, 0:MT_B, 0:NT_B, 0:512], value=0.0)
    for i_d0 in range(NUM_BLK):
        for i_d1 in range(MT_B):
            load_tensor_block(sbuf_lhs, lhs_T, par_ofs=i_d0 * TPB * 128, free_ofs=i_d1 * 128)
            for i_d2 in range(NT_B):
                load_tensor_block(sbuf_rhs, rhs, par_ofs=i_d0 * TPB * 128, free_ofs=i_d2 * 512)
                for i_k in range(TPB):
                    nisa.nc_matmul(
                        dst=psum_out[0:128, i_d1, i_d2, 0:512],
                        stationary=sbuf_lhs[0:128, i_k, 0, 0:128],
                        moving=sbuf_rhs[0:128, i_k, 0, 0:512],
                    )
    save_tensor_block(output, psum_out, par_ofs=0, free_ofs=0)


@nki.jit
def interleave_before(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Interleave BEFORE: depth 0 (block+tile adjacent). tpb=2, order (d0, d1, d2)."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, MT_B, NT_B, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=psum_out[0:128, 0:MT_B, 0:NT_B, 0:512], value=0.0)
    for i_block_d0 in range(NUM_BLK):
        for i_tile_d0 in range(TPB):
            for i_d1 in range(MT_B):
                k_ofs = (i_block_d0 * TPB + i_tile_d0) * 128
                load_tensor_block(sbuf_lhs, lhs_T, par_ofs=k_ofs, free_ofs=i_d1 * 128)
                for i_d2 in range(NT_B):
                    load_tensor_block(sbuf_rhs, rhs, par_ofs=k_ofs, free_ofs=i_d2 * 512)
                    nisa.nc_matmul(
                        dst=psum_out[0:128, i_d1, i_d2, 0:512],
                        stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                        moving=sbuf_rhs[0:128, 0, 0, 0:512],
                    )
    save_tensor_block(output, psum_out, par_ofs=0, free_ofs=0)


@nki.jit
def interleave_after(lhs_T: Any, rhs: Any, output: Any) -> None:
    """Interleave AFTER: depth 2 (d0 tile loop past d1 and d2). tpb=2."""
    sbuf_lhs = nl.ndarray((128, 1, 1, 128), dtype=lhs_T.dtype, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 1, 1, 512), dtype=rhs.dtype, buffer=nl.sbuf)
    psum_out = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)
    sbuf_out = nl.ndarray((128, MT_B, NT_B, 512), dtype=output.dtype, buffer=nl.sbuf)
    nisa.memset(dst=sbuf_out[0:128, 0:MT_B, 0:NT_B, 0:512], value=0.0)
    for i_block_d0 in range(NUM_BLK):
        for i_d1 in range(MT_B):
            for i_d2 in range(NT_B):
                nisa.tensor_copy(dst=psum_out[0:128, 0, 0, 0:512], src=sbuf_out[0:128, i_d1, i_d2, 0:512])
                for i_tile_d0 in range(TPB):
                    k_ofs = (i_block_d0 * TPB + i_tile_d0) * 128
                    load_tensor_block(sbuf_lhs, lhs_T, par_ofs=k_ofs, free_ofs=i_d1 * 128)
                    load_tensor_block(sbuf_rhs, rhs, par_ofs=k_ofs, free_ofs=i_d2 * 512)
                    nisa.nc_matmul(
                        dst=psum_out[0:128, 0, 0, 0:512],
                        stationary=sbuf_lhs[0:128, 0, 0, 0:128],
                        moving=sbuf_rhs[0:128, 0, 0, 0:512],
                    )
                nisa.tensor_copy(dst=sbuf_out[0:128, i_d1, i_d2, 0:512], src=psum_out[0:128, 0, 0, 0:512])
    save_tensor_block(output, sbuf_out, par_ofs=0, free_ofs=0)


"""
======== Group C: §5.4 naive lowering (transpose + matmul) ========
d0=128 (1 tile), d1=2048 (16 tiles), d2=8192 (16 unified tiles, ig=4)
result = lhs_T.T @ rhs_T.T, float16
"""


@nki.jit
def section5_naive(lhs_T: Any, rhs_T: Any, result: Any) -> None:
    """§5.4 lowered kernel: one loop nest per op, no fusion."""

    """
    Op 0: nc_transpose(rhs_T) → rhs
    sbuf_rhs at min tile size (128) per §4. Matmul reshapes (4,128)→(1,512).
    """
    sbuf_rhs_T = nl.ndarray((128, 4, 1, 128), dtype=nl.float16, buffer=nl.sbuf)
    psum_rhs_temp = nl.ndarray((128, 1, 1, 128), dtype=nl.float16, buffer=nl.psum)
    sbuf_rhs = nl.ndarray((128, 1, 64, 128), dtype=nl.float16, buffer=nl.sbuf)
    sbuf_rhs_op1 = sbuf_rhs.reshape((128, 1, 16, 512))

    for i_block_d2 in range(16):
        for i_block_d0 in range(1):
            load_tensor_block(sbuf_rhs_T, rhs_T, par_ofs=i_block_d2 * 512, free_ofs=i_block_d0 * 128)
            for i_tile_d2 in range(1):
                for i_tile_d0 in range(1):
                    for i_ig_d2 in range(4):
                        for i_ig_d0 in range(1):
                            ld2 = i_tile_d2 * 4 + i_ig_d2
                            td0 = i_tile_d0 * 1 + i_ig_d0
                            gd2 = i_block_d2 * 4 + ld2
                            nisa.nc_transpose(psum_rhs_temp[0:128, td0, 0, 0:128], sbuf_rhs_T[0:128, ld2, td0, 0:128])
                            nisa.tensor_copy(sbuf_rhs[0:128, td0, gd2, 0:128], psum_rhs_temp[0:128, td0, 0, 0:128])

    """
    Op 1: nc_matmul(lhs_T, rhs) → result
    """
    sbuf_lhs_T = nl.ndarray((128, 1, 1, 128), dtype=nl.float16, buffer=nl.sbuf)
    psum_result = nl.ndarray((128, 1, 1, 512), dtype=nl.float32, buffer=nl.psum)

    for i_block_d1 in range(16):
        for i_block_d2 in range(16):
            for i_block_d0 in range(1):
                load_tensor_block(sbuf_lhs_T, lhs_T, par_ofs=i_block_d0 * 128, free_ofs=i_block_d1 * 128)
                nisa.memset(psum_result[0:128, 0, 0, 0:512], value=0.0)
                for i_tile_d1 in range(1):
                    for i_tile_d2 in range(1):
                        for i_tile_d0 in range(1):
                            for i_ig_d1 in range(1):
                                for i_ig_d2 in range(1):
                                    for i_ig_d0 in range(1):
                                        td0 = i_tile_d0 * 1 + i_ig_d0
                                        td1 = i_tile_d1 * 1 + i_ig_d1
                                        d2_ut = i_block_d2 + i_tile_d2
                                        nisa.nc_matmul(
                                            psum_result[0:128, td1, i_ig_d2, 0:512],
                                            sbuf_lhs_T[0:128, td0, td1, 0:128],
                                            sbuf_rhs_op1[0:128, td0, d2_ut, 0:512],
                                        )
                save_tensor_block(result, psum_result, par_ofs=i_block_d1 * 128, free_ofs=i_block_d2 * 512)


"""
======== Test harness ========
"""


def run_matmul_kernel(
    kernel: Callable[..., None],
    lhs_t_np: NDArray[np.float32],
    rhs_np: NDArray[np.float32],
    output_shape: tuple[int, int],
) -> NDArray[np.float32]:
    """Run a 3-arg matmul kernel on the CPU simulator."""
    out = np.zeros(output_shape, dtype=np.float32)
    nki.simulate(kernel)(lhs_t_np, rhs_np, out)
    return out


def main() -> None:
    """Run all design doc kernels and compare against numpy reference."""
    np.random.seed(42)

    """Group A data"""
    ka, ma, na = KT_A * 128, MT_A * 128, NT_A * 512
    lhs_a = np.random.randn(ka, ma).astype(np.float32)
    rhs_a = np.random.randn(ka, na).astype(np.float32)
    ref_a = lhs_a.T @ rhs_a

    """Group B data"""
    kb, mb, nb = KT_B * 128, MT_B * 128, NT_B * 512
    lhs_b = np.random.randn(kb, mb).astype(np.float32)
    rhs_b = np.random.randn(kb, nb).astype(np.float32)
    ref_b = lhs_b.T @ rhs_b

    """Group C data"""
    lhs_T_c = np.random.randn(128, 2048).astype(np.float16)
    rhs_T_c = np.random.randn(8192, 128).astype(np.float16)
    ref_c = (lhs_T_c.astype(np.float64).T @ rhs_T_c.astype(np.float64).T).astype(np.float16)

    group_a: list[tuple[str, Callable[..., None]]] = [
        ("load_placement/before  (d0,d1,d2)", lp_before),
        ("load_placement/after   (rhs full d2)", lp_after),
        ("loop_reorder/(d0,d2,d1)", lr_d0d2d1),
        ("loop_reorder/(d1,d0,d2)", lr_d1d0d2),
        ("loop_reorder/(d1,d2,d0)", lr_d1d2d0),
        ("loop_reorder/(d2,d0,d1)", lr_d2d0d1),
        ("loop_reorder/(d2,d1,d0)", lr_d2d1d0),
    ]

    group_b: list[tuple[str, Callable[..., None]]] = [
        ("tiles_per_block/before (tpb=1)", tpb_before),
        ("tiles_per_block/after  (tpb=2)", tpb_after),
        ("interleave/before (depth 0)", interleave_before),
        ("interleave/after  (depth 2)", interleave_after),
    ]

    all_passed = True

    print("=" * 70)
    print("Group A: §6 transforms — K=256, M=256, N=1024 (tpb=1)")
    print("=" * 70)
    for name, kernel in group_a:
        out = run_matmul_kernel(kernel, lhs_a, rhs_a, (ma, na))
        max_err = float(np.max(np.abs(out - ref_a)))
        ok = bool(np.allclose(out, ref_a, rtol=1e-4, atol=1e-4))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:40s}  max_err={max_err:.2e}")
        if not ok:
            all_passed = False

    print()
    print("=" * 70)
    print("Group B: §6 transforms — K=512, M=256, N=1024 (tpb=2, 2 blocks)")
    print("=" * 70)
    for name, kernel in group_b:
        out = run_matmul_kernel(kernel, lhs_b, rhs_b, (mb, nb))
        max_err = float(np.max(np.abs(out - ref_b)))
        ok = bool(np.allclose(out, ref_b, rtol=1e-4, atol=1e-4))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:40s}  max_err={max_err:.2e}")
        if not ok:
            all_passed = False

    print()
    print("=" * 70)
    print("Group C: §5.4 naive lowering — transpose+matmul, float16")
    print("=" * 70)
    result_c = np.zeros((2048, 8192), dtype=np.float16)
    nki.simulate(section5_naive)(lhs_T_c, rhs_T_c, result_c)
    max_err = float(np.max(np.abs(result_c.astype(np.float64) - ref_c.astype(np.float64))))
    ok = bool(np.allclose(result_c, ref_c, rtol=1e-2, atol=1e-2))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {'section5/naive (transpose+matmul)':40s}  max_err={max_err:.2e}")
    if not ok:
        all_passed = False

    print()
    if all_passed:
        print("All 12 kernels PASSED.")
    else:
        print("Some kernels FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
