"""
Test NKI CPU simulator correctness for before/after code examples
from the four single-loop-nest transform design docs.

All kernels compute C = lhs_T.T @ rhs (matmul) with different loop
structures, load placements, and buffer layouts. Every variant must
produce the same result as the numpy reference.

Group A (tpb=1): load_placement before/after + 6 loop orderings
  K=256 (2 tiles), M=256 (2 tiles), N=1024 (2 tiles)

Group B (tpb=2): tiles_per_block before/after + interleave before/after
  K=512 (4 tiles, tpb=2, 2 blocks), M=256 (2 tiles), N=1024 (2 tiles)
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
======== Test harness ========
"""


def run_kernel(
    kernel: Callable[..., None],
    lhs_t_np: NDArray[np.float32],
    rhs_np: NDArray[np.float32],
    output_shape: tuple[int, int],
) -> NDArray[np.float32]:
    """Run a kernel on the CPU simulator and return the output."""
    out = np.zeros(output_shape, dtype=np.float32)
    nki.simulate(kernel)(lhs_t_np, rhs_np, out)
    return out


def main() -> None:
    """Run all 11 transform doc kernels and compare against numpy reference."""
    np.random.seed(42)

    ka, ma, na = KT_A * 128, MT_A * 128, NT_A * 512
    lhs_a = np.random.randn(ka, ma).astype(np.float32)
    rhs_a = np.random.randn(ka, na).astype(np.float32)
    ref_a = lhs_a.T @ rhs_a

    kb, mb, nb = KT_B * 128, MT_B * 128, NT_B * 512
    lhs_b = np.random.randn(kb, mb).astype(np.float32)
    rhs_b = np.random.randn(kb, nb).astype(np.float32)
    ref_b = lhs_b.T @ rhs_b

    group_a_kernels: list[tuple[str, Callable[..., None]]] = [
        ("load_placement/before  (d0,d1,d2)", lp_before),
        ("load_placement/after   (rhs full d2)", lp_after),
        ("loop_reorder/(d0,d2,d1)", lr_d0d2d1),
        ("loop_reorder/(d1,d0,d2)", lr_d1d0d2),
        ("loop_reorder/(d1,d2,d0)", lr_d1d2d0),
        ("loop_reorder/(d2,d0,d1)", lr_d2d0d1),
        ("loop_reorder/(d2,d1,d0)", lr_d2d1d0),
    ]

    group_b_kernels: list[tuple[str, Callable[..., None]]] = [
        ("tiles_per_block/before (tpb=1)", tpb_before),
        ("tiles_per_block/after  (tpb=2)", tpb_after),
        ("interleave/before (depth 0)", interleave_before),
        ("interleave/after  (depth 2)", interleave_after),
    ]

    all_passed = True

    print("=" * 70)
    print("Group A: K=256, M=256, N=1024  (tpb=1)")
    print("=" * 70)
    for name, kernel in group_a_kernels:
        out = run_kernel(kernel, lhs_a, rhs_a, (ma, na))
        max_err = float(np.max(np.abs(out - ref_a)))
        ok = bool(np.allclose(out, ref_a, rtol=1e-4, atol=1e-4))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:40s}  max_err={max_err:.2e}")
        if not ok:
            all_passed = False

    print()
    print("=" * 70)
    print("Group B: K=512, M=256, N=1024  (tpb=2, 2 blocks)")
    print("=" * 70)
    for name, kernel in group_b_kernels:
        out = run_kernel(kernel, lhs_b, rhs_b, (mb, nb))
        max_err = float(np.max(np.abs(out - ref_b)))
        ok = bool(np.allclose(out, ref_b, rtol=1e-4, atol=1e-4))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:40s}  max_err={max_err:.2e}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("All 11 kernels PASSED.")
    else:
        print("Some kernels FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
