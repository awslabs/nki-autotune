"""Roofline cost model for Trn2 NeuronCore-v3.

Provides static arithmetic intensity analysis from NKI kernel IR to
distinguish memory-bound vs compute-bound variants and compute a
tighter theoretical ceiling per variant.

Hardware specs (Trn2 NeuronCore-v3):
    PE frequency: 2.4 GHz
    BF16/FP16 peak: 79 TFLOPS (2x128x128 FLOPS/cycle)
    FP8 peak: 158 TFLOPS (4x128x128 FLOPS/cycle)
    HBM bandwidth per NC: 375 GB/s
"""

import math
from typing import NamedTuple

from nkigym.codegen.types import NKIKernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.dma_copy import NKIDmaCopy

_PE_FREQ_HZ = 2.4e9

_PEAK_FLOPS_PER_CYCLE: dict[str, int] = {
    "nl.float8_e4m3fn": 4 * 128 * 128,
    "nl.float8_e5m2": 4 * 128 * 128,
    "nl.float16": 2 * 128 * 128,
    "nl.bfloat16": 2 * 128 * 128,
    "nl.float32": 2 * 128 * 128,
}

_HBM_BW_BYTES_PER_SEC = 375e9

_DTYPE_BYTES: dict[str, int] = {
    "nl.float8_e4m3fn": 1,
    "nl.float8_e5m2": 1,
    "nl.float16": 2,
    "nl.bfloat16": 2,
    "nl.float32": 4,
    "nl.float64": 8,
    "nl.int8": 1,
    "nl.int16": 2,
    "nl.int32": 4,
}


class RooflineAnalysis(NamedTuple):
    """Result of static roofline analysis on a kernel.

    Attributes:
        total_flops: Total floating-point operations (2 * mac_count).
        total_hbm_bytes: Total bytes transferred to/from HBM.
        arithmetic_intensity: FLOP/byte ratio.
        ridge_point: Arithmetic intensity where compute = memory ceiling.
        peak_tflops: Hardware peak TFLOPS for this dtype.
        roofline_peak_tflops: Theoretical max TFLOPS for this kernel.
        bound: Either ``"compute"`` or ``"memory"``.
    """

    total_flops: int
    total_hbm_bytes: int
    arithmetic_intensity: float
    ridge_point: float
    peak_tflops: float
    roofline_peak_tflops: float
    bound: str


def _dtype_bytes(dtype_str: str) -> int:
    """Return byte width for an NKI dtype string.

    Args:
        dtype_str: NKI dtype like ``"nl.float16"``.

    Returns:
        Number of bytes per element.

    Raises:
        ValueError: If dtype is not recognized.
    """
    result = _DTYPE_BYTES.get(dtype_str)
    if result is None:
        raise ValueError(f"Unknown NKI dtype: {dtype_str!r}")
    return result


def _collect_hbm_names(kernel: NKIKernel) -> set[str]:
    """Collect names of all tensors that live in HBM.

    HBM tensors include kernel input parameters, ``"output"``, and any
    ``NKIAlloc`` with ``buffer="shared_hbm"``.

    Args:
        kernel: The NKI kernel to analyze.

    Returns:
        Set of tensor names residing in HBM.
    """
    names: set[str] = set(kernel.params)
    names.add("output")
    for block in kernel.blocks:
        for stmt in block.body:
            if isinstance(stmt, NKIAlloc) and stmt.buffer == "shared_hbm":
                names.add(stmt.dst)
    return names


def _ref_transfer_bytes(ref_slices: tuple[tuple[int, int], ...], elem_bytes: int) -> int:
    """Compute bytes transferred for a tensor reference.

    Args:
        ref_slices: Per-axis (start, stop) slice bounds.
        elem_bytes: Bytes per element.

    Returns:
        Total bytes for the slice region.
    """
    elements = math.prod(stop - start for start, stop in ref_slices)
    return elements * elem_bytes


def _count_hbm_bytes(kernel: NKIKernel, hbm_names: set[str], elem_bytes: int) -> int:
    """Sum bytes across all DMA copies that touch HBM.

    A DMA copy touches HBM if either its source or destination name
    is in the HBM name set.

    Args:
        kernel: The NKI kernel to analyze.
        hbm_names: Set of tensor names residing in HBM.
        elem_bytes: Bytes per element for the kernel dtype.

    Returns:
        Total HBM transfer bytes.
    """
    total = 0
    for block in kernel.blocks:
        for stmt in block.body:
            if not isinstance(stmt, NKIDmaCopy):
                continue
            src_is_hbm = stmt.src.name in hbm_names
            dst_is_hbm = stmt.dst.name in hbm_names
            if src_is_hbm:
                total += _ref_transfer_bytes(stmt.src.slices, elem_bytes)
            if dst_is_hbm:
                total += _ref_transfer_bytes(stmt.dst.slices, elem_bytes)
    return total


def analyze_roofline(kernel: NKIKernel) -> RooflineAnalysis:
    """Perform static roofline analysis on an NKI kernel.

    Walks block bodies to compute total FLOPS and HBM transfer bytes,
    then derives arithmetic intensity, ridge point, and roofline peak.

    Args:
        kernel: The NKI kernel to analyze.

    Returns:
        A ``RooflineAnalysis`` with all computed fields.
    """
    total_flops = kernel.mac_count * 2
    elem_bytes = _dtype_bytes(kernel.dtype)
    hbm_names = _collect_hbm_names(kernel)
    total_hbm_bytes = _count_hbm_bytes(kernel, hbm_names, elem_bytes)

    flops_per_cycle = _PEAK_FLOPS_PER_CYCLE.get(kernel.dtype, 2 * 128 * 128)
    peak_tflops = _PE_FREQ_HZ * flops_per_cycle / 1e12
    ridge_point = peak_tflops * 1e12 / _HBM_BW_BYTES_PER_SEC

    if total_hbm_bytes == 0:
        arithmetic_intensity = float("inf")
        bound = "compute"
        roofline_peak_tflops = peak_tflops
    elif total_flops == 0:
        arithmetic_intensity = 0.0
        bound = "memory"
        roofline_peak_tflops = 0.0
    else:
        arithmetic_intensity = total_flops / total_hbm_bytes
        if arithmetic_intensity >= ridge_point:
            bound = "compute"
            roofline_peak_tflops = peak_tflops
        else:
            bound = "memory"
            roofline_peak_tflops = arithmetic_intensity * _HBM_BW_BYTES_PER_SEC / 1e12

    return RooflineAnalysis(
        total_flops=total_flops,
        total_hbm_bytes=total_hbm_bytes,
        arithmetic_intensity=arithmetic_intensity,
        ridge_point=ridge_point,
        peak_tflops=peak_tflops,
        roofline_peak_tflops=roofline_peak_tflops,
        bound=bound,
    )
