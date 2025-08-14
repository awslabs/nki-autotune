import json
import subprocess
from typing import Dict, List, Tuple

import numpy as np
from neuronxcc.starfish.penguin.ir.ir import BIRKernel, NativeKernelTemplate, TensorContractTensorOp
from neuronxcc.starfish.penguin.targets.tonga.TongaISAInst import MatMulOp, MatMulSparseOp


def get_matmul_mac_count(traced_kernel):
    flops = 0
    for inst in traced_kernel._code.insts:
        if isinstance(inst, (MatMulOp, MatMulSparseOp, NativeKernelTemplate, BIRKernel)):
            flops += inst.total_arithmetic_ops
        elif isinstance(inst, (TensorContractTensorOp)):
            lhs_shape = inst.lhs_shape
            rhs_shape = inst.rhs_shape
            if len(lhs_shape) == 2:
                batch_factor = 1
                M, K = lhs_shape
            elif len(lhs_shape) == 3:
                batch_factor, M, K = lhs_shape
            else:
                raise ValueError(f"Unrecognized lhs_shape {lhs_shape}. Expecting (batch (optional), M, K)")
            if len(rhs_shape) == 2:
                _batch_factor = 1
                _K, N = rhs_shape
            elif len(rhs_shape) == 3:
                _batch_factor, _K, N = rhs_shape
            else:
                raise ValueError(f"Unrecognized rhs_shape {rhs_shape}. Expecting (batch (optional), K, N)")
            assert (
                K == _K and batch_factor == _batch_factor
            ), f"Incompatible matrix shapes. Received LHS {lhs_shape}. RHS {rhs_shape}."
            flops += 2 * batch_factor * M * N * K
    # One MAC is 2 flops
    mac_count = flops // 2
    return mac_count


def calculate_mfu(mac_count: int, time_ms: float, target_instance_family: str) -> float:
    """
    Calculate Model Flops Utilization based on a given MAC.

    Parameters:
    mac_count (int): Number of multiply-accumulate operations
    time_ms (float): Execution time in milliseconds
    target_instance_family (str): Target hardware instance family (e.g., "trn1", "trn2")

    Returns:
    float: PE utilization percentage
    FIXME: check dtype
    """
    if any(target_instance_family.startswith(family) for family in {"sunda", "trainium", "trn1", "inf2"}):
        pe_freq = 2.8 * 1e9  # Hz (2.8 GHz)
        num_lnc = 1
    elif any(target_instance_family.startswith(family) for family in {"trn2", "inf3", "gen3", "cayman"}):
        pe_freq = 2.4 * 1e9  # Hz (2.4 GHz)
        num_lnc = 2
    else:
        raise NotImplementedError("Unknown target instance: " + target_instance_family)

    # Calculate total FLOPS (2 operations per MAC - multiply and add)
    flops = 2 * mac_count
    actual_latency_s = time_ms / 1000
    actual_pe_cycles = actual_latency_s * pe_freq
    theoretical_pe_cycles = flops / (2 * 128 * 128 * num_lnc)
    mfu = theoretical_pe_cycles / actual_pe_cycles
    return mfu


def calculate_mfu_from_shapes(
    lhsT_shape: Tuple[int, ...], rhs_shape: Tuple[int, ...], time_ms: float, target_instance_family: str
) -> float:
    """
    Calculate model PE utilization for a GEMM operation using matrix shapes.

    Parameters:
    lhsT_shape (tuple): Shape of matrix A (k, m)
    rhs_shape (tuple): Shape of matrix B (k, n)
    time_ms (float): Execution time in milliseconds
    target_instance_family (str): Target hardware instance family

    Returns:
    float: PE utilization percentage
    """
    k, m = lhsT_shape
    _k, n = rhs_shape
    assert k == _k, f"Incompatible matrix dimensions: lhsT {lhsT_shape} and rhs {rhs_shape}"

    # Calculate MAC count
    mac_count = m * k * n
    mfu = calculate_mfu(mac_count, time_ms, target_instance_family)
    return mfu


def extract_metrics(
    neff: str, ntff: str, latency: float, matmul_mac_count: int, target_instance_family: str
) -> Dict[str, float]:
    dump_json_cmd = f"neuron-profile view -n {neff} -s {ntff} --output-format summary-json"
    process = subprocess.run(dump_json_cmd, shell=True, capture_output=True, text=True, check=True)
    json_str = process.stdout

    data = json.loads(json_str)

    # Get the first (and only) key in the dictionary
    first_key = next(iter(data))
    metrics = data[first_key]

    mfu = calculate_mfu(matmul_mac_count, latency, target_instance_family)
    metrics["mfu_estimated_percent"] = mfu
    return metrics


def tensor_to_matmul_mac_count(tensors: List[np.ndarray]) -> int:
    if len(tensors) == 2 and len(tensors[0].shape) == 2 and len(tensors[1].shape) == 2:
        mac_count = tensors[0].shape[0] * tensors[0].shape[1] * tensors[1].shape[1]
    else:
        mac_count = 0
    return mac_count


def check_correctness(desired, actual, atol, rtol):
    abs_diff = np.abs(actual - desired)
    # Avoid division by zero in relative difference calculation
    rel_diff = np.divide(abs_diff, np.abs(desired), out=np.zeros_like(abs_diff), where=np.abs(desired) != 0)

    # Calculate tolerance threshold using numpy's allclose formula
    tolerance = atol + rtol * np.abs(desired)
    mismatches = abs_diff > tolerance
    total_mismatches = np.sum(mismatches)
    total_elements = desired.size

    if total_mismatches > 0:
        # Calculate statistics
        mismatch_percentage = (total_mismatches / total_elements) * 100
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)

        # Generate error message with statistics and mismatch regions
        regions_summary = generate_mismatch_summary(mismatches)

        err_msg = (
            f"Mismatched elements: {total_mismatches} / {total_elements} ({mismatch_percentage:.6f}%)\n"
            f"Max absolute difference: {max_abs_diff}\n"
            f"Max relative difference: {max_rel_diff}\n"
            f"{regions_summary}"
        )

        raise AssertionError(err_msg)


def generate_mismatch_summary(mismatches):
    """Generate a summary of contiguous regions with mismatches."""
    if len(mismatches.shape) == 2:  # For 2D arrays
        return summarize_2d_mismatches(mismatches)
    else:
        # For other dimensions
        return summarize_nd_mismatches(mismatches)


def summarize_2d_mismatches(mismatches):
    """Summarize mismatches in 2D arrays as contiguous regions, sorted by size."""
    total_mismatches = np.sum(mismatches)

    if total_mismatches == 0:
        return "No mismatches found."

    if total_mismatches == 1:
        row, col = np.where(mismatches)
        return f"Only element [{row[0]}, {col[0]}] is wrong."

    # Find contiguous regions
    region_info = []  # Will store (size, r_start, c_start, r_end, c_end) tuples
    rows, cols = mismatches.shape

    # Process the array to find rectangular regions
    visited = np.zeros_like(mismatches, dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if mismatches[r, c] and not visited[r, c]:
                # Find the largest rectangle starting at (r,c)
                max_r, max_c = r, c

                # Extend rows
                while max_r + 1 < rows and mismatches[max_r + 1, c]:
                    max_r += 1

                # Find the maximum width for this range of rows
                width = 1
                while c + width < cols:
                    can_extend = True
                    for row_idx in range(r, max_r + 1):
                        if not mismatches[row_idx, c + width]:
                            can_extend = False
                            break
                    if can_extend:
                        width += 1
                    else:
                        break

                # Mark this region as visited
                visited[r : max_r + 1, c : c + width] = True

                # Calculate region size
                region_size = (max_r - r + 1) * width

                # Add region info: (size, r_start, c_start, r_end, c_end)
                region_info.append((region_size, r, c, max_r, c + width - 1))

    # Sort regions by size (descending) and then by coordinates (ascending)
    # For ties in size, sort by row_start, then col_start
    region_info.sort(key=lambda x: (-x[0], x[1], x[2]))

    # Format region strings
    region_strings = []
    for i, (size, r_start, c_start, r_end, c_end) in enumerate(region_info):
        # Only display top 10 regions if there are more than 10
        if i >= 10 and len(region_info) > 10:
            remaining = len(region_info) - 10
            region_strings.append(f"... {remaining} more regions not shown")
            break

        if r_start == r_end and c_start == c_end:
            region_strings.append(f"[{r_start}, {c_start}] (size: {size})")
        else:
            region_strings.append(f"[{r_start}:{r_end+1}, {c_start}:{c_end+1}] (size: {size})")

    if len(region_strings) == 1:
        return f"Elements {region_strings[0]} are wrong."
    else:
        total_regions = len(region_info)
        header = f"Found {total_regions} mismatched regions, sorted by size (largest first):"
        return f"{header}\n" + "\n".join(region_strings)


def summarize_nd_mismatches(mismatches):
    """Handle mismatches in arrays with dimensions other than 2."""
    total_mismatches = np.sum(mismatches)
    if total_mismatches == 1:
        coords = np.where(mismatches)
        coord_str = ", ".join(str(dim[0]) for dim in coords)
        return f"Only element [{coord_str}] is wrong."

    # For higher dimensions, just report the total and some examples
    coords = np.where(mismatches)
    # Get up to 5 examples
    examples = []
    for i in range(min(5, total_mismatches)):
        example = "[" + ", ".join(str(dim[i]) for dim in coords) + "]"
        examples.append(example)

    example_str = ", ".join(examples)
    if total_mismatches > 5:
        example_str += ", ..."

    return f"Found {total_mismatches} mismatches. Examples: {example_str}"
