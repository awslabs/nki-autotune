# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import numpy as np


def calculate_mfu(mac_count: int, time_ms: float) -> float:
    """Calculate Model Flops Utilization based on a given MAC.

    Args:
        mac_count: Number of multiply-accumulate operations.
        time_ms: Execution time in milliseconds.

    Returns:
        PE utilization percentage.
    """
    pe_freq = 2.4e9

    flops = 2 * mac_count
    actual_latency_s = time_ms / 1000
    actual_pe_cycles = actual_latency_s * pe_freq
    theoretical_pe_cycles = flops / (2 * 128 * 128)
    mfu = theoretical_pe_cycles / actual_pe_cycles
    return mfu


def extract_metrics(neff: str, ntff: str, latency: float, matmul_mac_count: int) -> dict[str, float]:
    """Extract performance metrics from NEFF/NTFF files using neuron-profile.

    Args:
        neff: Path to the NEFF file.
        ntff: Path to the NTFF trace file.
        latency: Measured latency in milliseconds.
        matmul_mac_count: Number of MAC operations for MFU calculation.

    Returns:
        Dictionary of metric names to values, including estimated MFU.
    """
    dump_json_cmd = f"neuron-profile view -n {neff} -s {ntff} --output-format summary-json"
    process = subprocess.run(dump_json_cmd, shell=True, capture_output=True, text=True, check=True)
    json_str = process.stdout

    data = json.loads(json_str)

    first_key = next(iter(data))
    metrics = data[first_key]

    mfu = calculate_mfu(matmul_mac_count, latency)
    metrics["mfu_estimated_percent"] = mfu
    return metrics


def check_correctness(desired: np.ndarray, actual: np.ndarray, atol: float, rtol: float, verbose: bool = False) -> None:
    """Check numerical correctness between desired and actual arrays.

    Args:
        desired: Expected reference array.
        actual: Actual output array to verify.
        atol: Absolute tolerance threshold.
        rtol: Relative tolerance threshold.
        verbose: If True, print summary on success.

    Raises:
        AssertionError: If mismatched elements exceed tolerance.
    """
    assert desired.shape == actual.shape, f"Shape mismatch: desired {desired.shape}, actual {actual.shape}"
    abs_diff = np.abs(actual - desired)
    rel_diff = np.divide(abs_diff, np.abs(desired), out=np.zeros_like(abs_diff), where=np.abs(desired) != 0)

    tolerance = atol + rtol * np.abs(desired)
    mismatches = abs_diff > tolerance
    total_mismatches = np.sum(mismatches)
    total_elements = desired.size

    mismatch_percentage = (total_mismatches / total_elements) * 100
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)

    if total_mismatches > 0:
        regions_summary = generate_mismatch_summary(mismatches, desired, actual)
        err_msg = (
            f"Correctness check FAILED\n"
            f"  Tolerance used: atol={atol}, rtol={rtol}\n"
            f"  Mismatched elements: {total_mismatches} / {total_elements} ({mismatch_percentage:.6f}%)\n"
            f"  Max absolute difference: {max_abs_diff}\n"
            f"  Max relative difference: {max_rel_diff}\n"
            f"  {regions_summary}"
        )
        raise AssertionError(err_msg)
    if verbose:
        print(
            f"Correctness check PASSED\n"
            f"  Tolerance used: atol={atol}, rtol={rtol}\n"
            f"  Total elements: {total_elements}\n"
            f"  Max absolute difference: {max_abs_diff}\n"
            f"  Max relative difference: {max_rel_diff}"
        )


def generate_mismatch_summary(mismatches: np.ndarray, desired: np.ndarray, actual: np.ndarray) -> str:
    """Generate a summary of contiguous regions with mismatches.

    Args:
        mismatches: Boolean array indicating mismatched positions.
        desired: Expected reference array.
        actual: Actual output array.

    Returns:
        Formatted string summarizing mismatched regions.
    """
    if len(mismatches.shape) == 2:
        return summarize_2d_mismatches(mismatches, desired, actual)
    else:
        return summarize_nd_mismatches(mismatches, desired, actual)


def summarize_2d_mismatches(mismatches: np.ndarray, desired: np.ndarray, actual: np.ndarray) -> str:
    """Summarize mismatches in 2D arrays as contiguous regions, sorted by size.

    Args:
        mismatches: 2D boolean array indicating mismatched positions.
        desired: Expected reference array.
        actual: Actual output array.

    Returns:
        Formatted string describing mismatched regions with sample values.
    """
    total_mismatches = np.sum(mismatches)

    if total_mismatches == 0:
        return "No mismatches found."

    if total_mismatches == 1:
        row, col = np.where(mismatches)
        r, c = row[0], col[0]
        return f"Only element [{r}, {c}] is wrong.\n  Desired: {desired[r, c]}\n  Actual:  {actual[r, c]}"

    region_info = []
    rows, cols = mismatches.shape

    visited = np.zeros_like(mismatches, dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if mismatches[r, c] and not visited[r, c]:
                max_r, max_c = r, c

                while max_r + 1 < rows and mismatches[max_r + 1, c]:
                    max_r += 1

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

                visited[r : max_r + 1, c : c + width] = True

                region_size = (max_r - r + 1) * width

                region_info.append((region_size, r, c, max_r, c + width - 1))

    region_info.sort(key=lambda x: (-x[0], x[1], x[2]))

    topK_with_vals = 3
    topK = 5
    region_strings = []
    num_regions_with_values = min(topK_with_vals, len(region_info))

    for i, (size, r_start, c_start, r_end, c_end) in enumerate(region_info):
        if i >= topK and len(region_info) > topK:
            remaining = len(region_info) - topK
            region_strings.append(f"... {remaining} more regions not shown")
            break

        if r_start == r_end and c_start == c_end:
            region_str = f"\nRegion {i+1}: [{r_start}, {c_start}] (size: {size})"
            if i < num_regions_with_values:
                region_str += f"\n  Desired: {desired[r_start, c_start]}"
                region_str += f"\n  Actual:  {actual[r_start, c_start]}"
        else:
            region_str = f"\nRegion {i+1}: [{r_start}:{r_end+1}, {c_start}:{c_end+1}] (size: {size})"
            if i < num_regions_with_values:
                desired_region = desired[r_start : r_end + 1, c_start : c_end + 1]
                actual_region = actual[r_start : r_end + 1, c_start : c_end + 1]

                height = r_end - r_start + 1
                width = c_end - c_start + 1

                if height <= 3 and width <= 4:
                    region_str += "\n  Desired:\n" + _format_array(desired_region, "    ")
                    region_str += "\n  Actual:\n" + _format_array(actual_region, "    ")
                else:
                    region_str += f" (showing sample of {height}x{width} region)"
                    sample_h = min(3, height)
                    sample_w = min(4, width)

                    desired_sample = desired_region[:sample_h, :sample_w]
                    actual_sample = actual_region[:sample_h, :sample_w]

                    region_str += "\n  Desired (top-left corner):\n" + _format_array(
                        desired_sample, "    ", show_ellipsis=(height > 3 or width > 4)
                    )
                    region_str += "\n  Actual (top-left corner):\n" + _format_array(
                        actual_sample, "    ", show_ellipsis=(height > 3 or width > 4)
                    )

        region_strings.append(region_str)

    total_regions = len(region_info)
    header = f"Found {total_regions} mismatched regions, sorted by size (largest first):"
    return f"{header}\n" + "".join(region_strings)


def _format_array(arr: np.ndarray, indent: str = "", show_ellipsis: bool = False) -> str:
    """Format a numpy array for display with proper indentation.

    Args:
        arr: Array to format.
        indent: String prefix for each line.
        show_ellipsis: If True, append ellipsis to indicate truncation.

    Returns:
        Formatted string representation of the array.
    """
    with np.printoptions(precision=6, suppress=True, threshold=100):
        lines = str(arr).split("\n")
        formatted = [indent + line for line in lines]
        result = "\n".join(formatted)
        if show_ellipsis:
            result = result.replace("]", ", ...]")
        return result


def summarize_nd_mismatches(mismatches: np.ndarray, desired: np.ndarray, actual: np.ndarray) -> str:
    """Summarize mismatches in arrays with dimensions other than 2.

    Args:
        mismatches: Boolean array indicating mismatched positions.
        desired: Expected reference array.
        actual: Actual output array.

    Returns:
        Formatted string with mismatch count and example values.
    """
    total_mismatches = np.sum(mismatches)
    if total_mismatches == 1:
        coords = np.where(mismatches)
        coord_str = ", ".join(str(dim[0]) for dim in coords)
        idx = tuple(dim[0] for dim in coords)
        return f"Only element [{coord_str}] is wrong.\n  Desired: {desired[idx]}\n  Actual:  {actual[idx]}"

    coords = np.where(mismatches)
    examples = []
    for i in range(min(5, total_mismatches)):
        coord_list = [str(dim[i]) for dim in coords]
        idx = tuple(dim[i] for dim in coords)
        example = f"\n  [{', '.join(coord_list)}]: Desired={desired[idx]}, Actual={actual[idx]}"
        examples.append(example)

    example_str = "".join(examples)
    if total_mismatches > 5:
        example_str += f"\n  ... {total_mismatches - 5} more mismatches not shown"

    return f"Found {total_mismatches} mismatches. Examples:{example_str}"
