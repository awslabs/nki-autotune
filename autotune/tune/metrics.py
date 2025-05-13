import json
import subprocess
from typing import Dict, Tuple

import numpy as np


def calculate_pe_utilization(mac_count: int, time_ms: float, target_instance_family: str) -> float:
    """
    Calculate PE utilization based on a given MAC.

    Parameters:
    mac_count (int): Number of multiply-accumulate operations
    time_ms (float): Execution time in milliseconds
    target_instance_family (str): Target hardware instance family (e.g., "trn1", "trn2")

    Returns:
    float: PE utilization percentage
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
    pe_utilization = theoretical_pe_cycles / actual_pe_cycles
    return pe_utilization


def calculate_GEMM_pe_utilization(
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

    # Call the direct MAC count function
    return calculate_pe_utilization(mac_count, time_ms, target_instance_family)


def extract_metrics(neff: str, ntff: str) -> Dict[str, float]:
    dump_json_cmd = f"neuron-profile view -n {neff} -s {ntff} --output-format summary-json"
    process = subprocess.run(dump_json_cmd, shell=True, capture_output=True, text=True, check=True)
    json_str = process.stdout

    data = json.loads(json_str)

    # Get the first (and only) key in the dictionary
    first_key = next(iter(data))
    metrics = data[first_key]

    # Extract key performance metrics
    important_metrics = {
        "hfu": metrics["hfu_estimated_percent"],
        "mbu": metrics["mbu_estimated_percent"],
        "Tensor Engine Utilization Time": metrics["tensor_engine_active_time_percent"],
        "Total Time (ms)": metrics["total_time"] * 1000,
        "Arithmetic Intensity": metrics["mm_arithmetic_intensity"],
    }
    return metrics


def allclose(a, b, rtol=1e-05, atol=1e-08):
    """
    Compare arrays with detailed information about differences.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Calculate absolute differences
    abs_diff = np.abs(a - b)

    # Calculate relative differences where b is not zero
    # (avoiding division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / np.abs(b)

    # Replace inf/nan from division by zero
    rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)

    # Find max differences
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)

    # Check if arrays are close according to numpy's definition
    is_close = np.all(abs_diff <= atol + rtol * np.abs(b))

    # Find locations of largest differences
    max_abs_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    max_rel_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)

    return {
        "is_close": is_close,
        "max_absolute_diff": max_abs_diff,
        "max_absolute_diff_location": max_abs_idx,
        "max_relative_diff": max_rel_diff,
        "max_relative_diff_location": max_rel_idx,
        "provided_rtol": rtol,
        "provided_atol": atol,
        "needed_rtol": max_rel_diff if max_rel_diff > rtol else rtol,
        "needed_atol": max_abs_diff if max_abs_diff > atol else atol,
    }
