import json
import os
import subprocess
from typing import Tuple

from autotune.cache.directories import split_file_info


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


def dump_profile_json(neff: str, ntff: str) -> str:
    directory, neff_name, file_type = split_file_info(neff)
    output_json_file = f"{directory}/{neff_name}.json"
    dump_json_cmd = f"neuron-profile view -n {neff} -s {ntff} --output-format json --output-file {output_json_file}"
    subprocess.run(dump_json_cmd, shell=True)
    return output_json_file


def parse_hfu(profile_json: str):
    with open(profile_json, "r") as f:
        data = json.load(f)
        hfu = data["summary"][0]["hfu_estimated_percent"]
    os.remove(profile_json)
    return hfu
