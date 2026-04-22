"""Hardware detection helpers for the remote worker."""

import json
import subprocess


def detect_neuron_cores() -> int:
    """Detect available Neuron cores via neuron-ls.

    Raises:
        RuntimeError: If neuron-ls fails or reports 0 cores.
    """
    result = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"neuron-ls failed: {result.stderr}")
    devices = json.loads(result.stdout)
    total_cores = sum(d["nc_count"] for d in devices)
    if total_cores == 0:
        raise RuntimeError("neuron-ls reports 0 Neuron cores")
    return total_cores
