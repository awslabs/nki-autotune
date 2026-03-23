"""CPU simulator for rendered NKI kernel source code.

Executes NKI kernels using numpy at float64 precision for
correctness verification without requiring Neuron hardware.
"""

from nkigym.simulate.simulate import simulate_kernel

__all__ = ["simulate_kernel"]
