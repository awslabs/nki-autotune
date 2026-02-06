"""Lowering passes for NKI Gym.

Lowers the optimized nkigym IR to target-specific kernel code. The lowering
is nearly 1:1, translating each nkigym operation to its NKI equivalent.

Available lowering targets:
- gym_to_nki: Lowers to NKI (Neuron Kernel Interface) with explicit buffer management.

To add a new lowering target, create a new module in this package that accepts
a callable (nkigym function) and returns target-specific source code.
"""

from nkigym.lower.gym_to_nki import lower_gym_to_nki

__all__ = ["lower_gym_to_nki"]
