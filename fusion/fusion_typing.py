"""Type definitions for the MegaFuse fusion framework."""

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

# Core type aliases for operator functions
StateType = Union[float, np.ndarray]
AccumulatorType = Union[float, np.ndarray]

# Blocking operator function: O¹_k = fx(O¹_{k-1}, V¹_k)
FxFunction = Callable[[StateType, float], StateType]

# Accumulation transform: Transforms blocking state to scaling factor
GbFunction = Callable[[StateType], float]

# Input preprocessing: Preprocesses inputs for accumulation
HbFunction = Callable[[float, float], float]


@dataclass
class FusionConfig:
    """Configuration for fusion execution."""

    epsilon: float = 1e-6  # Numerical stability parameter
    verify: bool = False  # Whether to verify against standard execution
    tolerance: float = 1e-4  # Tolerance for verification


@dataclass
class FusionState:
    """Maintains state during fusion execution."""

    blocking_state: StateType
    accumulator: AccumulatorType
    prev_gb_value: float
    step: int = 0
