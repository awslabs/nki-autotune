"""Type definitions for the fusion framework."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Core type aliases
StateType = Any  # Blocking operator state O¹_k
AccumulatorType = Any  # Accumulator state Õ²_k
ScalarType = float  # Scaling factors and gB values

# Function signatures for the fusion pattern
FxFunction = Callable[[StateType, np.ndarray], StateType]
GbFunction = Callable[[StateType], ScalarType]
HbFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class FusionState:
    """Maintains state during fusion execution."""

    blocking_state: StateType  # O¹_k
    accumulator: AccumulatorType  # Õ²_k
    prev_gb_value: float  # g_B(O¹_{k-1}) for scaling computation


@dataclass
class FusionConfig:
    """Configuration for fusion execution."""

    epsilon: float = 1e-10  # Small value to avoid division by zero
