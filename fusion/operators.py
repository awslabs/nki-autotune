"""Operator definitions for fusion framework."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from fusion.fusion_typing import AccumulatorType, FxFunction, GbFunction, HbFunction, StateType


@dataclass
class BlockingOperator:
    """
    Defines the blocking operator X in "X + Accumulation" pattern.

    The blocking operator fx accumulates state that must be computed
    sequentially and creates a dependency barrier in standard execution.

    Example: For RMSNorm, fx accumulates sum of squares: O¹_k = O¹_{k-1} + V¹_k²
    """

    fx: FxFunction  # State update function: O¹_k = fx(O¹_{k-1}, V¹_k)
    initial_state: StateType  # Initial state O¹_0

    def update_state(self, current_state: StateType, input_value: np.ndarray) -> StateType:
        """Update the blocking state with new input."""
        return self.fx(current_state, input_value)


@dataclass
class AccumulationOperator:
    """
    Defines the accumulation operation in "X + Accumulation" pattern.

    The accumulation operation consists of:
    - gB: Transforms the blocking state (e.g., normalization factor)
    - hB: Preprocesses inputs for accumulation (e.g., element-wise multiply)

    Together they compute: B_k = gB(O¹_k) * hB(V¹_k, V²_k)

    Example: For RMSNorm+Matmul:
    - gB(state) = 1/√(state/K + ε)
    - hB(v1, v2) = v1 * v2
    """

    gB: GbFunction  # Transform blocking state to scaling factor
    hB: HbFunction  # Preprocess accumulation inputs
    initial_accumulator: AccumulatorType  # Initial accumulator Õ²_0

    def transform_state(self, blocking_state: StateType) -> float:
        """Transform blocking state using gB function."""
        return self.gB(blocking_state)

    def preprocess_inputs(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Preprocess inputs for accumulation using hB function."""
        return self.hB(v1, v2)

    def compute_accumulation_term(self, blocking_state: StateType, v1: np.ndarray, v2: np.ndarray) -> Any:
        """Compute the accumulation term B_k = gB(O¹_k) * hB(V¹_k, V²_k)."""
        gb_value = self.transform_state(blocking_state)
        hb_value = self.preprocess_inputs(v1, v2)
        return gb_value * hb_value


# Helper functions for creating common operators


def create_rmsnorm_blocking_operator(initial_state: float = 0.0) -> BlockingOperator:
    """Create blocking operator for RMSNorm (sum of squares)."""

    def fx(state: float, v1_k: np.ndarray) -> float:
        return state + float(v1_k**2)

    return BlockingOperator(fx=fx, initial_state=initial_state)


def create_rmsnorm_accumulation_operator(
    K: int, epsilon: float = 1e-6, initial_accumulator: float = 0.0
) -> AccumulationOperator:
    """Create accumulation operator for RMSNorm+Matmul."""

    def gB(blocking_state: float) -> float:
        # Normalization factor: 1/√(mean_square + ε)
        return 1.0 / (blocking_state / K + epsilon) ** 0.5

    def hB(v1_k: np.ndarray, v2_k: np.ndarray) -> np.ndarray:
        # Element-wise multiplication for matmul
        return v1_k * v2_k

    return AccumulationOperator(gB=gB, hB=hB, initial_accumulator=initial_accumulator)
