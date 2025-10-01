"""Operator definitions for the MegaFuse fusion framework."""

from dataclasses import dataclass

import numpy as np

from fusion.fusion_typing import AccumulatorType, FxFunction, GbFunction, HbFunction, StateType


@dataclass
class BlockingOperator:
    """
    Defines a blocking operator with sequential dependencies.

    The blocking operator maintains state O¹_k that depends on the
    previous state O¹_{k-1} and current input V¹_k.

    Attributes:
        fx: Update function O¹_k = fx(O¹_{k-1}, V¹_k)
        initial_state: Initial value O¹_0
    """

    fx: FxFunction
    initial_state: StateType

    def update_state(self, state: StateType, v1_k: float) -> StateType:
        """Update blocking state with new input."""
        return self.fx(state, v1_k)


@dataclass
class AccumulationOperator:
    """
    Defines accumulation operations that can be fused with blocking operators.

    The accumulation operator computes B_k = gB(O¹_k) * hB(V¹_k, V²_k)
    where gB transforms the blocking state and hB preprocesses inputs.

    Attributes:
        gB: Transform function for blocking state (e.g., normalization factor)
        hB: Preprocessing function for accumulation inputs
        initial_accumulator: Initial value Õ²_0
    """

    gB: GbFunction
    hB: HbFunction
    initial_accumulator: AccumulatorType

    def transform_state(self, blocking_state: StateType) -> float:
        """Transform blocking state to scaling factor."""
        return self.gB(blocking_state)

    def preprocess_inputs(self, v1_k: float, v2_k: float) -> float:
        """Preprocess inputs for accumulation."""
        return self.hB(v1_k, v2_k)

    def compute_accumulation_term(self, blocking_state: StateType, v1_k: float, v2_k: float) -> float:
        """Compute B_k = gB(O¹_k) * hB(V¹_k, V²_k)."""
        return self.transform_state(blocking_state) * self.preprocess_inputs(v1_k, v2_k)


# Pre-built operator patterns
class RMSNormMatmulOperators:
    """Pre-built operators for RMSNorm + Matmul fusion."""

    @staticmethod
    def create_blocking_operator(K: int) -> BlockingOperator:
        """Create blocking operator for sum of squares."""

        def fx(state: StateType, v1_k: float) -> StateType:
            return state + v1_k * v1_k

        return BlockingOperator(fx=fx, initial_state=0.0)

    @staticmethod
    def create_accumulation_operator(K: int, epsilon: float = 1e-6) -> AccumulationOperator:
        """Create accumulation operator for normalized matmul."""

        def gB(blocking_state: StateType) -> float:
            # RMSNorm factor: 1/sqrt(sum_squares/K + epsilon)
            return float(1.0 / np.sqrt(blocking_state / K + epsilon))

        def hB(v1_k: float, v2_k: float) -> float:
            # Element-wise multiplication
            return v1_k * v2_k

        return AccumulationOperator(gB=gB, hB=hB, initial_accumulator=0.0)


def create_rmsnorm_matmul(K: int, epsilon: float = 1e-6) -> tuple[BlockingOperator, AccumulationOperator]:
    """
    Create operators for RMSNorm + Matmul fusion.

    Args:
        K: Dimension size
        epsilon: Numerical stability parameter

    Returns:
        Tuple of (blocking_operator, accumulation_operator)
    """
    return (
        RMSNormMatmulOperators.create_blocking_operator(K),
        RMSNormMatmulOperators.create_accumulation_operator(K, epsilon),
    )
