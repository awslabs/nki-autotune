#!/usr/bin/env python3
"""Test script for FusionChain implementation with 2D tensor RMSNorm+Matmul fusion."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fusion.fusion_chain import FusionChain
from fusion.metrics import check_correctness
from fusion.operators import AccumulationOperator, BlockingOperator
from fusion.tensors import Tensor


def test_2d_rmsnorm_matmul_fusion():
    """
    Test FusionChain with 2D tensors using custom fx, gB, hB functions.
    Demonstrates RMSNorm+Matmul fusion on batched inputs - processes all batches at once.
    """
    print("=" * 80)
    print("2D Tensor RMSNorm + Matmul Fusion Test (Batch Processing)")
    print("=" * 80)

    # Set parameters
    batch_size = 64
    hidden_dim = 1024
    epsilon = 1e-6

    print(f"\nParameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Epsilon: {epsilon}")

    # Create 2D input tensors
    np.random.seed(42)
    V1_2d = np.random.randn(batch_size, hidden_dim).astype(np.float32)
    V2_2d = np.random.randn(batch_size, hidden_dim).astype(np.float32)

    print(f"\nInput shapes:")
    print(f"  V1: {V1_2d.shape}")
    print(f"  V2: {V2_2d.shape}")

    # Define custom fx function (blocking operator)
    def fx(state, v1_k):
        """
        RMSNorm blocking operator: accumulate sum of squares.
        O¹_k = O¹_{k-1} + V¹_k²
        """
        return state + v1_k**2

    # Define custom gB function (transformation)
    def gB(blocking_state):
        """
        Transform blocking state to normalization factor.
        gB(O¹_k) = 1 / √(O¹_k/K + ε)
        """
        return 1.0 / np.sqrt(blocking_state / hidden_dim + epsilon)

    # Define custom hB function (accumulation preprocessing)
    def hB(v1_k, v2_k):
        """
        Preprocess inputs for accumulation (element-wise multiply).
        hB(V¹_k, V²_k) = V¹_k * V²_k
        """
        return v1_k * v2_k

    # Create operators with custom functions
    blocking_op = BlockingOperator(fx=fx, initial_state=0.0)  # O¹_0 = 0
    accumulation_op = AccumulationOperator(gB=gB, hB=hB, initial_accumulator=0.0)  # Õ²_0 = 0

    # Create FusionChain
    chain = FusionChain(blocking_operator=blocking_op, accumulation_operator=accumulation_op, K=hidden_dim)

    # Process all batches at once using 2D tensors
    V1 = Tensor(V1_2d)  # Shape: (batch_size, hidden_dim)
    V2 = Tensor(V2_2d)  # Shape: (batch_size, hidden_dim)

    print(f"\nProcessing all {batch_size} batches in parallel...")

    # Execute both algorithms on the entire batch
    result_fused = chain.execute_fused(V1, V2)
    result_standard = chain.execute_standard(V1, V2)

    # Results are now 1D arrays with one result per batch
    results_fused = result_fused.data
    results_standard = result_standard.data

    print(f"\nOutput shape: {results_fused.shape}")

    # Show some example outputs
    print(f"\nExample outputs (first 5 batches):")
    for i in range(min(5, batch_size)):
        print(f"  Batch {i}: Fused={results_fused[i]:.6f}, Standard={results_standard[i]:.6f}")

    # Verify correctness
    print("\n" + "-" * 40)
    print("Verifying correctness...")
    check_correctness(
        desired=results_standard,
        actual=results_fused,
        atol=1e-4,  # Adjusted for float32 precision
        rtol=1e-4,  # Adjusted for float32 precision
    )

    # Additional verification
    abs_diff = np.abs(results_fused - results_standard)
    print(f"\nError Statistics:")
    print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
    print(f"  Mean absolute difference: {np.mean(abs_diff):.2e}")
    print(f"  Std absolute difference: {np.std(abs_diff):.2e}")

    return True


if __name__ == "__main__":
    test_2d_rmsnorm_matmul_fusion()
