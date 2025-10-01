#!/usr/bin/env python3
"""Test script for FusionChain implementation with 2D tensor RMSNorm+Matmul fusion."""

import sys

import numpy as np

from fusion.fusion_chain import FusionChain
from fusion.metrics import check_correctness
from fusion.operators import AccumulationOperator, BlockingOperator
from fusion.tensors import Tensor


def test_2d_rmsnorm_matmul_fusion():
    """
    Test FusionChain with 2D tensors using custom fx, gB, hB functions.
    Demonstrates RMSNorm+Matmul fusion on batched inputs.
    """
    print("=" * 80)
    print("2D Tensor RMSNorm + Matmul Fusion Test")
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

    # Process each batch element independently
    results_fused = []
    results_standard = []

    print(f"\nProcessing {batch_size} batches...")

    for batch_idx in range(batch_size):
        # Extract batch element
        V1 = Tensor(V1_2d[batch_idx])
        V2 = Tensor(V2_2d[batch_idx])

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

        # Execute both algorithms
        result_fused = chain.execute_fused(V1, V2)
        result_standard = chain.execute_standard(V1, V2)

        results_fused.append(result_fused.data)
        results_standard.append(result_standard.data)

        # Show progress for first few batches
        if batch_idx < 3:
            print(f"  Batch {batch_idx}: Fused={result_fused.data:.6f}, Standard={result_standard.data:.6f}")

    # Stack results into 1D arrays
    results_fused = np.array(results_fused)
    results_standard = np.array(results_standard)

    print(f"\nOutput shape: {results_fused.shape}")

    # Verify correctness
    print("\n" + "-" * 40)
    print("Verifying correctness...")
    check_correctness(
        desired=results_standard,
        actual=results_fused,
        atol=1e-4,  # Adjusted for float32 precision
        rtol=1e-4,  # Adjusted for float32 precision
    )
    return True


def main():
    """Run the test demonstration."""
    print("\n" + "-" * 80)
    print(" " * 20 + "FUSIONCHAIN TEST")
    print("-" * 80)

    success = test_2d_rmsnorm_matmul_fusion()

    # Final summary
    print("\n" + "-" * 80)
    if success:
        print(" " * 25 + "TEST PASSED!")
        print("-" * 80)
    else:
        print(" " * 25 + "TEST FAILED")
        print("-" * 80)
        sys.exit(1)

    print("\n")


if __name__ == "__main__":
    main()
