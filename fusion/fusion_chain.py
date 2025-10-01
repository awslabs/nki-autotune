"""FusionChain implementation for universal kernel fusion based on the MegaFuse paper."""

from typing import Optional, Union

import numpy as np

from fusion.fusion_typing import FusionConfig
from fusion.operators import AccumulationOperator, BlockingOperator, create_rmsnorm_matmul
from fusion.tensors import Tensor


class FusionChain:
    """
    Implements Algorithm 3 from MegaFuse: Online fusion for X + Accumulation patterns.

    This class automatically handles the fusion mechanics for any combination of
    blocking operator (fx) and accumulation operations (gB, hB), computing
    scaling factors and managing recurrence relations to enable concurrent
    execution while maintaining mathematical equivalence.

    The key insight from MegaFuse is using scaling factors s_k = gB(O¹_k) / gB(O¹_{k-1})
    to correct for partial results and enable online computation without waiting for
    the blocking operator to complete.

    Example:
        # Using pre-built RMSNorm+Matmul pattern
        chain = RMSNormMatmul(K=128, epsilon=1e-6)
        result = chain(input_tensor, weight_tensor)

        # Using custom operators
        blocking = BlockingOperator(fx=lambda s, v: s + v**2, initial_state=0.0)
        accumulation = AccumulationOperator(
            gB=lambda s: 1/np.sqrt(s/K + epsilon),
            hB=lambda v1, v2: v1 * v2,
            initial_accumulator=0.0
        )
        chain = FusionChain(blocking, accumulation, K=128)
        result = chain.execute_fused(V1, V2)
    """

    def __init__(
        self, blocking_operator: BlockingOperator, accumulation_operator: AccumulationOperator, K: Optional[int] = None
    ):
        """
        Initialize the FusionChain.

        Args:
            blocking_operator: Defines fx and initial state O¹_0
            accumulation_operator: Defines gB, hB, and initial accumulator Õ²_0
            K: Total number of fusion steps (can be inferred from input if None)
        """
        self.blocking_op = blocking_operator
        self.accumulation_op = accumulation_operator
        self.K = K

    def __call__(
        self, V1: Union[Tensor, np.ndarray], V2: Union[Tensor, np.ndarray], config: Optional[FusionConfig] = None
    ) -> Tensor:
        """Convenience method for execute_fused."""
        if isinstance(V1, np.ndarray):
            V1 = Tensor(V1)
        if isinstance(V2, np.ndarray):
            V2 = Tensor(V2)
        return self.execute_fused(V1, V2, config)

    def execute_fused(self, V1: Tensor, V2: Tensor, config: Optional[FusionConfig] = None) -> Tensor:
        """
        Execute Algorithm 3: Online Fusion for batched inputs.

        Implements the MegaFuse online fusion algorithm which breaks the sequential
        dependency by creating an alternative output series Õ² that depends on
        partial results of the blocking operator rather than waiting for completion.

        The algorithm maintains the recurrence relation:
            Õ²_k = s_k * Õ²_{k-1} + B_k
        where:
            s_k = gB(O¹_k) / gB(O¹_{k-1}) is the scaling factor
            B_k = gB(O¹_k) * hB(V¹_k, V²_k) is the accumulation term

        Args:
            V1: First input tensor with shape (batch_size, K)
            V2: Second input tensor with shape (batch_size, K)
            config: Optional configuration settings

        Returns:
            Tensor containing Õ²_K with shape (batch_size,)
            Note: Õ²_K = O²_K (equivalent to standard execution result)
        """
        if config is None:
            config = FusionConfig()

        # Validate inputs
        if V1.ndim != 2 or V2.ndim != 2:
            raise ValueError(f"Expected 2D tensors, got shapes {V1.shape} and {V2.shape}")
        if V1.shape != V2.shape:
            raise ValueError(f"Input shapes must match, got {V1.shape} and {V2.shape}")

        batch_size, K = V1.shape
        if self.K is not None and K != self.K:
            raise ValueError(f"K dimension {K} doesn't match expected {self.K}")

        # Initialize states for all batches
        if not np.isscalar(self.blocking_op.initial_state):
            raise NotImplementedError("Non-scalar initial states not yet supported")

        # O¹_0 for all batches
        blocking_states = np.full(batch_size, self.blocking_op.initial_state, dtype=np.float32)
        # Õ²_0 for all batches
        accumulators = np.full(batch_size, self.accumulation_op.initial_accumulator, dtype=np.float32)

        # Compute initial gB values for scaling
        initial_state_val = (
            float(self.blocking_op.initial_state)
            if np.isscalar(self.blocking_op.initial_state)
            else self.blocking_op.initial_state
        )
        prev_gb_values = np.array(
            [self.accumulation_op.transform_state(initial_state_val) for _ in range(batch_size)], dtype=np.float32
        )

        # Main fusion loop - Algorithm 3
        for k in range(K):
            # Get k-th column for all batches
            v1_k = V1.data[:, k]  # Shape: (batch_size,)
            v2_k = V2.data[:, k]  # Shape: (batch_size,)

            # Step 1: Update blocking state O¹_k = fx(O¹_{k-1}, V¹_k)
            for i in range(batch_size):
                blocking_states[i] = self.blocking_op.update_state(blocking_states[i], v1_k[i])

            # Step 2: Compute current gB values
            curr_gb_values = np.array(
                [self.accumulation_op.transform_state(blocking_states[i]) for i in range(batch_size)], dtype=np.float32
            )

            # Step 3: Compute scaling factors s_k = gB(O¹_k) / gB(O¹_{k-1})
            # Handle division by zero with epsilon
            s_k = np.where(
                np.abs(prev_gb_values) > config.epsilon,
                curr_gb_values / prev_gb_values,
                np.ones(batch_size, dtype=np.float32),
            )

            # Step 4: Compute accumulation terms B_k = gB(O¹_k) * hB(V¹_k, V²_k)
            B_k = np.array(
                [
                    self.accumulation_op.compute_accumulation_term(blocking_states[i], v1_k[i], v2_k[i])
                    for i in range(batch_size)
                ],
                dtype=np.float32,
            )

            # Step 5: Update accumulators Õ²_k = s_k * Õ²_{k-1} + B_k
            accumulators = s_k * accumulators + B_k

            # Update for next iteration
            prev_gb_values = curr_gb_values.copy()

        # Verify if requested
        if config.verify:
            result_standard = self.execute_standard(V1, V2)
            if not np.allclose(accumulators, result_standard.data, rtol=config.tolerance):
                raise ValueError(
                    f"Fusion verification failed. Max diff: {np.max(np.abs(accumulators - result_standard.data))}"
                )

        return Tensor(accumulators)

    def execute_standard(self, V1: Tensor, V2: Tensor) -> Tensor:
        """
        Execute Algorithm 2: Standard (unfused) execution for verification.

        This implements the traditional two-phase execution where the blocking
        operator must complete entirely before accumulation can begin:

        Phase 1: Complete all blocking operations to get O¹_K
        Phase 2: Perform accumulation with final gB(O¹_K)

        This method is primarily used for verification that the fused algorithm
        produces mathematically equivalent results.

        Args:
            V1: First input tensor with shape (batch_size, K)
            V2: Second input tensor with shape (batch_size, K)

        Returns:
            Tensor containing O²_K with shape (batch_size,)
        """
        if V1.ndim != 2 or V2.ndim != 2:
            raise ValueError(f"Expected 2D tensors, got shapes {V1.shape} and {V2.shape}")

        batch_size, K = V1.shape
        if self.K is not None and K != self.K:
            raise ValueError(f"K dimension {K} doesn't match expected {self.K}")

        # Phase 1: Complete blocking operations for all batches
        if not np.isscalar(self.blocking_op.initial_state):
            raise NotImplementedError("Non-scalar initial states not yet supported")

        blocking_states = np.full(batch_size, self.blocking_op.initial_state, dtype=np.float32)

        # Compute O¹_K for all batches
        for k in range(K):
            v1_k = V1.data[:, k]
            for i in range(batch_size):
                blocking_states[i] = self.blocking_op.update_state(blocking_states[i], v1_k[i])

        # Compute final gB(O¹_K) values
        final_gb_values = np.array(
            [self.accumulation_op.transform_state(blocking_states[i]) for i in range(batch_size)], dtype=np.float32
        )

        # Phase 2: Accumulation with final blocking states
        accumulators = np.full(batch_size, self.accumulation_op.initial_accumulator, dtype=np.float32)

        for k in range(K):
            v1_k = V1.data[:, k]
            v2_k = V2.data[:, k]
            for i in range(batch_size):
                # B_k = gB(O¹_K) * hB(V¹_k, V²_k)
                B_k = final_gb_values[i] * self.accumulation_op.preprocess_inputs(v1_k[i], v2_k[i])
                accumulators[i] = accumulators[i] + B_k

        return Tensor(accumulators)


class RMSNormMatmul(FusionChain):
    """
    Pre-built fusion chain for RMSNorm + Matmul pattern.

    This pattern appears frequently in transformer models where layer normalization
    is followed by a linear projection. The MegaFuse algorithm enables these
    operations to execute concurrently rather than sequentially.

    Example:
        chain = RMSNormMatmul(K=128, epsilon=1e-6)
        normalized_output = chain(input_tensor, weight_tensor)
    """

    def __init__(self, K: Optional[int] = None, epsilon: float = 1e-6):
        """
        Initialize RMSNorm+Matmul fusion chain.

        Args:
            K: Dimension size (can be inferred from input)
            epsilon: Numerical stability parameter for RMSNorm
        """
        # Use dummy K if not provided, will be inferred from input
        k_val = K if K is not None else 1
        blocking_op, accumulation_op = create_rmsnorm_matmul(k_val, epsilon)
        super().__init__(blocking_op, accumulation_op, K)
        self.epsilon = epsilon
        self._k_val = k_val

    def execute_fused(self, V1: Tensor, V2: Tensor, config: Optional[FusionConfig] = None) -> Tensor:
        """Execute with automatic K inference if needed."""
        if self.K is None:
            # Infer K from input and recreate operators
            K = V1.shape[1]
            self.blocking_op, self.accumulation_op = create_rmsnorm_matmul(K, self.epsilon)
            self.K = K
        return super().execute_fused(V1, V2, config)
