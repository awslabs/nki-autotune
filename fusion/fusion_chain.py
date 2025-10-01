"""FusionChain implementation for universal kernel fusion."""

from typing import Optional

from fusion.fusion_typing import FusionConfig, FusionState
from fusion.operators import AccumulationOperator, BlockingOperator
from fusion.tensors import Tensor


class FusionChain:
    """
    Implements Algorithm 3: Online fusion for X + Accumulation patterns.

    This class automatically handles the fusion mechanics for any combination of
    blocking operator (fx) and accumulation operations (gB, hB), computing
    scaling factors and managing recurrence relations to enable concurrent
    execution while maintaining mathematical equivalence.

    The key insight is using scaling factors s_k = gB(O¹_{k-1}) / gB(O¹_k) to
    correct for partial results and enable online computation.
    """

    def __init__(
        self, blocking_operator: BlockingOperator, accumulation_operator: AccumulationOperator, K: Optional[int] = None
    ):
        """
        Initialize the FusionChain.

        Args:
            blocking_operator: Defines fx and initial state
            accumulation_operator: Defines gB, hB, and initial accumulator
            K: Total number of fusion steps (can be inferred from input if None)
        """
        self.blocking_op = blocking_operator
        self.accumulation_op = accumulation_operator
        self.K = K

    def execute_fused(self, V1: Tensor, V2: Tensor, config: Optional[FusionConfig] = None) -> Tensor:
        """
        Execute Algorithm 3: Online Fusion.

        Implements the core fusion loop:
        for k = 1 to K:
            O¹_k = fx(O¹_{k-1}, V¹_k)
            s_k = gB(O¹_{k-1}) / gB(O¹_k)
            B_k = gB(O¹_k) * hB(V¹_k, V²_k)
            Õ²_k = s_k * Õ²_{k-1} + B_k

        Args:
            V1: First input tensor (for blocking operator and accumulation)
            V2: Second input tensor (for accumulation)
            config: Optional configuration settings

        Returns:
            Tensor containing Õ²_K (final fused result)
        """
        if config is None:
            config = FusionConfig()

        # Determine K from input if not specified
        K = self.K if self.K is not None else len(V1)

        # Initialize state
        state = FusionState(
            blocking_state=self.blocking_op.initial_state,
            accumulator=self.accumulation_op.initial_accumulator,
            prev_gb_value=self.accumulation_op.gB(self.blocking_op.initial_state),
        )

        # Main fusion loop (Algorithm 3)
        for k in range(K):
            # Step 1: Update blocking state
            # O¹_k = fx(O¹_{k-1}, V¹_k)
            state.blocking_state = self.blocking_op.update_state(state.blocking_state, V1[k])

            # Step 2: Compute current gB value
            # gB(O¹_k)
            curr_gb = self.accumulation_op.transform_state(state.blocking_state)

            # Step 3: Compute scaling factor
            # s_k = gB(O¹_k) / gB(O¹_{k-1})
            # This is the key to online fusion - scaling previous accumulation
            if k == 0 or abs(state.prev_gb_value) < config.epsilon:
                s_k = 1.0
            else:
                s_k = curr_gb / state.prev_gb_value

            # Step 4: Compute accumulation term
            # B_k = gB(O¹_k) * hB(V¹_k, V²_k)
            B_k = self.accumulation_op.compute_accumulation_term(state.blocking_state, V1[k], V2[k])

            # Step 5: Update accumulator with recurrence relation
            # Õ²_k = s_k * Õ²_{k-1} + B_k
            state.accumulator = s_k * state.accumulator + B_k

            # Update for next iteration
            state.prev_gb_value = curr_gb

        return Tensor(state.accumulator)

    def execute_standard(self, V1: Tensor, V2: Tensor) -> Tensor:
        """
        Execute Algorithm 2: Standard (unfused) execution for verification.

        This implements the traditional two-phase execution:
        Phase 1: Complete all blocking operations
        Phase 2: Perform accumulation with final blocking state

        Args:
            V1: First input tensor
            V2: Second input tensor

        Returns:
            Tensor containing O²_K (standard result)
        """
        # Determine K from input if not specified
        K = self.K if self.K is not None else len(V1)

        # Phase 1: Complete blocking operation
        # for k = 1 to K: O¹_k = fx(O¹_{k-1}, V¹_k)
        blocking_state = self.blocking_op.initial_state
        for k in range(K):
            blocking_state = self.blocking_op.update_state(blocking_state, V1[k])

        # Compute final gB value
        # gB(O¹_K)
        final_gb = self.accumulation_op.transform_state(blocking_state)

        # Phase 2: Accumulation with final blocking state
        # for k = 1 to K: O²_k = O²_{k-1} + gB(O¹_K) * hB(V¹_k, V²_k)
        accumulator = self.accumulation_op.initial_accumulator
        for k in range(K):
            # B_k = gB(O¹_K) * hB(V¹_k, V²_k)
            B_k = final_gb * self.accumulation_op.preprocess_inputs(V1[k], V2[k])
            accumulator = accumulator + B_k

        return Tensor(accumulator)
