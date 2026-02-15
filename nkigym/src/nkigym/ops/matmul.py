"""Matrix multiplication: dst = stationary.T @ moving."""

import numpy as np

from nkigym.ops.base import GymOp, Tensor


class MatmulOp(GymOp):
    """Matrix multiplication contracting the K dimension.

    Computes ``stationary.T @ moving`` where stationary is [K, M]
    and moving is [K, N], producing output [M, N].
    """

    op_name = "nc_matmul"
    inputs = (Tensor("stationary", ("K", "M")), Tensor("moving", ("K", "N")))
    outputs = (Tensor("result", ("M", "N")),)

    def simulate(
        self, stationary: np.ndarray, moving: np.ndarray, *, acc: np.ndarray | None = None, **kwargs: object
    ) -> np.ndarray:
        """Compute stationary.T @ moving, optionally accumulating.

        Args:
            stationary: Left-hand side array of shape [K, M].
            moving: Right-hand side array of shape [K, N].
            acc: Accumulation buffer to add the result into.

        Returns:
            Result array of shape [M, N], or acc + result if acc is provided.
        """
        result = np.matmul(stationary.T, moving)
        if acc is not None:
            return acc + result
        return result

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Compute output shape: [K,M] x [K,N] -> [M,N].

        Args:
            input_shapes: Tuple of (stationary_shape, moving_shape).

        Returns:
            Shape tuple (M, N).
        """
        return (input_shapes[0][1], input_shapes[1][1])
