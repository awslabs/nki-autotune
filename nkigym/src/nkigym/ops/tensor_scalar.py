"""Element-wise tensor-scalar operation."""

from collections.abc import Callable

import numpy as np

from nkigym.ops.base import GymOp, Tensor


class TensorScalarOp(GymOp):
    """Element-wise tensor-scalar operation with broadcast.

    The binary function is passed as the ``op`` keyword argument
    to ``simulate``. Defaults to multiplication if not specified.
    """

    op_name = "tensor_scalar"
    inputs = (Tensor("data", ("P", "F")), Tensor("operand0", ("P", 1)))
    outputs = (Tensor("result", ("P", "F")),)

    def simulate(
        self,
        data: np.ndarray,
        operand0: float | np.ndarray,
        *,
        op: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        """Apply element-wise tensor-scalar operation.

        Args:
            data: Input tensor array.
            operand0: Scalar or [P, 1] array.
            op: Binary function (e.g., ``np.add``). Defaults to ``np.multiply``.

        Returns:
            Result array of same shape as data.
        """
        if op is None:
            op = np.multiply
        return op(data, operand0)

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Output shape matches first input (data) shape.

        Args:
            input_shapes: Tuple of (data_shape, operand0_shape).

        Returns:
            Same shape as data.
        """
        return input_shapes[0]
