"""Element-wise binary operation between two tensors."""

from collections.abc import Callable

import numpy as np

from nkigym.ops.base import GymOp, Tensor


class TensorTensorOp(GymOp):
    """Element-wise binary operation between two tensors.

    The binary function is passed as the ``op`` keyword argument
    to ``simulate``. Defaults to addition if not specified.
    """

    op_name = "tensor_tensor"
    inputs = (Tensor("data1", ("P", "F")), Tensor("data2", ("P", "F")))
    outputs = (Tensor("result", ("P", "F")),)

    def simulate(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        *,
        op: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        """Apply element-wise binary operation.

        Args:
            data1: First input array.
            data2: Second input array.
            op: Binary function (e.g., ``np.multiply``). Defaults to ``np.add``.

        Returns:
            Result array of same shape.
        """
        if op is None:
            op = np.add
        return op(data1, data2)

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Output shape matches first input shape.

        Args:
            input_shapes: Tuple of (data1_shape, data2_shape).

        Returns:
            Same shape as first input.
        """
        return input_shapes[0]
