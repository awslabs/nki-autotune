"""Element-wise activation: dst = op(data)."""

from collections.abc import Callable

import numpy as np

from nkigym.ops.base import GymOp, Tensor


class ActivationOp(GymOp):
    """Element-wise activation function applied to a tensor.

    The activation function is passed as the ``op`` keyword argument
    to ``simulate``. Defaults to identity if not specified.
    """

    op_name = "activation"
    inputs = (Tensor("data", ("P", "F")),)
    outputs = (Tensor("result", ("P", "F")),)

    def simulate(
        self, data: np.ndarray, *, op: Callable[[np.ndarray], np.ndarray] | None = None, **kwargs: object
    ) -> np.ndarray:
        """Apply activation function to input.

        Args:
            data: Input array of shape [P, F].
            op: Activation function (e.g., ``np.tanh``). Identity if None.

        Returns:
            Activated array of same shape.
        """
        if op is None:
            return data
        return op(data)

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Output shape matches input shape.

        Args:
            input_shapes: Tuple containing (data_shape,).

        Returns:
            Same shape as input.
        """
        return input_shapes[0]
