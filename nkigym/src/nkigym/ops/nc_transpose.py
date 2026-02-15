"""Transpose: swaps partition and free axes."""

import numpy as np

from nkigym.ops.base import GymOp, Tensor


class NcTransposeOp(GymOp):
    """Transpose swapping the two axes: data[P, F] -> [F, P]."""

    op_name = "nc_transpose"
    inputs = (Tensor("data", ("P", "F")),)
    outputs = (Tensor("result", ("F", "P")),)

    def simulate(self, data: np.ndarray, **kwargs: object) -> np.ndarray:
        """Transpose the input array.

        Args:
            data: Input array of shape [P, F].

        Returns:
            Transposed array of shape [F, P].
        """
        return np.transpose(data)

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Compute output shape: [P, F] -> [F, P].

        Args:
            input_shapes: Tuple containing (data_shape,).

        Returns:
            Transposed shape tuple (F, P).
        """
        return (input_shapes[0][1], input_shapes[0][0])
