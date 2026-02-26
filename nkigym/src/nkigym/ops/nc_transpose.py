"""Transpose: swaps partition and free axes."""

import numpy as np

from nkigym.codegen.context import get_kwarg
from nkigym.ir.tensor import TensorRef
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

    def to_nki(self, stmt: "GymStatement", ctx: "_LoweringContext") -> list[str]:
        """Lower nc_transpose to ``nisa.nc_transpose``.

        Args:
            stmt: The nc_transpose statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        data_ref = get_kwarg(stmt, "data")

        if not isinstance(data_ref, TensorRef):
            raise ValueError("nc_transpose missing data operand")

        data = ctx.subscript(data_ref)
        out_name = stmt.output.name
        out_sub = ctx.subscript(stmt.output)
        ctx.buffers[out_name] = "SBUF"

        shape_str = repr(stmt.output.shape)
        return [
            f"{out_name} = nl.ndarray({shape_str}, dtype={ctx.dtype}, buffer=nl.sbuf)",
            f"nisa.nc_transpose(dst={out_sub}, data={data})",
        ]
