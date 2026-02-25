"""Element-wise tensor-scalar operation."""

from collections.abc import Callable

import numpy as np

from nkigym.codegen.context import get_kwarg
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import GymOp, Tensor


class TensorScalarOp(GymOp):
    """Element-wise tensor-scalar operation with broadcast.

    The binary function is passed as the ``op`` keyword argument
    to ``simulate``. Defaults to multiplication if not specified.
    """

    op_name = "tensor_scalar"
    inputs = (Tensor("data", ("P", "F")), Tensor("operand0", ("P", 1)))
    outputs = (Tensor("result", ("P", "F")),)

    def simulate(self, data: np.ndarray, operand0: float | np.ndarray, **kwargs: object) -> np.ndarray:
        """Apply element-wise tensor-scalar operation.

        Args:
            data: Input tensor array.
            operand0: Scalar or [P, 1] array.
            op: Binary function (e.g., ``np.add``). Defaults to ``np.multiply``.

        Returns:
            Result array of same shape as data.
        """
        op: Callable[[np.ndarray, object], np.ndarray] = kwargs.get("op", np.multiply)
        return op(data, operand0)

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Output shape matches first input (data) shape.

        Args:
            input_shapes: Tuple of (data_shape, operand0_shape).

        Returns:
            Same shape as data.
        """
        return input_shapes[0]

    def to_nki(self, stmt: "GymStatement", ctx: "_LoweringContext") -> list[str]:
        """Lower tensor_scalar to ``nisa.tensor_scalar``.

        Args:
            stmt: The tensor_scalar statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        data_ref = get_kwarg(stmt, "data")
        operand_ref = get_kwarg(stmt, "operand0")
        op_str = get_kwarg(stmt, "op")

        if not isinstance(data_ref, TensorRef):
            raise ValueError("tensor_scalar missing data operand")

        data = ctx.subscript(data_ref)
        out_name = stmt.output.name
        out_sub = ctx.subscript(stmt.output)
        ctx.buffers[out_name] = "SBUF"

        operand = str(operand_ref)
        if isinstance(operand_ref, TensorRef):
            operand = ctx.subscript(operand_ref)

        op_kwarg = ""
        if op_str is not None:
            nki_op = str(op_str).replace("np.", "nl.")
            op_kwarg = f", op0={nki_op}"

        shape_str = repr(stmt.output.shape)
        return [
            f"{out_name} = nl.ndarray({shape_str}, dtype=nl.float32, buffer=nl.sbuf)",
            f"nisa.tensor_scalar(dst={out_sub}, data={data}{op_kwarg}, operand0={operand})",
        ]
