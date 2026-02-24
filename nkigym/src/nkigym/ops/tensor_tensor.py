"""Element-wise binary operation between two tensors."""

from collections.abc import Callable

import numpy as np

from nkigym.codegen.context import get_kwarg
from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import GymOp, Tensor


class TensorTensorOp(GymOp):
    """Element-wise binary operation between two tensors.

    The binary function is passed as the ``op`` keyword argument
    to ``simulate``. Defaults to addition if not specified.
    """

    op_name = "tensor_tensor"
    inputs = (Tensor("data1", ("P", "F")), Tensor("data2", ("P", "F")))
    outputs = (Tensor("result", ("P", "F")),)

    def simulate(self, data1: np.ndarray, data2: np.ndarray, **kwargs: object) -> np.ndarray:
        """Apply element-wise binary operation.

        Args:
            data1: First input array.
            data2: Second input array.
            op: Binary function (e.g., ``np.multiply``). Defaults to ``np.add``.

        Returns:
            Result array of same shape.
        """
        op: Callable[[np.ndarray, np.ndarray], np.ndarray] = kwargs.get("op", np.add)
        return op(data1, data2)

    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Output shape matches first input shape.

        Args:
            input_shapes: Tuple of (data1_shape, data2_shape).

        Returns:
            Same shape as first input.
        """
        return input_shapes[0]

    def to_nki(self, stmt: "GymStatement", ctx: "_LoweringContext") -> list[str]:
        """Lower tensor_tensor to ``nisa.tensor_tensor``.

        Args:
            stmt: The tensor_tensor statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        d1_ref = get_kwarg(stmt, "data1")
        d2_ref = get_kwarg(stmt, "data2")
        op_str = get_kwarg(stmt, "op")

        if not isinstance(d1_ref, TensorRef) or not isinstance(d2_ref, TensorRef):
            raise ValueError("tensor_tensor missing operands")

        d1 = ctx.subscript(d1_ref)
        d2 = ctx.subscript(d2_ref)
        out_name = stmt.output.name
        ctx.buffers[out_name] = "SBUF"

        op_part = ""
        if op_str is not None:
            nki_op = str(op_str).replace("np.", "nl.")
            op_part = f", op={nki_op}"

        shape_str = repr(stmt.output.shape)
        return [
            f"{out_name} = nl.ndarray({shape_str}, dtype=nl.float32, buffer=nl.sbuf)",
            f"nisa.tensor_tensor(dst={out_name}, data1={d1}, data2={d2}{op_part})",
        ]
