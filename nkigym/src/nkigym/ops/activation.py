"""Element-wise activation: dst = op(data)."""

from collections.abc import Callable

import numpy as np

from nkigym.codegen.context import _LoweringContext, get_kwarg, value_to_nki
from nkigym.ir.tensor import TensorRef
from nkigym.ir.types import GymStatement
from nkigym.ops.base import GymOp, Tensor


class ActivationOp(GymOp):
    """Element-wise activation function applied to a tensor.

    The activation function is passed as the ``op`` keyword argument
    to ``simulate``. Defaults to identity if not specified.
    """

    op_name = "activation"
    inputs = (Tensor("data", ("P", "F")),)
    outputs = (Tensor("result", ("P", "F")),)

    @classmethod
    def simulate(cls, data: np.ndarray, **kwargs: object) -> np.ndarray:  # type: ignore[override]
        """Apply activation function to input.

        Args:
            data: Input array of shape [P, F].
            op: Activation function (e.g., ``np.tanh``). Identity if None.

        Returns:
            Activated array of same shape.
        """
        op: Callable[[np.ndarray], np.ndarray] = kwargs.get("op")  # type: ignore[assignment]
        result = data
        if op is not None:
            result = op(data)
        return result

    @classmethod
    def output_shape(cls, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Output shape matches input shape.

        Args:
            input_shapes: Tuple containing (data_shape,).

        Returns:
            Same shape as input.
        """
        return input_shapes[0]

    @classmethod
    def to_nki(cls, stmt: GymStatement, ctx: _LoweringContext) -> list[str]:  # type: ignore[override]
        """Lower activation to ``nisa.activation``.

        Args:
            stmt: The activation statement.
            ctx: Lowering context.

        Returns:
            List of NKI source lines.
        """
        data_ref = get_kwarg(stmt, "data")
        op_str = get_kwarg(stmt, "op")

        if not isinstance(data_ref, TensorRef):
            raise ValueError("activation missing data operand")

        data = ctx.subscript(data_ref)
        out_name = stmt.output.name
        out_sub = ctx.subscript(stmt.output)
        ctx.buffers[out_name] = "SBUF"

        func_str = "nl.identity"
        if op_str is not None:
            func_str = value_to_nki(op_str)

        shape_str = repr(stmt.output.shape)
        return [
            f"{out_name} = nl.ndarray({shape_str}, dtype={ctx.dtype}, buffer=nl.sbuf)",
            f"nisa.activation(dst={out_sub}, op={func_str}, data={data})",
        ]
