"""1D compound tensor-scalar op with literal operands."""

from typing import ClassVar

from nkigym.ops.base import NKIOp, _op_display_name


class NKITensorScalarConst(NKIOp):
    """Tensor-scalar with literal constants: compound two-op chain.

    ``nisa.tensor_scalar(dst=..., data=..., operand0=val, op0=..., op1=..., operand1=val)``

    Operates on 1D ``(P,)`` data with scalar literal operands.
    Supports compound ``op1``/``operand1`` for chained operations.

    Attributes:
        op_name: Registry key ``"tensor_scalar_const"``.
        OPERAND_AXES: Single operand ``data`` with axes ``(P,)``.
        OUTPUT_AXES: Output axes ``(P,)``.
        TILE_LIMITS: Partition axis capped at 128.
    """

    op_name: ClassVar[str] = "tensor_scalar_const"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P",)}
    OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P",)
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    @classmethod
    def render_post_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render compound tensor_scalar as an inter-pass op.

        Args:
            dst_expr: Destination SBUF expression.
            operand_exprs: Must contain ``"data"`` key with source expression.
            config_kwargs: Contains ``op0``, ``operand0``, and optionally ``op1``, ``operand1``.

        Returns:
            NKI tensor_scalar source line with optional compound op.
        """
        kwargs = dict(config_kwargs)
        op0_name = _op_display_name(kwargs.get("op0", "multiply"))
        operand0 = kwargs.get("operand0", 0)
        base = (
            f"nisa.tensor_scalar(dst={dst_expr}, data={operand_exprs['data']}, operand0={operand0}, op0=nl.{op0_name}"
        )
        op1 = kwargs.get("op1")
        if op1 is not None:
            op1_name = _op_display_name(op1)
            operand1 = kwargs.get("operand1", 0)
            base = f"{base}, op1=nl.{op1_name}, operand1={operand1}"
        return f"{base})"
