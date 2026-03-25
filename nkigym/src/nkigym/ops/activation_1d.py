"""Element-wise activation op for 1D column vectors."""

from typing import ClassVar

from nkigym.ops.base import NKIOp, _op_display_name


class NKIActivation1D(NKIOp):
    """1D Activation: ``nisa.activation(dst=..., data=..., op=...)``.

    Same ISA call as 2D activation but operates on 1D ``(P,)`` data.
    Disambiguated from ``NKIActivation`` via post-parse shape inference.

    Attributes:
        op_name: Registry key ``"activation_1d"``.
        OPERAND_AXES: Single operand ``data`` with axes ``(P,)``.
        OUTPUT_AXES: Output axes ``(P,)`` --- element-wise, 1D.
        TILE_LIMITS: Partition axis capped at 128.
    """

    op_name: ClassVar[str] = "activation_1d"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P",)}
    OUTPUT_AXES: ClassVar[tuple[str, ...]] = ("P",)
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    @classmethod
    def render_post_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render 1D activation as an inter-pass op.

        Args:
            dst_expr: Destination SBUF expression.
            operand_exprs: Must contain ``"data"`` key with source expression.
            config_kwargs: Must contain ``("op", <str>)`` for the activation.

        Returns:
            NKI activation source line.
        """
        kwargs = dict(config_kwargs)
        name = _op_display_name(kwargs.get("op", "identity"))
        return f"nisa.activation(dst={dst_expr}, data={operand_exprs['data']}, op=nl.{name})"
