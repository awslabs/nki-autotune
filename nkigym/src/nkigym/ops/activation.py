"""Element-wise activation op definition for schedule-based rendering."""

from typing import ClassVar

from nkigym.ops.base import NKIOp


class NKIActivation(NKIOp):
    """Activation: ``nisa.activation(dst=..., data=..., op=...)``.

    Attributes:
        op_name: Registry key ``"activation"``.
        OPERAND_AXES: Single operand ``data`` with axes ``(P, F)``.
        OUTPUT_AXES: Output axes ``(P, F)`` — element-wise, no reduction.
        TILE_LIMITS: No tile size overrides.
    """

    op_name: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[tuple[str, str]] = ("P", "F")
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": 128}

    @classmethod
    def render_post_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render activation as a post-compute op.

        Args:
            dst_expr: Destination SBUF expression.
            operand_exprs: Must contain ``"data"`` key with source expression.
            config_kwargs: Must contain ``("op", <func>)`` for the activation.

        Returns:
            NKI activation source line.
        """
        kwargs = dict(config_kwargs)
        value = kwargs.get("op", "identity")
        name = value if isinstance(value, str) else getattr(value, "__name__", str(value))
        return f"nisa.activation(dst={dst_expr}, data={operand_exprs['data']}, op=nl.{name})"
