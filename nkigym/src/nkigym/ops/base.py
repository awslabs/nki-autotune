"""NKIOp base class with auto-registry and schedule render class methods."""

from typing import Any, ClassVar


def _op_display_name(value: object) -> str:
    """Extract display name from an NKI op value (str or callable).

    Args:
        value: Either a string name or an object with ``__name__``.

    Returns:
        Human-readable name suitable for NKI source rendering.
    """
    return value if isinstance(value, str) else getattr(value, "__name__", str(value))


class NKIOp:
    """Base for all NKI operator definitions.

    Subclasses with a non-empty ``op_name`` are auto-registered.
    Schedule-based rendering uses class attributes (OPERAND_AXES,
    OUTPUT_AXES, TILE_LIMITS) and class methods (render_compute,
    render_post_compute).

    Attributes:
        op_name: Registry key (e.g. ``"nc_matmul"``, ``"activation"``).
        OPERAND_AXES: Maps operand name to axis label tuple.
        OUTPUT_AXES: Output axis labels.
        TILE_LIMITS: Per-axis tile size overrides.
    """

    op_name: ClassVar[str] = ""
    _registry: ClassVar[dict[str, type["NKIOp"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register subclasses with non-empty op_name."""
        super().__init_subclass__(**kwargs)
        if cls.op_name:
            NKIOp._registry[cls.op_name] = cls

    @classmethod
    def all_ops(cls) -> dict[str, type["NKIOp"]]:
        """Return a copy of all registered NKIOp subclasses.

        Returns:
            Dictionary mapping op_name to NKIOp subclass.
        """
        return dict(cls._registry)

    @classmethod
    def render_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render schedule compute line with pre-computed slice expressions.

        Args:
            dst_expr: Destination expression (e.g. ``"psum_acc[0:128, 0:128]"``).
            operand_exprs: Operand name to subscripted expression mapping.
            config_kwargs: Non-tensor keyword arguments from the op call.

        Returns:
            NKI source line for this compute operation.
        """
        raise NotImplementedError

    @classmethod
    def render_post_compute(
        cls, dst_expr: str, operand_exprs: dict[str, str], config_kwargs: tuple[tuple[str, object], ...]
    ) -> str:
        """Render a post-compute op line (e.g. activation, tensor_tensor).

        Args:
            dst_expr: Destination SBUF expression.
            operand_exprs: Operand name to subscripted SBUF expression mapping.
            config_kwargs: Non-tensor keyword arguments from the op call.

        Returns:
            NKI source line for this post-compute operation.
        """
        raise NotImplementedError
