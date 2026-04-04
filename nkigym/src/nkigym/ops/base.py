"""NKIOp base class with __call__ simulation and render_isa code generation.

Each NKIOp subclass maps 1:1 to a real nisa.* ISA instruction.
Subclasses implement __call__() for CPU simulation (numpy at float64)
and render_isa() for emitting the ISA call body inside a loop nest.

The eager kernel generator (codegen.eager) builds the loop nest and
calls render_isa() for each op.
"""

from abc import abstractmethod
from typing import Any, ClassVar

from nkigym.codegen.ir import RenderContext


def _op_display_name(op: object) -> str:
    """Convert an op value to its NKI display name.

    Handles numpy ufuncs, strings, and other callables.

    Args:
        op: Operation value (numpy function, string, or callable).

    Returns:
        String name suitable for ``nl.<name>`` rendering.
    """
    result = op if isinstance(op, str) else getattr(op, "__name__", str(op))
    return result


def _get_output_axes_tuple(cls: type) -> tuple[str, ...]:
    """Extract the primary output's axis labels from OUTPUT_AXES.

    Returns the first entry's axes for multi-output ops.

    Args:
        cls: NKIOp subclass (or its type).

    Returns:
        Axis labels of the primary output (e.g. ``("M", "N")``).
    """
    raw = getattr(cls, "OUTPUT_AXES", ())
    result = raw
    if isinstance(raw, dict):
        result = next(iter(raw.values()))
    return result


def _get_schedule_output_axes(cls: type) -> tuple[str, ...]:
    """Extract output axes for the schedule renderer.

    Uses ``SCHEDULE_OUTPUT_AXES`` override if present (for multi-output
    ops like activation_reduce where the schedule renderer only sees
    the reduction output). Falls back to ``_get_output_axes_tuple``.

    Args:
        cls: NKIOp subclass (or its type).

    Returns:
        Axis labels for schedule rendering.
    """
    override = getattr(cls, "SCHEDULE_OUTPUT_AXES", None)
    result = _get_output_axes_tuple(cls)
    if override is not None:
        result = override
    return result


class NKIOp:
    """Base for all NKI operator definitions.

    Subclasses define:
    - NAME: maps to the nisa.* call name
    - OPERAND_AXES: maps operand name to axis label tuple
    - OUTPUT_AXES: maps output name to axis label tuple
    - MAX_TILE_SIZES: per-axis tile size limits

    Attributes:
        NAME: Registry key and ISA call name.
        OPERAND_AXES: Maps operand name to axis label tuple.
        OUTPUT_AXES: Maps output name to axis label tuple.
        MAX_TILE_SIZES: Per-axis tile size overrides.
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {}
    _registry: ClassVar[dict[str, type["NKIOp"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register subclasses with non-empty NAME."""
        super().__init_subclass__(**kwargs)
        if cls.NAME:
            NKIOp._registry[cls.NAME] = cls

    @classmethod
    def all_ops(cls) -> dict[str, type["NKIOp"]]:
        """Return a copy of all registered NKIOp subclasses.

        Returns:
            Dictionary mapping NAME to NKIOp subclass.
        """
        return dict(cls._registry)

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation using numpy at float64 precision.

        Takes input arrays + config, returns output array(s).
        """

    @abstractmethod
    def render_isa(self, ctx: RenderContext) -> str:
        """Emit the ISA call inside the innermost loop.

        If any tile dim exceeds MAX_TILE_SIZES, emit an
        in-place sub-loop that iterates in chunks.

        Args:
            ctx: Render context with outputs, operands, config.

        Returns:
            NKI source line(s) for the ISA call.
        """
