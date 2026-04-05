"""NKIOp base class with __call__ simulation and render code generation.

Each NKIOp subclass maps 1:1 to a real nisa.* ISA instruction.
Subclasses implement __call__() for CPU simulation (numpy at float64)
and render() for emitting a complete loop nest for one op.

Design doc reference: nkigym_ir_guide.md sections 1.1 and 2.1.
"""

from abc import abstractmethod
from typing import Any, ClassVar

from nkigym.codegen.hardware import _PSUM_OPS, Hardware
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
    - AXIS_ROLES: maps logical axis name to physical role

    Attributes:
        NAME: Registry key and ISA call name.
        OPERAND_AXES: Maps operand name to axis label tuple.
        OUTPUT_AXES: Maps output name to axis label tuple.
        AXIS_ROLES: Maps logical axis to ``"partition"`` | ``"free"`` | ``"accumulation"``.
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    AXIS_ROLES: ClassVar[dict[str, str]] = {}
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

    def _free_limit(self, dtype_bytes: int) -> int:
        """Max free-axis tile size for this op.

        Args:
            dtype_bytes: Element size in bytes.

        Returns:
            PSUM_FREE_MAX for TensorEngine ops, else sbuf_free_max.
        """
        result = Hardware.PSUM_FREE_MAX if self.NAME in _PSUM_OPS else Hardware.sbuf_free_max(dtype_bytes)
        return result

    def max_tile_sizes(self, dtype_bytes: int = 4) -> dict[str, int]:
        """Derive per-axis tile limits from Hardware constants.

        - partition -> SBUF_PARTITION_MAX (128)
        - accumulation -> SBUF_PARTITION_MAX (128)
        - free + PSUM ops (nc_matmul, nc_transpose) -> PSUM_FREE_MAX (512)
        - free + other ops -> sbuf_free_max(dtype_bytes)

        Transpose overrides: both dims <= TRANSPOSE_BLOCK.

        Args:
            dtype_bytes: Element size in bytes (default 4 for fp32).

        Returns:
            Maps axis label to max tile size.
        """
        role_limits = {
            "partition": Hardware.SBUF_PARTITION_MAX,
            "accumulation": Hardware.SBUF_PARTITION_MAX,
            "free": self._free_limit(dtype_bytes),
        }
        return {axis: role_limits[role] for axis, role in self.AXIS_ROLES.items()}

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation using numpy at float64 precision.

        Takes input arrays + config, returns output array(s).
        """

    @abstractmethod
    def render(self, ctx: RenderContext) -> list[str]:
        """Emit a complete loop nest for this op.

        Produces all code for one op: output buffer allocation,
        parallel output loops, PSUM accumulator (if reduction),
        reduction loops, DMA loads, ISA call, tensor_copy,
        and DMA store (if final op).

        Args:
            ctx: Render context with all metadata needed for code generation.

        Returns:
            List of NKI source lines (without base indent).
        """
