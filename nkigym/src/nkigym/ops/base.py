"""NKIOp base class with __call__ simulation and RenderContext.

Each NKIOp subclass maps 1:1 to a real nisa.* ISA instruction.
Subclasses implement __call__() for CPU simulation (numpy at float64)
and render() for emitting NKI source lines.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class Tensor:
    """A tensor visible in the current rendering scope.

    Attributes:
        name: Tensor name (e.g. ``"lhs_T"``).
        shape: Full shape (e.g. ``(2048, 2048)``).
        dtype: Dtype string (e.g. ``"bfloat16"``).
        location: Where it lives: ``"hbm"``, ``"sbuf"``, or ``"psum"``.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    location: str


@dataclass
class RenderContext:
    """Running context passed to each NKIOp.render().

    The renderer builds this from input_specs (HBM tensors),
    then passes it to each op. Ops read their inputs from
    tensors and can add new tensors (SBUF/PSUM buffers).

    Attributes:
        tensors: All tensors in scope, keyed by name.
        kwargs: Non-tensor keyword arguments from the op call.
    """

    tensors: dict[str, Tensor] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)


class NKIOp:
    """Base for all NKI operator definitions.

    Subclasses define:
    - NAME: maps to the nisa.* call name
    - OPERAND_AXES: maps operand name to axis label tuple
    - OUTPUT_AXES: maps output name to axis label tuple

    Attributes:
        NAME: Registry key and ISA call name.
        OPERAND_AXES: Maps operand name to axis label tuple.
        OUTPUT_AXES: Maps output name to axis label tuple.
    """

    NAME: ClassVar[str] = ""
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Any:
        """CPU simulation using numpy at float64 precision."""

    @abstractmethod
    def render(self, ctx: RenderContext, operand_map: dict[str, str]) -> list[str]:
        """Emit NKI source lines for this op.

        Args:
            ctx: Running render context with tensors and kwargs.
            operand_map: Maps op slot name to tensor name in ctx
                (e.g. ``{"stationary": "lhs_T", "moving": "rhs"}``).

        Returns:
            List of NKI source lines (without base indent).
        """
