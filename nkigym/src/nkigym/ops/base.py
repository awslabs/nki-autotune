"""NKIOp frozen dataclass base with registry, simulate, mac_count, and render."""

from dataclasses import dataclass
from typing import Any, ClassVar

from nkigym.ir.tensor import TensorRef


def _render_ref(ref: TensorRef) -> str:
    """Render TensorRef as 'name[start:end, start:end]'.

    Args:
        ref: Tensor reference with name and slices.

    Returns:
        Subscripted string representation.
    """
    slices = ", ".join(f"{s}:{e}" for s, e in ref.slices)
    return f"{ref.name}[{slices}]"


@dataclass(frozen=True)
class NKIOp:
    """Base for all NKI statements.

    Subclasses with a non-empty ``op_name`` are auto-registered.
    Compute ops override ``simulate`` and optionally ``mac_count``.
    Pure IR nodes (alloc, dma_copy, tensor_copy) only implement ``render``.
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

    def mac_count(self) -> int:
        """Count MACs for this op invocation.

        Default returns 0. Override in compute-heavy ops.

        Returns:
            Number of multiply-accumulate operations.
        """
        return 0

    def simulate(self, env: dict[str, Any]) -> None:
        """Execute this statement in the simulation environment.

        Args:
            env: Mutable variable environment mapping names to numpy arrays.
        """
        raise NotImplementedError

    def render(self) -> str:
        """Render this statement as an NKI source line.

        Returns:
            NKI source code string.
        """
        raise NotImplementedError
