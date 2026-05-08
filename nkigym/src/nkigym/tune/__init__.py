"""Performance tuning for nkigym kernels.

Every rewrite conforms to :class:`KernelRewrite` — ``is_legal(module)`` →
``apply(module) -> KernelModule``.
"""

from typing import Protocol, runtime_checkable

from nkigym.codegen.ir import KernelModule


@runtime_checkable
class KernelRewrite(Protocol):
    """A performance-related kernel transform."""

    def is_legal(self, module: KernelModule) -> bool:
        """Return ``True`` when the rewrite is applicable to the current state."""
        ...

    def apply(self, module: KernelModule) -> KernelModule:
        """Return the post-transform :class:`KernelModule`.

        Callers must check :meth:`is_legal` first; ``apply`` on an illegal
        input is not guaranteed to raise.
        """
        ...


__all__ = ["KernelRewrite"]
