"""Performance tuning for nkigym kernels.

Every rewrite conforms to :class:`KernelRewrite` — ``is_legal(module)`` →
``apply(module) -> KernelModule``.
"""

from typing import Protocol, runtime_checkable

from nkigym.codegen.ir import KernelModule


class AtomLegalityError(Exception):
    """Raised by ``KernelRewrite.apply`` when the atom is stale against the current module.

    Atoms are typically enumerated against one module and later applied
    against a module reached via a composition of other rewrites. When a
    preceding rewrite invalidates a captured precondition (e.g. a
    :class:`ComputeAt` narrows the LCA so an :class:`Annotate`
    ``buffer_degree`` exceeds the new ``num_tiles / required_tiles`` cap),
    the atom's ``apply`` must re-validate and raise this error instead of
    silently producing an incorrect module.

    Callers that orchestrate atom composition (the batch sampler) catch
    :class:`AtomLegalityError` and skip to the next atom; this keeps
    generic :class:`ValueError` available for real bugs.
    """


@runtime_checkable
class KernelRewrite(Protocol):
    """A performance-related kernel transform."""

    def is_legal(self, module: KernelModule) -> bool:
        """Return ``True`` when the rewrite is applicable to the current state."""
        ...

    def apply(self, module: KernelModule) -> KernelModule:
        """Return the post-transform :class:`KernelModule`.

        Callers must check :meth:`is_legal` first; ``apply`` on an illegal
        input may raise :class:`AtomLegalityError` for atoms that
        implement apply-time re-validation.
        """
        ...


__all__ = ["AtomLegalityError", "KernelRewrite"]
