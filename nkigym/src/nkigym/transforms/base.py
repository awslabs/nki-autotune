"""Base classes for the rewrite-transform interface.

Each concrete transform under :mod:`nkigym.transforms` subclasses
:class:`Transform` and exposes:

* ``analyze(ir) -> list[TransformOption]`` — enumerate every legal
  option for this transform on ``ir``.
* ``apply(ir, option) -> KernelIR`` — re-check legality, deep-copy
  ``ir``, mutate the copy, return it. Raises
  :class:`TransformLegalityError` on illegal options. Loud failures
  only — no try/except recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from nkigym.ir import KernelIR


@dataclass(frozen=True)
class TransformOption:
    """Marker base for per-transform option payloads.

    Subclasses are frozen dataclasses (so options are hashable, useful
    for deduplication in samplers).
    """


class TransformLegalityError(ValueError):
    """Raised by :meth:`Transform.apply` when ``option`` is illegal for ``ir``."""


_OptionT = TypeVar("_OptionT", bound=TransformOption)


class Transform(Generic[_OptionT]):
    """Base class for stateless rewrite transforms.

    Subclasses override :meth:`analyze` and :meth:`apply`. Instances
    carry no state — the same instance can be reused across many
    ``ir``'s.
    """

    def analyze(self, ir: KernelIR) -> list[_OptionT]:
        """Return every legal option for this transform on ``ir``."""
        raise NotImplementedError

    def apply(self, ir: KernelIR, option: _OptionT) -> KernelIR:
        """Re-check legality, deep-copy ``ir``, mutate the copy, return it."""
        raise NotImplementedError


__all__ = ["Transform", "TransformLegalityError", "TransformOption"]
