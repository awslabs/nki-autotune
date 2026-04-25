"""Base class for ``KernelIR`` graph rewrites.

Every rewrite is a two-stage transformation:

1. ``analyze(ir)`` — pure inspection. Walks the IR, identifies every
   site that matches the rewrite's pattern, returns a list of opaque
   *match* records. No IR mutation. Safe to call, inspect, or drop.
2. ``apply(ir, matches)`` — consumes the match list and produces a new
   ``KernelIR`` with those specific rewrites applied. The caller picks
   which matches to forward — all of them, a subset, or none — so the
   rewrite is no longer all-or-nothing.

The default ``__call__`` runs ``apply(ir, analyze(ir))`` for backwards
compatible "rewrite everything" behaviour.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from nkigym.kernel_ir.ir import KernelIR

Match = TypeVar("Match")


class GraphRewrite(ABC, Generic[Match]):
    """Two-stage ``KernelIR`` graph rewrite.

    Subclasses parameterise ``Match`` — the opaque record describing
    one pattern occurrence — and implement ``analyze`` + ``apply``.
    """

    @abstractmethod
    def analyze(self, ir: KernelIR) -> list[Match]:
        """Return every site in ``ir`` that matches this rewrite's pattern."""

    @abstractmethod
    def apply(self, ir: KernelIR, matches: list[Match]) -> KernelIR:
        """Produce a new IR with ``matches`` rewritten. Caller curates the list."""

    def __call__(self, ir: KernelIR) -> KernelIR:
        """Rewrite every match — convenience wrapper around ``analyze`` + ``apply``."""
        return self.apply(ir, self.analyze(ir))
