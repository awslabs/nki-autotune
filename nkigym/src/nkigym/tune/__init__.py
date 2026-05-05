"""Performance tuning for nkigym kernels.

Exposes the :class:`KernelRewrite` protocol shared by every structural
or graph rewrite applied inside the ``"tune"`` stage of
``nkigym_compile``. Concrete rewrites live in sibling modules:

* :mod:`nkigym.tune.fuse_loops` — :class:`FuseLoops` loop fusion atom.
* :mod:`nkigym.tune.stage` — the ``_run_tune`` stage driver.
"""

from typing import Protocol, runtime_checkable

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import LoopForest


@runtime_checkable
class KernelRewrite(Protocol):
    """A performance-related kernel transform.

    Rewrites mutate the ``(op_graph, forest)`` pair. Structural rewrites
    (e.g. :class:`FuseLoops`) leave ``op_graph`` untouched; graph
    rewrites mutate it. The ``tune`` stage treats both uniformly.
    """

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return ``True`` when the rewrite is applicable to the current state."""
        ...

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Return the post-transform ``(op_graph, forest)`` pair.

        Callers must check :meth:`is_legal` first; ``apply`` on an
        illegal input is not guaranteed to raise.
        """
        ...


__all__ = ["KernelRewrite"]
