"""Envelope IR for an ``f_nkigym`` kernel.

:class:`KernelIR` bundles the three passes that lift a source-level
``@nkigym_kernel`` callable into a canonical schedule plus its
dependency view:

1. :class:`~nkigym.ir.dimension_analysis.DimensionAnalysis` — symbolic
   trace + cross-op dim unification (concrete dim ids, per-tensor
   shapes, per-op axis maps).
2. :class:`~nkigym.ir.tree.KernelTree` — canonical ``networkx``-backed
   schedule tree built from the analysis.
3. :class:`~nkigym.ir.dependency.Dependency` — producer-consumer graph
   over the tree's leaves, used by rewrite atoms (``ComputeAt`` etc.)
   to check that moves preserve dataflow order.

:func:`build_initial_ir` runs all three passes in order and returns
the envelope. :meth:`KernelIR.dump` writes every diagram side-by-side.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkigym.ir.dependency import Dependency
from nkigym.ir.dimension_analysis import DimensionAnalysis, analyze_dimensions
from nkigym.ir.tree import KernelTree, build_initial_tree


@dataclass
class KernelIR:
    """Envelope holding analysis, schedule tree, and dependency graph.

    Attributes:
        analysis: Dim-unification result produced by
            :func:`analyze_dimensions`.
        tree: Canonical schedule tree produced by
            :func:`build_initial_tree`.
        dependency: Producer-consumer graph derived from ``tree``.
    """

    analysis: DimensionAnalysis
    tree: KernelTree
    dependency: Dependency

    def dump(self, cache_dir: str | Path) -> None:
        """Write ``tree.*`` and ``dependency.*`` diagrams into ``cache_dir``."""
        self.tree.dump(cache_dir)
        self.dependency.dump(cache_dir)


def build_initial_ir(func: Callable[..., Any], input_specs: dict[str, tuple[int, ...]]) -> KernelIR:
    """Run dim analysis, build the schedule tree, then derive the dependency graph.

    Args:
        func: An ``@nkigym_kernel``-decorated callable.
        input_specs: ``{param_name: shape}`` for every positional param.

    Returns:
        A populated :class:`KernelIR` envelope.
    """
    analysis = analyze_dimensions(func, input_specs)
    tree = build_initial_tree(analysis)
    dependency = Dependency(tree)
    return KernelIR(analysis=analysis, tree=tree, dependency=dependency)


__all__ = ["KernelIR", "build_initial_ir"]
