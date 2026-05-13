"""Envelope IR for an ``f_nkigym`` kernel.

:class:`KernelIR` bundles the two passes that lift a source-level
``@nkigym_kernel`` callable into a canonical schedule:

1. :class:`~nkigym.ir.dimension_analysis.DimensionAnalysis` — symbolic
   trace + cross-op dim unification (concrete dim ids, per-tensor
   shapes, per-op axis maps).
2. :class:`~nkigym.ir.tree.KernelTree` — canonical ``networkx``-backed
   schedule tree built from the analysis.

:func:`build_initial_ir` is the single entry point that runs the
analysis once and feeds it into :func:`build_initial_tree`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkigym.ir.dimension_analysis import DimensionAnalysis, analyze_dimensions
from nkigym.ir.tree import KernelTree, build_initial_tree


@dataclass
class KernelIR:
    """Envelope holding the analysis and the canonical schedule tree.

    Attributes:
        analysis: Dim-unification result produced by
            :func:`analyze_dimensions`.
        tree: Canonical schedule tree produced by
            :func:`build_initial_tree`.
    """

    analysis: DimensionAnalysis
    tree: KernelTree

    def dump(self, cache_dir: str | Path) -> None:
        """Delegate to :meth:`KernelTree.dump` for diagram output."""
        self.tree.dump(cache_dir)


def build_initial_ir(func: Callable[..., Any], input_specs: dict[str, tuple[int, ...]]) -> KernelIR:
    """Run dim analysis on ``func`` and build the canonical schedule tree.

    Args:
        func: An ``@nkigym_kernel``-decorated callable.
        input_specs: ``{param_name: shape}`` for every positional param.

    Returns:
        A populated :class:`KernelIR` envelope.
    """
    analysis = analyze_dimensions(func, input_specs)
    tree = build_initial_tree(analysis)
    return KernelIR(analysis=analysis, tree=tree)


__all__ = ["KernelIR", "build_initial_ir"]
