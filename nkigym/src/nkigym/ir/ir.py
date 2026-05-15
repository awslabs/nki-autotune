"""Envelope IR for an ``f_nkigym`` kernel.

:class:`KernelIR` is the single envelope. It carries the kernel
signature, return-tensor identity, per-dim sizes, tensor table,
canonical schedule tree, and the producer-consumer dependency graph
that rewrite atoms consult.

:func:`build_initial_ir` runs dim unification, tree construction, and
dependency graph construction, then flattens the analysis output onto
a :class:`KernelIR` instance. :meth:`KernelIR.dump` writes the tree
and dependency diagrams side-by-side into a cache directory.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nkigym.ir.dependency import Dependency
from nkigym.ir.dimension_analysis import TensorDims, analyze_dimensions
from nkigym.ir.tree import KernelTree, build_initial_tree
from nkigym.ir.tree_visualize import dump_tree


@dataclass
class KernelIR:
    """Envelope holding signature, tensor table, schedule tree, and dependency graph.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_name: Identifier in the kernel's ``return`` statement.
        dim_sizes: ``dim_name → extent``.
        tensors: All named tensors, keyed by name.
        tree: Canonical schedule tree.
        dependency: Producer-consumer graph derived from ``tree``.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dim_sizes: dict[str, int]
    tensors: dict[str, TensorDims]
    tree: KernelTree
    dependency: Dependency

    def dump(self, cache_dir: str | Path) -> None:
        """Write ``envelope.md``, ``tree.*``, ``dependency.*``, and a black-formatted ``kernel.py`` into ``cache_dir``."""
        from nkigym.codegen import render

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / "envelope.md").write_text(self._render_envelope_md(), encoding="utf-8")
        dump_tree(self.tree, cache_dir)
        self.dependency.dump(cache_dir)
        kernel_path = cache_path / "kernel.py"
        kernel_path.write_text(render(self), encoding="utf-8")
        subprocess.run(["black", "--quiet", str(kernel_path)], check=True)

    def _render_envelope_md(self) -> str:
        """Render signature + dim_sizes + tensors as Markdown."""
        lines: list[str] = [
            f"# `{self.func_name}`",
            "",
            "## Signature",
            "",
            f"- **Params**: {', '.join(f'`{p}`' for p in self.param_names) or '_(none)_'}",
            f"- **Returns**: `{self.return_name}`",
            "",
            "## Dim sizes",
            "",
            "| Dim | Extent |",
            "| --- | ------ |",
        ]
        for dim, size in self.dim_sizes.items():
            lines.append(f"| `{dim}` | {size} |")
        lines.extend(
            [
                "",
                "## Tensors",
                "",
                "| Name | Origin | Location | Dtype | Shape | Dim ids |",
                "| ---- | ------ | -------- | ----- | ----- | ------- |",
            ]
        )
        params = set(self.param_names)
        for tensor in self.tensors.values():
            origin = "param" if tensor.name in params else "intermediate"
            shape = "(" + ", ".join(str(s) for s in tensor.shape) + ")"
            dim_ids = ", ".join(f"`{d}`" for d in tensor.dim_ids)
            lines.append(
                f"| `{tensor.name}` | {origin} | `{tensor.location}` | `{tensor.dtype}` | `{shape}` | {dim_ids} |"
            )
        lines.append("")
        return "\n".join(lines)


def build_initial_ir(func: Callable[..., Any], input_specs: dict[str, tuple[int, ...]]) -> KernelIR:
    """Run dim analysis, build the schedule tree, derive the dependency graph, flatten.

    Args:
        func: An ``@nkigym_kernel``-decorated callable.
        input_specs: ``{param_name: shape}`` for every positional param.

    Returns:
        A populated :class:`KernelIR` envelope.
    """
    analysis = analyze_dimensions(func, input_specs)
    tree = build_initial_tree(analysis)
    dependency = Dependency(tree)
    return KernelIR(
        func_name=analysis.func_name,
        param_names=analysis.param_names,
        return_name=analysis.return_name,
        dim_sizes=analysis.dim_sizes,
        tensors=analysis.tensors,
        tree=tree,
        dependency=dependency,
    )


__all__ = ["KernelIR", "build_initial_ir"]
