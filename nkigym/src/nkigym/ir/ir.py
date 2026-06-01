"""Envelope IR for an ``f_nkigym`` kernel.

:class:`KernelIR` is the single envelope. It carries the kernel
signature, return-tensor identity, schedule tree, and producer-consumer
dependency graph. Per-buffer and per-axis information is derived from
the tree on demand via :meth:`KernelIR.all_buffers` and
:meth:`KernelIR.axis_extent` — caching them on the envelope leads to
invalidation churn when transforms move buffers between blocks.

:func:`build_initial_ir` runs dim unification, tree construction, and
dependency graph construction, then flattens the analysis output onto
a :class:`KernelIR` instance. :meth:`KernelIR.dump` writes the tree
and dependency diagrams side-by-side into a cache directory.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nkigym.ir.dependency import Dependency
from nkigym.ir.dimension_analysis import analyze_dimensions
from nkigym.ir.tree import BlockNode, Buffer, KernelTree, build_initial_tree
from nkigym.ir.tree_visualize import dump_tree


@dataclass
class KernelIR:
    """Envelope holding signature and the schedule tree.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_name: Identifier in the kernel's ``return`` statement.
        tree: Canonical schedule tree.
        dependency: Producer-consumer graph derived from ``tree``.
        param_buffers: Parameter buffer metadata (shape/dtype/location).
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tree: KernelTree
    dependency: Dependency
    param_buffers: dict[str, Buffer] = field(default_factory=dict)

    def all_buffers(self) -> dict[str, Buffer]:
        """Walk every :class:`BlockNode` in pre-order; return ``name -> Buffer`` including parameters."""
        out: dict[str, Buffer] = dict(self.param_buffers)
        for nid in self.tree.blocks():
            block = self.tree.data(nid)
            assert isinstance(block, BlockNode)
            for buf in block.alloc_buffers:
                if buf.name in out:
                    raise ValueError(f"buffer {buf.name!r} declared by two blocks")
                out[buf.name] = buf
        return out

    def buffer(self, name: str) -> Buffer:
        """Resolve a buffer by name; raises :class:`KeyError` if absent."""
        buffers = self.all_buffers()
        if name not in buffers:
            raise KeyError(f"buffer {name!r} not found in any block.alloc_buffers")
        return buffers[name]

    def axis_extent(self, axis: str) -> int:
        """Return the extent of the iter_var named ``axis``.

        Walks blocks in pre-order; returns the first ``IterVar`` whose
        ``axis`` matches. Raises :class:`KeyError` if the axis is not
        declared anywhere in the tree.
        """
        for nid in self.tree.blocks():
            block = self.tree.data(nid)
            assert isinstance(block, BlockNode)
            for iv in block.iter_vars:
                if iv.axis == axis:
                    return iv.dom[1] - iv.dom[0]
        raise KeyError(f"no iter_var with axis {axis!r}")

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
        """Render signature + buffers as Markdown."""
        lines: list[str] = [
            f"# `{self.func_name}`",
            "",
            "## Signature",
            "",
            f"- **Params**: {', '.join(f'`{p}`' for p in self.param_names) or '_(none)_'}",
            f"- **Returns**: `{self.return_name}`",
            "",
            "## Buffers",
            "",
            "| Name | Location | Dtype | Shape |",
            "| ---- | -------- | ----- | ----- |",
        ]
        for buf in self.all_buffers().values():
            shape = "(" + ", ".join(str(s) for s in buf.shape) + ")"
            lines.append(f"| `{buf.name}` | `{buf.location}` | `{buf.dtype}` | `{shape}` |")
        lines.append("")
        return "\n".join(lines)


def build_initial_ir(func: Callable[..., Any], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Run dim analysis, build the schedule tree, derive the dependency graph, flatten.

    Args:
        func: An ``@nkigym_kernel``-decorated callable.
        input_specs: ``{param_name: (shape, dtype)}`` for every positional param.

    Returns:
        A populated :class:`KernelIR` envelope.
    """
    analysis = analyze_dimensions(func, input_specs)
    tree = build_initial_tree(analysis)
    dependency = Dependency(tree)
    param_buffers = {
        name: Buffer(
            name=name,
            shape=tuple(analysis.tensors[name].shape),
            dtype=analysis.tensors[name].dtype,
            location=analysis.tensors[name].location,
        )
        for name in analysis.param_names
    }
    return KernelIR(
        func_name=analysis.func_name,
        param_names=analysis.param_names,
        return_name=analysis.return_name,
        tree=tree,
        dependency=dependency,
        param_buffers=param_buffers,
    )


__all__ = ["KernelIR", "build_initial_ir"]
