"""Envelope IR for an ``f_nkigym`` kernel.

:class:`KernelIR` is the single envelope. It carries the kernel
signature, return-tensor identity, schedule tree, and producer-consumer
dependency graph. :meth:`KernelIR.all_buffers` uses the dependency
sidecar's validated declarations for its owning tree, and scans a
rewrite's tree until its sidecar is rebuilt. :meth:`KernelIR.axis_extent`
derives axis information from the current tree.

:func:`build_initial_ir` runs dim unification, tree construction, and
dependency graph construction, then flattens the analysis output onto
a :class:`KernelIR` instance. :meth:`KernelIR.dump` writes the envelope
metadata and generated kernel into a cache directory.
"""

from __future__ import annotations

import pickle
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from nkigym.ir.canonical_build import build_canonical_blocknode_tree
from nkigym.ir.dependency import Dependency
from nkigym.ir.dimension_analysis import analyze_dimensions
from nkigym.ir.tree import Buffer, KernelTree


@dataclass
class KernelIR:
    """Envelope holding signature and the schedule tree.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_names: Identifiers in the kernel's ``return`` statement.
        tree: Canonical schedule tree.
        dependency: Producer-consumer graph derived from ``tree``.
        param_buffers: Parameter buffer metadata (shape/dtype/location).
    """

    func_name: str
    param_names: list[str]
    return_names: tuple[str, ...]
    tree: KernelTree
    dependency: Dependency
    param_buffers: dict[str, Buffer] = field(default_factory=dict)

    def __setstate__(self, payload: bytes | dict[str, object]) -> None:
        """Restore ordinary state dictionaries and legacy serialized snapshots."""
        self.__dict__.update(pickle.loads(payload) if isinstance(payload, bytes) else payload)
        self.__dict__.pop("_pickle_cache", None)

    @property
    def return_name(self) -> str:
        """Return the sole output name for transforms limited to one output."""
        if len(self.return_names) != 1:
            raise ValueError(f"{self.func_name} has {len(self.return_names)} outputs; one output is required")
        return self.return_names[0]

    def all_buffers(self) -> dict[str, Buffer]:
        """Return parameters and the current tree's validated declaration snapshot."""
        buffers = getattr(self.dependency, "_buffers", None) if self.dependency._tree is self.tree else None
        buffers = self.dependency._buffer_map(self.tree) if buffers is None else buffers
        if self.param_buffers.keys() & buffers.keys():
            raise ValueError("a parameter buffer is also declared in block.alloc_buffers")
        return self.param_buffers | buffers

    def buffer(self, name: str) -> Buffer:
        """Resolve a buffer by name; raises :class:`KeyError` if absent."""
        return self.all_buffers()[name]

    def axis_extent(self, axis: str) -> int:
        """Return the extent of the iter_var named ``axis``.

        Walks blocks in pre-order; returns the first ``IterVar`` whose
        ``axis`` matches. Raises :class:`KeyError` if the axis is not
        declared anywhere in the tree.
        """
        for nid in self.tree.blocks():
            for iv in self.tree.block(nid).iter_vars:
                if iv.axis == axis:
                    return iv.dom[1] - iv.dom[0]
        raise KeyError(f"no iter_var with axis {axis!r}")

    def dump(self, cache_dir: str | Path) -> None:
        """Write ``envelope.md`` and a black-formatted ``kernel.py`` into ``cache_dir``."""
        from nkigym.codegen import render

        (cache_path := Path(cache_dir)).mkdir(parents=True, exist_ok=True)
        (cache_path / "envelope.md").write_text(self._render_envelope_md(), encoding="utf-8")
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
            f"- **Returns**: {', '.join(f'`{name}`' for name in self.return_names)}",
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


def build_initial_ir(func: Callable[..., object], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> KernelIR:
    """Run dim analysis, build the schedule tree, derive the dependency graph, flatten.

    Args:
        func: An ``@nkigym_kernel``-decorated callable.
        input_specs: ``{param_name: (shape, dtype)}`` for every positional param.

    Returns:
        A populated :class:`KernelIR` envelope.
    """
    analysis = analyze_dimensions(func, input_specs)
    tree = build_canonical_blocknode_tree(analysis)
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
        return_names=analysis.return_names,
        tree=tree,
        dependency=Dependency(tree),
        param_buffers=param_buffers,
    )
