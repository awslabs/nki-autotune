"""Shared Mermaid helpers: ``Flowchart`` builder + ``mmdc`` PNG renderer.

:class:`Flowchart` accumulates node declarations, edges, and per-node
CSS-class memberships, then emits a ``flowchart`` source string. It
encapsulates the shared envelope used by :mod:`nkigym.ir.tree` and
:mod:`nkigym.ir.dependency` — both lay out nodes + edges and tag each
node with a visual class.

:func:`render_png` invokes ``mmdc`` to turn a ``.mmd`` file into a
high-resolution PNG. Gym hosts run Chromium under AppArmor, so the
helper writes a puppeteer config with ``--no-sandbox`` and deletes it
in a ``finally`` block.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassStyle:
    """CSS-class style for one bucket of flowchart nodes.

    Attributes:
        name: Bucket / class name (e.g. ``"alloc"``).
        fill: ``fill`` color (e.g. ``"#fef"``).
        stroke: ``stroke`` color (e.g. ``"#963"``).
    """

    name: str
    fill: str
    stroke: str


@dataclass
class Flowchart:
    """Accumulator for a Mermaid flowchart source string.

    Attributes:
        direction: Mermaid direction token (e.g. ``"TB"``, ``"LR"``).
        styles: Ordered CSS styles; controls declaration order so that
            classes can reference them by name.
    """

    direction: str
    styles: list[ClassStyle]
    _decls: list[str] = field(default_factory=list, init=False)
    _edges: list[str] = field(default_factory=list, init=False)
    _classes: dict[str, list[str]] = field(default_factory=dict, init=False)

    def add_node(self, node_id: str, decl: str, class_name: str | None = None) -> None:
        """Append a node declaration and optionally tag it with a class."""
        self._decls.append(f"    {decl}")
        if class_name is not None:
            self._classes.setdefault(class_name, []).append(node_id)

    def add_edge(self, src: str, dst: str, label: str | None = None) -> None:
        """Append an edge ``src -> dst`` with an optional edge label."""
        if label is None:
            self._edges.append(f"    {src} --> {dst}")
        else:
            self._edges.append(f"    {src} -->|{label}| {dst}")

    def render(self) -> str:
        """Return the complete ``flowchart <direction> ...`` source string."""
        lines: list[str] = [f"flowchart {self.direction}", *self._decls, "", *self._edges, ""]
        for style in self.styles:
            lines.append(f"    classDef {style.name} fill:{style.fill},stroke:{style.stroke};")
        for style in self.styles:
            members = self._classes.get(style.name)
            if members:
                lines.append(f"    class {','.join(members)} {style.name};")
        return "\n".join(lines) + "\n"


def render_png(mmd: Path, png: Path) -> None:
    """Render ``mmd`` to ``png`` at ``mmdc -s 4`` with ``--no-sandbox``."""
    config_path = mmd.with_suffix(".puppeteer.json")
    config_path.write_text('{"args":["--no-sandbox"]}', encoding="utf-8")
    try:
        result = subprocess.run(
            ["mmdc", "-i", str(mmd), "-o", str(png), "-s", "4", "--puppeteerConfigFile", str(config_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mmdc failed (exit {result.returncode}):\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
    finally:
        config_path.unlink(missing_ok=True)


__all__ = ["ClassStyle", "Flowchart", "render_png"]
