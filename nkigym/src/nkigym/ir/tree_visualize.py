"""Mermaid visualization for :class:`nkigym.ir.tree.KernelTree`.

:func:`dump_tree` writes ``tree.mmd`` and a rendered ``tree.png`` into a
caller-supplied cache directory. The ``.png`` is rendered at
``mmdc -s 4`` with ``--no-sandbox`` (required on gym hosts under
AppArmor).
"""

from __future__ import annotations

from pathlib import Path

from nkigym.ir._mermaid import ClassStyle, Flowchart, render_png
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree, NodeData

_FLOWCHART_STYLES: list[ClassStyle] = [
    ClassStyle(name="loop", fill="#eef", stroke="#336"),
    ClassStyle(name="tensorize", fill="#ffe", stroke="#a60"),
    ClassStyle(name="leaf", fill="#efe", stroke="#363"),
    ClassStyle(name="block", fill="#ffd", stroke="#960"),
]


def dump_tree(tree: KernelTree, cache_dir: str | Path) -> None:
    """Write ``tree.mmd`` and ``tree.png`` for ``tree`` into ``cache_dir``."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    mmd_path = cache_path / "tree.mmd"
    png_path = cache_path / "tree.png"
    mmd_path.write_text(_to_mermaid(tree), encoding="utf-8")
    render_png(mmd_path, png_path)


def _to_mermaid(tree: KernelTree) -> str:
    """Render ``tree`` to a Mermaid ``flowchart TB`` source string."""
    flow = Flowchart(direction="TB", styles=_FLOWCHART_STYLES)
    for nid in tree.preorder():
        node_id = f"n{nid}"
        decl, class_name = _tree_node_decl(node_id, nid, tree.data(nid))
        flow.add_node(node_id, decl, class_name)
        for child in tree.children(nid):
            flow.add_edge(node_id, f"n{child}")
    return flow.render()


def _tree_node_decl(node_id: str, nid: int, data: NodeData) -> tuple[str, str | None]:
    """Return the Mermaid declaration + CSS class bucket for one tree node.

    Content comes from ``data.label()``; this function owns only the
    tree-position concerns: the ``#nid`` prefix, the node shape
    (``[[...]]`` for blocks, ``[...]`` otherwise), and the CSS bucket.
    """
    text = f"#{nid} {_mermaid_escape(data.label())}"
    if isinstance(data, BlockNode):
        decl, class_name = f'{node_id}[["{text}"]]', "block"
    elif isinstance(data, ForNode):
        decl, class_name = f'{node_id}["{text}"]', "loop"
    elif isinstance(data, ISANode):
        decl, class_name = f'{node_id}["{text}"]', "leaf"
    else:
        raise TypeError(f"unknown node data type: {type(data).__name__}")
    return (decl, class_name)


def _mermaid_escape(text: str) -> str:
    """Make a label safe inside a Mermaid node string.

    Only newlines need handling — they become ``<br/>`` line breaks.
    Square brackets are left literal: inside the quoted node string
    (``["..."]`` / ``[["..."]]``) Mermaid treats them as text, so the
    region offsets render as ``lhs_T[...]``. Entity-encoding them as
    ``&#91;``/``&#93;`` made Mermaid leak a stray ``&`` into the label.
    """
    return text.replace("\n", "<br/>")


__all__ = ["dump_tree"]
