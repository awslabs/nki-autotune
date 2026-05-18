"""Mermaid visualization for :class:`nkigym.ir.tree.KernelTree`.

:func:`dump_tree` writes ``tree.mmd`` and a rendered ``tree.png`` into a
caller-supplied cache directory. The ``.png`` is rendered at
``mmdc -s 4`` with ``--no-sandbox`` (required on gym hosts under
AppArmor).
"""

from __future__ import annotations

from pathlib import Path

from nkigym.ir._mermaid import ClassStyle, Flowchart, render_png
from nkigym.ir.tree import ForNode, ISANode, KernelTree, NodeData, RootNode
from nkigym.ops.alloc import NKIAlloc

_FLOWCHART_STYLES: list[ClassStyle] = [
    ClassStyle(name="alloc", fill="#fef", stroke="#963"),
    ClassStyle(name="loop", fill="#eef", stroke="#336"),
    ClassStyle(name="tensorize", fill="#ffe", stroke="#a60"),
    ClassStyle(name="leaf", fill="#efe", stroke="#363"),
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
    """Return the Mermaid declaration + class bucket for one tree node."""
    if isinstance(data, RootNode):
        return f'{node_id}(("#{nid} root"))', None
    if isinstance(data, ForNode):
        return (f'{node_id}["#{nid} Loop {data.dim} trip={data.trip}"]', "loop")
    if isinstance(data, ISANode):
        return (f'{node_id}["#{nid} {_isa_label(data)}"]', "alloc" if data.op_cls is NKIAlloc else "leaf")
    raise TypeError(f"unknown node data type: {type(data).__name__}")


def _isa_label(data: ISANode) -> str:
    """Build the Mermaid node label for an :class:`ISANode` payload."""
    parts: list[str] = [data.op_cls.__name__]
    if data.reads:
        parts.append(f"reads={','.join(data.reads)}")
    if data.writes:
        parts.append(f"writes={','.join(data.writes)}")
    if data.rmw:
        parts.append(f"rmw={','.join(data.rmw)}")
    parts.append(f"tensorize_sizes={data.tensorize_sizes}")
    parts.append(f"axis_map={data.axis_map}")
    parts.append(f"kwargs={data.kwargs}")
    return "<br/>".join(parts)


__all__ = ["dump_tree"]
