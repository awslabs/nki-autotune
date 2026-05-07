"""Mermaid `graph TD` serialiser for :class:`LoopForest`.

Produces a verbose, faithful rendering of the forest IR suitable for
auto-dumps and debugging. Labels include every meaningful
``LoopNode`` / ``BodyLeaf`` field; edge labels are the child index so
``path`` tuples can be read off the picture.
"""

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import LoopForest


def dump_forest_mermaid(forest: LoopForest, op_graph: OpGraph) -> str:
    """Return Mermaid `graph TD` source for ``forest``.

    Args:
        forest: The forest to serialise.
        op_graph: Used to resolve ``BodyLeaf.op_idx`` into op-class names.

    Returns:
        Mermaid source — starts with ``graph TD``. An empty forest
        produces exactly that single line (no nodes, no edges).
    """
    lines: list[str] = ["graph TD"]
    for root_idx, root in enumerate(forest):
        _emit(lines, root, path=(root_idx,), op_graph=op_graph, parent_id=None, child_idx=None)
    return "\n".join(lines)


def _emit(
    lines: list[str], node, path: tuple[int, ...], op_graph: OpGraph, parent_id: str | None, child_idx: int | None
) -> None:
    """Recursively emit node + outgoing edge, depth-first."""
    from nkigym.codegen.loop_forest import LoopNode

    node_id = _node_id(node, path)
    label = _node_label(node, path, op_graph)
    if isinstance(node, LoopNode):
        lines.append(f'{node_id}["{label}"]')
    else:
        lines.append(f'{node_id}(["{label}"])')
    if parent_id is not None:
        lines.append(f"{parent_id} -- {child_idx} --> {node_id}")
    if isinstance(node, LoopNode):
        for i, child in enumerate(node.children):
            _emit(lines, child, path=path + (i,), op_graph=op_graph, parent_id=node_id, child_idx=i)


def _node_id(node, path: tuple[int, ...]) -> str:
    """Mermaid node id derived from path. L_/B_ prefix + underscore-joined path."""
    from nkigym.codegen.loop_forest import LoopNode

    prefix = "L" if isinstance(node, LoopNode) else "B"
    suffix = "_".join(str(p) for p in path)
    return f"{prefix}_{suffix}"


def _node_label(node, path: tuple[int, ...], op_graph: OpGraph) -> str:
    """Human-readable label — multi-line, fields joined with <br/>."""
    from nkigym.codegen.loop_forest import BodyLeaf, LoopNode

    path_str = ", ".join(str(p) for p in path)
    if len(path) == 1:
        path_str += ","

    if isinstance(node, BodyLeaf):
        op_name = getattr(op_graph.ops[node.op_idx].op_cls, "__name__")
        return f"B({path_str})<br/>op={op_name} phase={node.phase}"
    assert isinstance(node, LoopNode)
    parts = [f"L({path_str})", f"dim={node.dim_id} trip={node.trip_count}", f"role={node.role.name} name={node.name}"]
    if node.reduce_op is not None:
        parts.append(f"reduce_op={node.reduce_op}")
    return "<br/>".join(parts)
