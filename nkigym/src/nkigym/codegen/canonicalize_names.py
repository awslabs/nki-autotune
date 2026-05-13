"""Canonical iter-var naming pass.

Assigns ``ForNode.name = "i_<axis_name>_<ordinal>"`` deterministically across
the schedule tree. Ordinals are per-tree and per-axis: sibling subtrees
see identical ordinal counts (counters are restored on ascent), but
nested same-axis ForNodes get increasing ordinals top-down (e.g. a Split
pair on ``d0`` renders as ``i_d0_0`` outside, ``i_d0_1`` inside).

The axis's display name is consulted via ``module.axes[iv.axis_id].name``;
identity remains ``axis_id`` so renames cannot change the tree's logic.
"""

from nkigym.ir.ir import ForNode, KernelIR, SBlock


def canonicalize_iter_var_names(module: KernelIR) -> None:
    """Assign canonical names to every ForNode in every tree.

    Args:
        module: KernelIR whose body tree will be named in place.
    """
    for root in module.body:
        _walk(root, module, counts={})


def _walk(node: ForNode | SBlock, module: KernelIR, counts: dict[int, int]) -> None:
    """Depth-first walker; assigns each ForNode a per-axis ordinal name.

    ``counts[axis_id]`` tracks how many same-axis ancestors the current
    node has. The node takes that as its ordinal, recurses with the
    counter incremented, then restores it so sibling subtrees start at
    the same count their parent saw.
    """
    if isinstance(node, SBlock):
        return
    axis_id = node.iter_var.axis_id
    axis_name = module.axes[axis_id].name
    k = counts.get(axis_id, 0)
    node.name = f"i_{axis_name}_{k}"
    counts[axis_id] = k + 1
    for child in node.children:
        _walk(child, module, counts)
    counts[axis_id] = k
