"""Canonical iter-var naming pass.

Assigns ``ForNode.name = "i_<dim>_<ordinal>"`` deterministically across
the schedule tree. Ordinals are per-tree and per-dim: sibling subtrees
see identical ordinal counts (counters are restored on ascent), but
nested same-dim ForNodes get increasing ordinals top-down (e.g. a Split
pair on ``d0`` renders as ``i_d0_0`` outside, ``i_d0_1`` inside).

Matches v1's canonical naming behavior. Runs after Task 13's
``place_buffers`` and before ``emit_source`` in the render pipeline.
"""

from nkigym.codegen.ir import ForNode, KernelModule, SBlock


def canonicalize_iter_var_names(module: KernelModule) -> None:
    """Assign canonical names to every ForNode in every tree.

    Args:
        module: KernelModule whose body tree will be named in place.
    """
    for root in module.body:
        _walk(root, counts={})


def _walk(node: ForNode | SBlock, counts: dict[str, int]) -> None:
    """Depth-first walker; assigns each ForNode a per-dim ordinal name.

    ``counts[dim]`` tracks how many same-dim ancestors the current node
    has. The node takes that as its ordinal, recurses with the counter
    incremented, then restores it so sibling subtrees start at the same
    count their parent saw.
    """
    if isinstance(node, SBlock):
        return
    dim = node.iter_var.dim_id
    k = counts.get(dim, 0)
    node.name = f"i_{dim}_{k}"
    counts[dim] = k + 1
    for child in node.children:
        _walk(child, counts)
    counts[dim] = k
