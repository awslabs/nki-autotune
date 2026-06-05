"""Region-regen core for ComputeAt / ReverseComputeAt.

A move relocates one block under a target loop. These pure functions
derive, from the affine ``iter_values`` of both the moved block and the
target's enclosing nest, which dims the target covers and what residual
domain each moved iter-var must sweep, then regenerate residual ForNodes
and rebind the moved block's regions. Works on tiled (affine) bindings,
not only bare-Var ones.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from nkigym.ir.arith.expr import Expr, from_affine, to_affine
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _block_local_descendants


def _loopvar_to_dim(tree: KernelTree, block_nid: int) -> dict[str, str]:
    """Map each loop_var the block binds to its concrete dim, via iter_values.

    A loop_var binds the iter_var whose iter_value affine mentions it
    (iter_values are affine over a single dim's loops). Works for tiled
    bindings (``i_d1_0*512 + i_d1_1*128``), not just bare Vars.
    """
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)
    out: dict[str, str] = {}
    for iv, value in zip(block.iter_vars, block.iter_values):
        for name in to_affine(value):
            if name is not None:
                out[name] = iv.axis
    return out


def dim_loops_of_block(tree: KernelTree, block_nid: int) -> dict[str, list[tuple[str, int]]]:
    """Map each concrete dim to the ForNodes driving it as ``(loop_var, extent)`` outer→inner.

    A dim's loops are the block's ENCLOSING ForNodes on that dim (loops a prior
    ComputeAt nested the block under — outer) followed by the block's own local
    ForNodes (inner). An already-sunk block owns no local loops on its covered
    dims; those dims are driven entirely by enclosing loops, so re-moving it
    still sees its full iteration domain. For a top-level block the enclosing
    list is empty and the result is exactly the block-local loops.

    The enclosing gather spans ALL ancestor ForNodes, crossing BlockNode walls:
    a block can be nested several blocks deep (a load co-located under the
    matmul, then under another producer), with a dim it binds driven by a loop
    above an intervening block. Filtering by ``lv_to_dim`` (the loop vars the
    block actually binds) keeps unrelated ancestor loops out, so only the
    block's own driving loops contribute — restricting to the block-local chain
    instead would drop a cross-wall driver and collapse the dim to a constant.
    """
    lv_to_dim = _loopvar_to_dim(tree, block_nid)
    out: dict[str, list[tuple[str, int]]] = {}
    for loop_var, extent in _all_enclosing_loops_of_block(tree, block_nid):
        if loop_var in lv_to_dim:
            out.setdefault(lv_to_dim[loop_var], []).append((loop_var, extent))
    for nid in _block_local_descendants(tree, block_nid):
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in lv_to_dim:
            out.setdefault(lv_to_dim[data.loop_var], []).append((data.loop_var, data.extent))
    return out


def _all_enclosing_loops_of_block(tree: KernelTree, block_nid: int) -> list[tuple[str, int]]:
    """Every ForNode ancestor of ``block_nid`` as ``(loop_var, extent)``, outer→inner.

    Spans the whole ancestor chain (crossing BlockNode boundaries), unlike a
    block-local walk that resets at each enclosing BlockNode. Callers filter by
    the block's bound loop vars, so loops owned by an enclosing block that the
    block does not index are dropped anyway; what this preserves is a driver of
    a dim the block DOES bind that happens to sit above an intervening block.
    """
    return [
        (tree.data(anc).loop_var, tree.data(anc).extent)
        for anc in tree.ancestors(block_nid)
        if isinstance(tree.data(anc), ForNode)
    ]


def enclosing_dim_loops(tree: KernelTree, target_loop_nid: int) -> dict[str, list[tuple[str, int]]]:
    """Map each concrete dim to the ForNodes at/above ``target_loop_nid`` within its block.

    Walks ``[target_loop_nid, *ancestors]`` up to (not into) the enclosing
    block, reading that block's loopvar→dim map. Outer→inner order.
    """
    block_nid = _enclosing_block(tree, target_loop_nid)
    lv_to_dim = _loopvar_to_dim(tree, block_nid)
    chain = [target_loop_nid, *reversed(tree.ancestors(target_loop_nid))]
    out: dict[str, list[tuple[str, int]]] = {}
    for nid in chain:
        data = tree.data(nid)
        if not isinstance(data, ForNode):
            continue
        """A loop on the chain may belong to a DIFFERENT (enclosing) block than
        the target's — e.g. the target sits in a producer's sub-block nested
        under the matmul's M loop. That loop's var is absent from this block's
        loopvar map, so fall back to parsing its dim from the dense name; the
        coverage solve must still see it tiling its dim (else a full-extent
        writer sunk under the target is treated as a free residual and replicated
        inside the foreign tiling loop -> clobber)."""
        dim = lv_to_dim.get(data.loop_var) or _dim_from_loopvar(data.loop_var)
        out.setdefault(dim, []).insert(0, (data.loop_var, data.extent))
    return out


def _dim_from_loopvar(loop_var: str) -> str:
    """``i_d1_0`` / ``i_d1_0_0`` -> ``d1``. Strip the ``i_`` prefix and trailing ``_<int>``."""
    body = loop_var[2:] if loop_var.startswith("i_") else loop_var
    return body.split("_")[0]


def _enclosing_block(tree: KernelTree, nid: int) -> int:
    """Return the nearest BlockNode ancestor of ``nid``."""
    result: int | None = None
    for anc in reversed(tree.ancestors(nid)):
        if isinstance(tree.data(anc), BlockNode):
            result = anc
            break
    if result is None:
        raise ValueError(f"no enclosing BlockNode for {nid}")
    return result


class DomainSolveError(ValueError):
    """Raised when a target's coverage does not cleanly divide a moved dim."""


@dataclass(frozen=True)
class DimDomain:
    """How one moved-block dim is re-domained under the target.

    Attributes:
        target_loops: the target's enclosing ``(loop_var, extent)`` on this
            dim that the moved dim binds to (empty if the target doesn't
            iterate the dim).
        residual_extent: trip count of the residual loop regenerated below
            the insertion point (1 = fully covered, no residual loop).
    """

    target_loops: list[tuple[str, int]]
    residual_extent: int


def solve_iter_domains(
    moved: dict[str, list[tuple[str, int]]], target: dict[str, list[tuple[str, int]]]
) -> dict[str, DimDomain]:
    """Per moved dim, split its iteration into target-covered + residual.

    ``moved`` / ``target`` are ``dim_loops_of_block`` / ``enclosing_dim_loops``
    outputs. For each moved dim, ``moved_product`` is the product of its
    trips; ``target_product`` the product of the target's trips on that dim
    (1 if absent). Requires ``target_product`` to divide ``moved_product``;
    residual = ``moved_product // target_product``.
    """
    out: dict[str, DimDomain] = {}
    for dim, loops in moved.items():
        moved_product = prod(e for _v, e in loops)
        target_loops = target.get(dim, [])
        target_product = prod(e for _v, e in target_loops)
        if moved_product % target_product != 0:
            raise DomainSolveError(
                f"dim {dim}: target coverage {target_product} does not divide moved extent {moved_product}"
            )
        out[dim] = DimDomain(target_loops=target_loops, residual_extent=moved_product // target_product)
    return out


def regen_and_rebind(tree: KernelTree, block_nid: int, solved: dict[str, DimDomain]) -> None:
    """Drop the moved block's ForNodes; regenerate residual loops; rebind iter_values.

    After this, the block's body is reached through one residual ForNode per
    dim with ``residual_extent > 1`` (chained outer→inner in iter_vars order),
    and ``iter_values`` bind each dim's iter_var to the affine over its target
    loops (covered) plus its residual loop var. The block's reads/writes and
    leaf operand_bindings keep their tensor structure; ``normalize_block``
    (called by the caller after splice) recomputes the region ``lo`` offsets
    from the surviving loops. This function only fixes loop topology + the
    iter_values skeleton.
    """
    block = tree.data(block_nid)
    assert isinstance(block, BlockNode)

    """Detach the block's body leaf (single ISA leaf) from its current parent.

    The leaf's parent may be a local ForNode (later stripped) OR the block
    itself when a prior move collapsed all of the block's local loops (the leaf
    is then edged directly from ``block_nid``). Removing that incoming edge here
    guarantees the leaf is parentless before the regenerated residual chain (or
    the block) re-attaches it; otherwise the direct ``block_nid -> leaf`` edge
    survives ``_strip_block_loops`` and the leaf ends up double-parented.
    """
    body_leaf = _single_body_leaf(tree, block_nid)
    for pred in list(tree.graph.predecessors(body_leaf)):
        tree.graph.remove_edge(pred, body_leaf)
    _strip_block_loops(tree, block_nid)

    """Regenerate residual ForNodes (one per dim with residual_extent > 1), chained."""
    residual_vars: dict[str, str] = {}
    parent = block_nid
    for iv in block.iter_vars:
        dom = solved.get(iv.axis)
        if dom is None or dom.residual_extent <= 1:
            continue
        loop_var = f"i_{iv.axis}__resid"
        new_for = tree.add_node(ForNode(loop_var=loop_var, extent=dom.residual_extent), parent=parent)
        residual_vars[iv.axis] = loop_var
        parent = new_for
    tree.graph.add_edge(parent, body_leaf)

    """Rebuild iter_values: covered dims -> affine over target loops; residual -> its loop var;
    both -> sum. normalize_block recomputes region lo's from these."""
    new_values: list[Expr] = []
    for iv in block.iter_vars:
        dom = solved.get(iv.axis)
        new_values.append(_dim_binding(dom, residual_vars.get(iv.axis)))
    tree.graph.nodes[block_nid]["data"] = replace(block, iter_values=tuple(new_values))


def _dim_binding(dom: DimDomain | None, residual_var: str | None) -> Expr:
    """Affine binding for one dim: target-loop affine + residual-loop term.

    target loops ``l_0(t_0)..l_{k-1}(t_{k-1})`` contribute ``Σ l_j * (Π
    inner-target-trips * residual_extent)``; the residual loop contributes
    ``+ residual_var``. With no target loops and no residual the dim is
    loopless (``Const(0)``). The exact element scaling is re-derived by
    ``normalize_block``; here we only need every surviving loop var to appear
    so the loopvar→dim map is recoverable.
    """
    coeffs: dict[str | None, int] = {None: 0}
    if dom is not None:
        inner = dom.residual_extent
        for loop_var, extent in reversed(dom.target_loops):
            coeffs[loop_var] = inner
            inner *= extent
    if residual_var is not None:
        coeffs[residual_var] = 1
    return from_affine(coeffs)


def _single_body_leaf(tree: KernelTree, block_nid: int) -> int:
    """Return the one ISA leaf in the block's local scope."""
    leaves = [n for n in _block_local_descendants(tree, block_nid) if isinstance(tree.data(n), ISANode)]
    if len(leaves) != 1:
        raise DomainSolveError(f"block {block_nid} must have exactly one ISA leaf; got {len(leaves)}")
    return leaves[0]


def _strip_block_loops(tree: KernelTree, block_nid: int) -> None:
    """Remove every ForNode in the block's local scope, leaving the leaf detached."""
    for nid in _block_local_descendants(tree, block_nid):
        if isinstance(tree.data(nid), ForNode):
            tree.graph.remove_node(nid)


__all__ = [
    "dim_loops_of_block",
    "enclosing_dim_loops",
    "DomainSolveError",
    "DimDomain",
    "solve_iter_domains",
    "regen_and_rebind",
]
