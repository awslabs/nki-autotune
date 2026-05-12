"""``ComputeAt`` atom — TVM ``sch.compute_at(block, loop)`` equivalent.

Moves an :class:`SBlock` under a target :class:`ForNode`, **preserving the
block's inner loop chain** (key change from v1 which regenerated from
``dim_role``).

Legality (spec §4.4):

- ``target_path`` resolves to a :class:`ForNode`.
- ``block_path`` resolves to an :class:`SBlock`.
- Target is NOT an ancestor of the block's current position.
- Target's subtree contains at least one consumer of the block's writes.
- **Prefix-match**: target's ancestor iter-var chain (root → target,
  inclusive) is a prefix of the block's ``iter_vars`` list — matched by
  ``axis_id`` in order. Otherwise illegal; user must ``Reorder`` first.
- **Role-lattice**: for each matched iter-var pair, ``max(target_role,
  block_role)`` must not demote the target's existing role.

Apply:

1. Identify matched iter-var pairs (one per dim).
2. Role-promote matched target iter vars when block's role is stronger:
   allocate fresh iter var, replace target's binding, rewrite all
   ``BufferAccess`` references + ``SBlock.iter_vars`` entries.
3. Rewrite block subtree's ``BufferAccess.pattern`` + ``SBlock.iter_vars``
   entries: block's iter var → target's iter var for each matched dim.
4. Block's remaining iter vars (unmatched suffix) stay — they become
   :class:`ForNode`s nested between target and block in the block's
   canonical iter-var order.
5. Remove block from old location; append the rebuilt subtree under
   target.
6. Canonical names are re-assigned by the render pipeline.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import (
    AccessRange,
    BufferAccess,
    ForNode,
    IterVar,
    KernelModule,
    SBlock,
    TreeIR,
    blocks_under,
    resolve_node,
)
from nkigym.ops.base import AxisRole
from nkigym.tune import AtomLegalityError

_ROLE_RANK: dict[AxisRole, int] = {AxisRole.PARALLEL: 0, AxisRole.SEQUENTIAL: 1, AxisRole.ACCUMULATION: 2}
"""Role lattice ``PAR ⊂ SEQ ⊂ ACC``. ComputeAt promotes to max."""


@dataclass(frozen=True)
class ComputeAt:
    """Move ``block_path`` under ``target_path`` with prefix-match + role promotion.

    Attributes:
        block_path: Path to the SBlock to move.
        target_path: Path to the ForNode under which the block will be placed.
    """

    block_path: tuple[int, ...]
    target_path: tuple[int, ...]

    def is_legal(self, module: KernelModule) -> bool:
        """Check structural + dataflow + prefix-match + role preconditions."""
        result: bool
        block = resolve_node(module.body, self.block_path)
        target = resolve_node(module.body, self.target_path)
        if not isinstance(block, SBlock):
            result = False
        elif not isinstance(target, ForNode):
            result = False
        elif _is_ancestor(self.target_path, self.block_path):
            result = False
        elif not _target_subtree_consumes_block_writes(module, target, block):
            result = False
        else:
            target_ancestor_ivs = _target_ancestor_iter_vars(module.body, self.target_path)
            matched = _match_prefix(target_ancestor_ivs, block.iter_vars)
            if matched is None:
                result = False
            else:
                result = _role_promotion_allowed(matched)
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Execute the ComputeAt.

        Raises:
            AtomLegalityError: ``is_legal`` returns False against ``module``.
        """
        if not self.is_legal(module):
            raise AtomLegalityError(f"ComputeAt.apply: illegal {self!r}")
        block = resolve_node(module.body, self.block_path)
        assert isinstance(block, SBlock)
        target_node = resolve_node(module.body, self.target_path)
        assert isinstance(target_node, ForNode)

        target_ancestor_ivs = _target_ancestor_iter_vars(module.body, self.target_path)
        matched = _match_prefix(target_ancestor_ivs, block.iter_vars)
        assert matched is not None

        """Step 1: role promotion. For each matched pair where block's role
        is stronger, allocate a fresh iter var with the stronger role and
        rewrite the target ForNode + every BufferAccess + every SBlock.iter_vars
        reference across the whole forest."""
        new_body = module.body
        id_replacements: dict[int, IterVar] = {}
        for target_iv, block_iv in matched:
            if _ROLE_RANK[block_iv.role] > _ROLE_RANK[target_iv.role]:
                promoted = module.allocate_iter_var(
                    axis_id=target_iv.axis_id, extent=target_iv.extent, role=block_iv.role
                )
                id_replacements[target_iv.var_id] = promoted
        if id_replacements:
            new_body = _rewrite_iter_var_ids(new_body, id_replacements)

        """Step 2: merge matched block iter vars into target iter vars. For
        each matched pair, rewrite block_iv.var_id → target_iv.var_id (using
        the promoted id when applicable) in the block subtree's BufferAccess
        entries + SBlock.iter_vars lists."""
        merge_map: dict[int, IterVar] = {}
        for target_iv, block_iv in matched:
            merged_target = id_replacements.get(target_iv.var_id, target_iv)
            merge_map[block_iv.var_id] = merged_target

        """Re-resolve the block and target in new_body after potential
        iter-var rewriting above. Paths are unchanged because rewriting
        only rebuilt nodes in place."""
        block_in_new = resolve_node(new_body, self.block_path)
        assert isinstance(block_in_new, SBlock)

        remaining_block_ivs = [iv for iv in block_in_new.iter_vars if iv.var_id not in merge_map]
        new_block = _rewrite_block_refs(block_in_new, merge_map, remaining_block_ivs)

        """Step 3: build the new subtree. Wrap the rewritten block in one
        ForNode per remaining block iter var (outermost first)."""
        subtree: ForNode | SBlock = new_block
        for iv in reversed(remaining_block_ivs):
            subtree = ForNode(iter_var=iv, children=[subtree])

        """Step 4: remove block from its old location; append subtree
        under the target. Use id() to re-resolve the target since removal
        may shift the target_path."""
        target_id = id(resolve_node(new_body, self.target_path))
        body_without = _remove_at_path(new_body, self.block_path)
        new_target_path = _find_node_path(body_without, target_id)
        if new_target_path is None:
            raise ValueError(
                f"ComputeAt.apply: target ForNode was consumed by removal — "
                f"block_path={self.block_path}, target_path={self.target_path}"
            )
        final_body = _prepend_under(body_without, new_target_path, subtree)
        return replace(module, body=final_body)


def enumerate_compute_at_atoms(module: KernelModule) -> list[ComputeAt]:
    """Emit every legal ``(block, target)`` pair across the forest.

    Alloc blocks are skipped — they carry no iter vars and placing them
    under a loop has no semantic meaning.
    """
    blocks: list[tuple[tuple[int, ...], SBlock]] = []
    fornodes: list[tuple[tuple[int, ...], ForNode]] = []

    def collect(node: ForNode | SBlock, path: tuple[int, ...]) -> None:
        """Walk ``node`` collecting paths to every SBlock + ForNode."""
        if isinstance(node, SBlock):
            blocks.append((path, node))
        else:
            fornodes.append((path, node))
            for i, c in enumerate(node.children):
                collect(c, path + (i,))

    for i, root in enumerate(module.body):
        collect(root, (i,))

    atoms: list[ComputeAt] = []
    for block_path, block in blocks:
        """Skip alloc blocks — empty iter_vars + zero-trip compute."""
        if not block.iter_vars:
            continue
        for target_path, _target in fornodes:
            atom = ComputeAt(block_path=block_path, target_path=target_path)
            if atom.is_legal(module):
                atoms.append(atom)
    return atoms


def _is_ancestor(maybe_ancestor: tuple[int, ...], descendant: tuple[int, ...]) -> bool:
    """Return True iff ``maybe_ancestor`` is a strict prefix of ``descendant``."""
    return len(maybe_ancestor) < len(descendant) and descendant[: len(maybe_ancestor)] == maybe_ancestor


def _target_ancestor_iter_vars(body: TreeIR, target_path: tuple[int, ...]) -> list[IterVar]:
    """Collect iter vars along the path from forest root to ``target_path``, inclusive.

    Walks each ForNode along the path and appends its iter var to the
    result. Returns an empty list if ``target_path`` is empty.
    """
    result: list[IterVar] = []
    siblings: list[ForNode | SBlock] = list(body)
    for idx in target_path:
        if idx >= len(siblings):
            break
        node = siblings[idx]
        if isinstance(node, ForNode):
            result.append(node.iter_var)
            siblings = node.children
        else:
            break
    return result


def _match_prefix(
    target_ancestor_ivs: list[IterVar], block_iter_vars: list[IterVar]
) -> list[tuple[IterVar, IterVar]] | None:
    """Return matched ``(target_iv, block_iv)`` pairs for the prefix.

    The target's ancestor chain (outermost → innermost) must match a
    prefix of the block's ``iter_vars`` list by ``axis_id`` position-wise.
    Returns ``None`` if any axis mismatches or the target chain is longer
    than the block's iter-var list.
    """
    result: list[tuple[IterVar, IterVar]] | None = None
    if len(target_ancestor_ivs) <= len(block_iter_vars):
        ok = True
        pairs: list[tuple[IterVar, IterVar]] = []
        for t_iv, b_iv in zip(target_ancestor_ivs, block_iter_vars):
            if t_iv.axis_id != b_iv.axis_id:
                ok = False
                break
            pairs.append((t_iv, b_iv))
        if ok:
            result = pairs
    return result


def _role_promotion_allowed(matched: list[tuple[IterVar, IterVar]]) -> bool:
    """Spec role-lattice rule.

    ``max(target_role, block_role) >= target_role`` is automatically true
    (max of a set always dominates each member). The spec forbids
    "demote" — i.e. the promoted role cannot be weaker than what target
    already has. Since max never demotes, every matched pair is allowed.
    Kept as a separate predicate for clarity + future constraint tweaks.
    """
    _ = matched
    return True


def _target_subtree_consumes_block_writes(module: KernelModule, target: ForNode, block: SBlock) -> bool:
    """Return True iff any SBlock in ``target``'s subtree reads a tensor
    that ``block`` writes.

    Both ``writes`` and ``reads_writes`` count as block-produced values;
    both ``reads`` and ``reads_writes`` count as target-side consumption.
    """
    _ = module
    written = {a.tensor_name for a in block.writes.values()} | {a.tensor_name for a in block.reads_writes.values()}
    result = False
    for descendant in blocks_under(target):
        read_names = {a.tensor_name for a in descendant.reads.values()} | {
            a.tensor_name for a in descendant.reads_writes.values()
        }
        if written & read_names:
            result = True
            break
    return result


def _target_subtree_produces_block_reads(module: KernelModule, target: ForNode, block: SBlock) -> bool:
    """Return True iff any SBlock in ``target``'s subtree writes a tensor
    that ``block`` reads.

    Dual of :func:`_target_subtree_consumes_block_writes`: used by
    :class:`~nkigym.tune.reverse_compute_at.ReverseComputeAt` whose legality
    requires the target's subtree to contain a producer of one of the
    block's reads. Both ``reads`` and ``reads_writes`` count as block-side
    consumption; both ``writes`` and ``reads_writes`` count as
    target-side production.
    """
    _ = module
    read = {a.tensor_name for a in block.reads.values()} | {a.tensor_name for a in block.reads_writes.values()}
    result = False
    for descendant in blocks_under(target):
        write_names = {a.tensor_name for a in descendant.writes.values()} | {
            a.tensor_name for a in descendant.reads_writes.values()
        }
        if read & write_names:
            result = True
            break
    return result


def _rewrite_iter_var_ids(body: TreeIR, id_replacements: dict[int, IterVar]) -> TreeIR:
    """Rewrite every IterVar reference in the forest whose ``var_id`` is
    in ``id_replacements``.

    Touches: ``ForNode.iter_var``, ``SBlock.iter_vars`` entries, and
    ``BufferAccess.iter_var_coeffs`` (coefficient keys). ``BufferAccess``
    is hashed on the ``iter_var_coeffs`` tuple, so the rewrite returns
    fresh :class:`AccessRange` instances with the new var_id keys.
    """

    def rewrite_access(acc: BufferAccess) -> BufferAccess:
        changed = False
        new_iv_ids_list: list[int] = []
        for iv_id in acc.iter_var_ids:
            if iv_id in id_replacements:
                new_iv_ids_list.append(id_replacements[iv_id].var_id)
                changed = True
            else:
                new_iv_ids_list.append(iv_id)
        new_pattern: list[AccessRange] = []
        for ar in acc.pattern:
            coeffs = ar.coeffs
            ar_changed = False
            new_coeffs: dict[int, int] = {}
            for iv_id, c in coeffs.items():
                if iv_id in id_replacements:
                    new_coeffs[id_replacements[iv_id].var_id] = c
                    ar_changed = True
                else:
                    new_coeffs[iv_id] = c
            if ar_changed:
                new_pattern.append(AccessRange.make(new_coeffs, ar.const_offset, ar.extent))
                changed = True
            else:
                new_pattern.append(ar)
        result = acc
        if changed:
            result = BufferAccess(
                tensor_name=acc.tensor_name, iter_var_ids=tuple(new_iv_ids_list), pattern=tuple(new_pattern)
            )
        return result

    def rewrite_block(block: SBlock) -> SBlock:
        new_ivs = [id_replacements.get(iv.var_id, iv) for iv in block.iter_vars]
        return SBlock(
            iter_vars=new_ivs,
            reads={k: rewrite_access(v) for k, v in block.reads.items()},
            writes={k: rewrite_access(v) for k, v in block.writes.items()},
            reads_writes={k: rewrite_access(v) for k, v in block.reads_writes.items()},
            body=block.body,
            annotations=dict(block.annotations),
        )

    def rewrite_node(node: ForNode | SBlock) -> ForNode | SBlock:
        if isinstance(node, SBlock):
            return rewrite_block(node)
        new_iv = id_replacements.get(node.iter_var.var_id, node.iter_var)
        return ForNode(
            iter_var=new_iv,
            children=[rewrite_node(c) for c in node.children],
            name=node.name,
            annotations=dict(node.annotations),
        )

    return [rewrite_node(n) for n in body]


def _rewrite_block_refs(block: SBlock, merge_map: dict[int, IterVar], remaining_block_ivs: list[IterVar]) -> SBlock:
    """Return a fresh :class:`SBlock` with iter-var references rewritten.

    - ``iter_vars`` set to the unmatched suffix (``remaining_block_ivs``).
    - Every :class:`BufferAccess` referencing a key of ``merge_map`` is
      rewritten: the coefficient on the block's iter var moves onto the
      target's iter var; the id list is updated accordingly.
    """

    def rewrite_access(acc: BufferAccess) -> BufferAccess:
        changed = False
        new_iv_ids_list: list[int] = []
        for iv_id in acc.iter_var_ids:
            if iv_id in merge_map:
                new_iv_ids_list.append(merge_map[iv_id].var_id)
                changed = True
            else:
                new_iv_ids_list.append(iv_id)
        new_pattern: list[AccessRange] = []
        for ar in acc.pattern:
            coeffs = ar.coeffs
            ar_changed = False
            new_coeffs: dict[int, int] = {}
            for iv_id, c in coeffs.items():
                if iv_id in merge_map:
                    target_id = merge_map[iv_id].var_id
                    new_coeffs[target_id] = new_coeffs.get(target_id, 0) + c
                    ar_changed = True
                else:
                    new_coeffs[iv_id] = new_coeffs.get(iv_id, 0) + c
            if ar_changed:
                new_pattern.append(AccessRange.make(new_coeffs, ar.const_offset, ar.extent))
                changed = True
            else:
                new_pattern.append(ar)
        result = acc
        if changed:
            """Dedupe new_iv_ids_list while preserving order."""
            seen: set[int] = set()
            deduped: list[int] = []
            for iv_id in new_iv_ids_list:
                if iv_id not in seen:
                    seen.add(iv_id)
                    deduped.append(iv_id)
            result = BufferAccess(tensor_name=acc.tensor_name, iter_var_ids=tuple(deduped), pattern=tuple(new_pattern))
        return result

    return SBlock(
        iter_vars=list(remaining_block_ivs),
        reads={k: rewrite_access(v) for k, v in block.reads.items()},
        writes={k: rewrite_access(v) for k, v in block.writes.items()},
        reads_writes={k: rewrite_access(v) for k, v in block.reads_writes.items()},
        body=block.body,
        annotations=dict(block.annotations),
    )


def _remove_at_path(body: TreeIR, path: tuple[int, ...]) -> TreeIR:
    """Return a new body with the node at ``path`` removed.

    Ancestor ForNodes whose children become empty are pruned recursively
    so the parent tree does not retain empty loops after removal.
    """
    if not path:
        raise ValueError("_remove_at_path: path must be non-empty")
    if len(path) == 1:
        return [*body[: path[0]], *body[path[0] + 1 :]]
    idx, rest = path[0], path[1:]
    parent = body[idx]
    assert isinstance(parent, ForNode)
    new_children = _remove_at_path(parent.children, rest)
    if not new_children:
        return [*body[:idx], *body[idx + 1 :]]
    new_parent = ForNode(
        iter_var=parent.iter_var, children=new_children, name=parent.name, annotations=dict(parent.annotations)
    )
    return [*body[:idx], new_parent, *body[idx + 1 :]]


def _append_under(body: TreeIR, target_path: tuple[int, ...], new_node: ForNode | SBlock) -> TreeIR:
    """Append ``new_node`` to the children of the ForNode at ``target_path``.

    ``ReverseComputeAt`` uses this to place a consumer AFTER the producer
    chain it was moved into (the consumer reads the producer's fresh
    output).

    If ``target_path`` is empty, ``new_node`` is appended at the forest
    root.
    """
    if not target_path:
        return [*body, new_node]
    idx, rest = target_path[0], target_path[1:]
    parent = body[idx]
    assert isinstance(parent, ForNode)
    if not rest:
        new_children = [*parent.children, new_node]
    else:
        new_children = _append_under(parent.children, rest, new_node)
    new_parent = ForNode(
        iter_var=parent.iter_var, children=new_children, name=parent.name, annotations=dict(parent.annotations)
    )
    return [*body[:idx], new_parent, *body[idx + 1 :]]


def _prepend_under(body: TreeIR, target_path: tuple[int, ...], new_node: ForNode | SBlock) -> TreeIR:
    """Prepend ``new_node`` to the children of the ForNode at ``target_path``.

    ``ComputeAt`` uses this to place a producer BEFORE the consumer chain
    it was moved into (the consumer must see the producer's output).

    If ``target_path`` is empty, ``new_node`` is prepended at the forest
    root.
    """
    if not target_path:
        return [new_node, *body]
    idx, rest = target_path[0], target_path[1:]
    parent = body[idx]
    assert isinstance(parent, ForNode)
    if not rest:
        new_children = [new_node, *parent.children]
    else:
        new_children = _prepend_under(parent.children, rest, new_node)
    new_parent = ForNode(
        iter_var=parent.iter_var, children=new_children, name=parent.name, annotations=dict(parent.annotations)
    )
    return [*body[:idx], new_parent, *body[idx + 1 :]]


def _find_node_path(body: TreeIR, target_id: int) -> tuple[int, ...] | None:
    """Return the path of the node whose ``id()`` matches ``target_id``, or None."""

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> tuple[int, ...] | None:
        """Recurse into ``node`` searching for the target id."""
        result: tuple[int, ...] | None = None
        if id(node) == target_id:
            result = path
        elif isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    result = r
                    break
        return result

    found: tuple[int, ...] | None = None
    for i, root in enumerate(body):
        r = walk(root, (i,))
        if r is not None:
            found = r
            break
    return found
