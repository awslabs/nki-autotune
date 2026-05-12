"""``ReverseComputeAt`` atom — dual of :class:`~nkigym.tune.compute_at.ComputeAt`.

Dataflow direction flipped: target's subtree must contain a **producer**
of one of the block's reads (instead of a **consumer** of the block's
writes).

All other semantics (prefix-match legality, role-lattice promotion,
iter-var merging + subtree rebuild + pattern rewriting) are identical to
:class:`ComputeAt`; implementation is shared via module-level helpers in
``compute_at`` — see spec ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md`` §4.5.

Use case: move a consumer block into a producer's scope (e.g. move an
:class:`NKIStore` into the subtree where its source SBUF buffer gets
written by the previous op).
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import ForNode, IterVar, KernelModule, SBlock, resolve_node
from nkigym.tune import AtomLegalityError
from nkigym.tune.compute_at import (
    _ROLE_RANK,
    _append_under,
    _find_node_path,
    _is_ancestor,
    _match_prefix,
    _remove_at_path,
    _rewrite_block_refs,
    _rewrite_iter_var_ids,
    _role_promotion_allowed,
    _target_ancestor_iter_vars,
    _target_subtree_produces_block_reads,
)


@dataclass(frozen=True)
class ReverseComputeAt:
    """Move a consumer ``SBlock`` under a target ``ForNode`` in a producer's scope.

    Attributes:
        block_path: Path to the consumer SBlock to move.
        target_path: Path to the target ForNode. Target's subtree must
            contain at least one producer of one of the block's reads.
    """

    block_path: tuple[int, ...]
    target_path: tuple[int, ...]

    def is_legal(self, module: KernelModule) -> bool:
        """Check structural + dataflow + prefix-match + role preconditions.

        Identical to :meth:`ComputeAt.is_legal` except the dataflow
        direction is flipped (producer instead of consumer check).
        """
        result: bool
        block = resolve_node(module.body, self.block_path)
        target = resolve_node(module.body, self.target_path)
        if not isinstance(block, SBlock):
            result = False
        elif not isinstance(target, ForNode):
            result = False
        elif _is_ancestor(self.target_path, self.block_path):
            result = False
        elif not _target_subtree_produces_block_reads(module, target, block):
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
        """Execute the ReverseComputeAt.

        Apply logic (iter-var merging + subtree rebuild + pattern
        rewriting) is identical to :meth:`ComputeAt.apply`; only the
        ``is_legal`` direction differs.

        Raises:
            AtomLegalityError: ``is_legal`` returns False against ``module``.
        """
        if not self.is_legal(module):
            raise AtomLegalityError(f"ReverseComputeAt.apply: illegal {self!r}")
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

        """Re-resolve the block in new_body after potential iter-var
        rewriting above. Paths are unchanged because rewriting only rebuilt
        nodes in place."""
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
                f"ReverseComputeAt.apply: target ForNode was consumed by removal — "
                f"block_path={self.block_path}, target_path={self.target_path}"
            )
        final_body = _append_under(body_without, new_target_path, subtree)
        return replace(module, body=final_body)


def enumerate_reverse_compute_at_atoms(module: KernelModule) -> list[ReverseComputeAt]:
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

    atoms: list[ReverseComputeAt] = []
    for block_path, block in blocks:
        """Skip alloc blocks — empty iter_vars + zero-trip compute."""
        if not block.iter_vars:
            continue
        for target_path, _target in fornodes:
            atom = ReverseComputeAt(block_path=block_path, target_path=target_path)
            if atom.is_legal(module):
                atoms.append(atom)
    return atoms
