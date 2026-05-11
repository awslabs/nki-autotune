"""``Reorder`` atom — n-ary iter-var-keyed loop reordering.

TVM-style ``sch.reorder([iv_1, iv_2, ...])``:

- The named iter vars form a contiguous ForNode chain in the tree
  (one ForNode per iter var, no other ForNodes interleaved).
- Legality: for every adjacent pair in the requested order, roles
  commute (PAR x PAR, ACC x ACC, PAR x ACC iff subtree-pure w.r.t.
  PAR dim; SEQ never).
- Apply: reshape the chain so ForNodes appear in the given order
  top-to-bottom; grandchildren subtree passes by reference; iter var
  IDs unchanged (no BufferAccess rewriting needed).

See ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md`` 4.2.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import ForNode, KernelModule, SBlock, blocks_under, replace_at_path
from nkigym.ops.base import AxisRole
from nkigym.tune import AtomLegalityError


@dataclass(frozen=True)
class Reorder:
    """Reorder a contiguous ForNode chain by iter var ids.

    Attributes:
        iter_var_ids: Requested order of iter vars, outermost first.
            Must form a contiguous chain in the current tree.
    """

    iter_var_ids: tuple[int, ...]

    def is_legal(self, module: KernelModule) -> bool:
        """Structural + role-commute preconditions."""
        chain = _find_chain(module, set(self.iter_var_ids))
        result = False
        if chain is not None and len(chain) == len(self.iter_var_ids):
            chain_ids = {n.iter_var.var_id for n in chain}
            if set(self.iter_var_ids) == chain_ids:
                requested_nodes = _select_by_ids(chain, self.iter_var_ids)
                if requested_nodes is not None:
                    result = _permutation_legal(chain, requested_nodes)
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Reshape the tree: replace chain top with requested-order ForNode sequence.

        The chain's deepest ForNode's children become the deepest new ForNode's
        children. Grandchildren subtree passes by reference.
        """
        if not self.is_legal(module):
            raise AtomLegalityError(f"Reorder.apply: illegal {self!r}")
        chain = _find_chain(module, set(self.iter_var_ids))
        assert chain is not None
        grandchildren: list[ForNode | SBlock] = list(chain[-1].children)
        requested_nodes = _select_by_ids(chain, self.iter_var_ids)
        assert requested_nodes is not None

        """Build the new chain: deepest first."""
        new_children: list[ForNode | SBlock] = grandchildren
        new_node: ForNode | None = None
        for node in reversed(requested_nodes):
            new_node = ForNode(
                iter_var=node.iter_var, children=new_children, name=None, annotations=dict(node.annotations)
            )
            new_children = [new_node]
        assert new_node is not None

        top_path = _find_top_path(module, chain[0])
        new_body = replace_at_path(module.body, top_path, new_node)
        return replace(module, body=new_body)


def _find_chain(module: KernelModule, iter_var_ids: set[int]) -> list[ForNode] | None:
    """Return the contiguous ForNode chain binding exactly ``iter_var_ids``.

    The chain is a path of parent->child ForNodes (each has exactly one
    ForNode child, except possibly the last one). Returns ``None`` if no
    such chain exists.

    The chain must bind exactly the given set of iter vars (no more, no
    less) otherwise the requested reorder is ambiguous.
    """

    def walk(node: ForNode | SBlock) -> list[ForNode] | None:
        result: list[ForNode] | None = None
        if isinstance(node, ForNode):
            if node.iter_var.var_id in iter_var_ids:
                chain: list[ForNode] = [node]
                current: ForNode = node
                while (
                    len(current.children) == 1
                    and isinstance(current.children[0], ForNode)
                    and current.children[0].iter_var.var_id in iter_var_ids
                ):
                    current = current.children[0]
                    chain.append(current)
                if len(chain) == len(iter_var_ids):
                    result = chain
            else:
                for child in node.children:
                    found = walk(child)
                    if found is not None:
                        result = found
                        break
        return result

    discovered: list[ForNode] | None = None
    for root in module.body:
        found = walk(root)
        if found is not None:
            discovered = found
            break
    return discovered


def _select_by_ids(chain: list[ForNode], ids: tuple[int, ...]) -> list[ForNode] | None:
    """Return the chain's ForNodes reordered to match ``ids``; ``None`` if mismatch."""
    id_to_node = {n.iter_var.var_id: n for n in chain}
    result: list[ForNode] | None = None
    if set(id_to_node) == set(ids):
        result = [id_to_node[i] for i in ids]
    return result


def _permutation_legal(original: list[ForNode], new_order: list[ForNode]) -> bool:
    """Check role-commute for every pair of iter vars whose relative order changed.

    A pair ``(a, b)`` where ``a`` was originally above ``b`` AND is now below
    ``b`` (in ``new_order``) constitutes a swap. Every such swap must pass the
    pair role-commute rule. Pairs whose relative order is unchanged don't
    need checking — their pairwise legality is already satisfied by the
    existing tree.
    """
    id_to_orig_pos = {n.iter_var.var_id: i for i, n in enumerate(original)}
    id_to_new_pos = {n.iter_var.var_id: i for i, n in enumerate(new_order)}
    id_to_node = {n.iter_var.var_id: n for n in original}
    result = True
    ids = list(id_to_orig_pos.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            if id_to_orig_pos[a_id] < id_to_orig_pos[b_id]:
                orig_outer, orig_inner = a_id, b_id
            else:
                orig_outer, orig_inner = b_id, a_id
            if id_to_new_pos[orig_outer] > id_to_new_pos[orig_inner]:
                """Relative order flipped — must check the swap is legal."""
                if not _roles_commute_pair(id_to_node[orig_outer], id_to_node[orig_inner]):
                    result = False
                    break
        if not result:
            break
    return result


def _roles_commute_pair(a: ForNode, b: ForNode) -> bool:
    """TVM-style role commute.

    PAR x PAR: always.
    ACC x ACC: legal (reduce_op distinctions not encoded in v2 iter vars).
    PAR x ACC: legal iff subtree is leaf-pure w.r.t. the PAR dim.
    SEQ: never.
    """
    result: bool
    role_a = a.iter_var.role
    role_b = b.iter_var.role
    if role_a == AxisRole.SEQUENTIAL or role_b == AxisRole.SEQUENTIAL:
        result = False
    elif role_a == AxisRole.PARALLEL and role_b == AxisRole.PARALLEL:
        result = True
    elif role_a == AxisRole.ACCUMULATION and role_b == AxisRole.ACCUMULATION:
        result = True
    else:
        par_dim = a.iter_var.dim_id if role_a == AxisRole.PARALLEL else b.iter_var.dim_id
        acc_node = a if role_a == AxisRole.ACCUMULATION else b
        result = _subtree_pure_wrt_dim(acc_node, par_dim)
    return result


def _subtree_pure_wrt_dim(node: ForNode | SBlock, par_dim: str) -> bool:
    """Return True iff no block under ``node`` writes a tensor indexed by
    ``par_dim`` as PARALLEL role.

    Consults each block's ``NKIOpCall.axis_map`` + ``dim_role``. A block
    with empty writes AND empty reads_writes contributes no write — skip.
    """
    result = True
    for block in blocks_under(node):
        if not block.writes and not block.reads_writes:
            continue
        for call in block.body:
            for concrete_dim in call.axis_map.values():
                if concrete_dim == par_dim and call.dim_role.get(concrete_dim) == AxisRole.PARALLEL:
                    result = False
                    break
            if not result:
                break
        if not result:
            break
    return result


def _find_top_path(module: KernelModule, top: ForNode) -> tuple[int, ...]:
    """Walk the tree to find the path to the node that IS ``top`` (by identity)."""

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> tuple[int, ...] | None:
        result: tuple[int, ...] | None = None
        if node is top:
            result = path
        elif isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                found = walk(c, path + (i,))
                if found is not None:
                    result = found
                    break
        return result

    for i, root in enumerate(module.body):
        found = walk(root, (i,))
        if found is not None:
            return found
    raise ValueError("Reorder: top of chain not found in tree")


def enumerate_reorder_atoms(module: KernelModule) -> list[Reorder]:
    """Emit every legal adjacent pair swap in the forest (n=2 only).

    Larger n-ary reorderings are future work — the current sampler /
    agent space composes pair-swaps to reach them.
    """
    atoms: list[Reorder] = []

    def walk(node: ForNode | SBlock) -> None:
        if isinstance(node, ForNode):
            if len(node.children) == 1 and isinstance(node.children[0], ForNode):
                outer_id = node.iter_var.var_id
                inner_id = node.children[0].iter_var.var_id
                atom = Reorder(iter_var_ids=(inner_id, outer_id))
                if atom.is_legal(module):
                    atoms.append(atom)
            for c in node.children:
                walk(c)

    for root in module.body:
        walk(root)
    return atoms
