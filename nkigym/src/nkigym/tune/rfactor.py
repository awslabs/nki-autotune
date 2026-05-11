"""RFactor rewrite — fission a reducer into staging-buffer decomposition.

Takes a reducer :class:`~nkigym.codegen.ir.SBlock` (matmul or
activation_reduce) and an outer factor; emits a staging buffer plus
either a per-outer-iteration PSUM accumulator (recipe ``"rmw"``) or
slot-indexed writes (recipe ``"slot"``), closed by a ``tensor_reduce``
over the outer axis.

The structure produced:

* Alloc for the staging tensor at the forest root.
* Outer loop wrapping init + compute + drain siblings.
* Closing ``NKITensorReduce`` block that reduces the staging tensor's
  outer axis into the original destination.

Port of the pre-refactor implementation onto the iter-var IR — SBlocks
replace ``BodyLeaf``; :class:`~nkigym.codegen.ir.IterVar` allocation goes
through ``module.allocate_iter_var``; all operand accesses are expressed
as :class:`~nkigym.codegen.ir.BufferAccess` with explicit patterns.

See ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md`` §4.6.
"""

from dataclasses import dataclass
from dataclasses import replace as dc_replace

from nkigym.codegen.ir import (
    AccessRange,
    BufferAccess,
    DimInfo,
    ForNode,
    IterVar,
    KernelModule,
    NKIOpCall,
    SBlock,
    Tensor,
    TreeIR,
    resolve_node,
)
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.tune import AtomLegalityError


def _block_dim_trip(block: SBlock, dim_id: str) -> int | None:
    """Return the trip count (iter-var extent) for ``dim_id`` in ``block``.

    Returns None if ``block`` has no iter var on ``dim_id``.
    """
    for iv in block.iter_vars:
        if iv.dim_id == dim_id:
            return iv.extent
    return None


def _block_dim_tile(module: KernelModule, block: SBlock, dim_id: str) -> int | None:
    """Return the per-op tile width (access-pattern extent) for ``dim_id`` in ``block``.

    Scans the block's reads / writes / reads_writes in order and returns
    the first extent found on a tensor dim matching ``dim_id``.
    Returns None when no access in the block touches ``dim_id``.

    Precedence: reads, then writes, then reads_writes. In canonical IR,
    all accesses touching a given dim in the same block should have
    identical extents for that dim.
    """
    all_accesses = list(block.reads.values()) + list(block.writes.values()) + list(block.reads_writes.values())
    for access in all_accesses:
        tensor = module.tensors.get(access.tensor_name)
        if tensor is None:
            continue
        for i, d in enumerate(tensor.dim_ids):
            if d == dim_id and i < len(access.pattern):
                return access.pattern[i].extent
    return None


@dataclass(frozen=True)
class RFactor:
    """Fission a reducer into outer-split + staging + close.

    Attributes:
        reducer_block_path: Path to the reducer's :class:`SBlock` —
            :class:`NKIMatmul` for recipe ``"rmw"``,
            :class:`NKIActivationReduce` for recipe ``"slot"``.
        outer_factor: The outer loop's trip count post-split. Must
            strictly divide the reducer block's iter-var extent on the
            accumulation dim, and be strictly between 1 and that extent.
    """

    reducer_block_path: tuple[int, ...]
    outer_factor: int

    def is_legal(self, module: KernelModule) -> bool:
        """Structural + divisibility + recipe-specific preconditions."""
        block = resolve_node(module.body, self.reducer_block_path)
        if not isinstance(block, SBlock):
            return False
        if len(block.body) != 1:
            return False
        call = block.body[0]
        recipe = call.op_cls.RFACTOR_RECIPE
        if recipe is None:
            return False
        acc_dim = _accumulation_dim(call)
        if acc_dim is None:
            return False
        dim_info = module.dims.get(acc_dim)
        if dim_info is None:
            return False
        num_t = _block_dim_trip(block, acc_dim)
        if num_t is None:
            return False
        if num_t <= 1:
            return False
        if self.outer_factor <= 1 or self.outer_factor >= num_t:
            return False
        if num_t % self.outer_factor != 0:
            return False
        if recipe == "rmw":
            return _is_legal_rmw(block, acc_dim)
        if recipe == "slot":
            return _is_legal_slot(block, acc_dim)
        return False

    def apply(self, module: KernelModule) -> KernelModule:
        """Run recipe-specific rewrite; return a fresh :class:`KernelModule`."""
        if not self.is_legal(module):
            raise AtomLegalityError(f"RFactor.apply: illegal {self!r}")
        block = resolve_node(module.body, self.reducer_block_path)
        assert isinstance(block, SBlock)
        call = block.body[0]
        recipe = call.op_cls.RFACTOR_RECIPE
        if recipe == "rmw":
            return _apply_rmw(module, self.reducer_block_path, self.outer_factor)
        if recipe == "slot":
            return _apply_slot(module, self.reducer_block_path, self.outer_factor)
        raise AtomLegalityError(f"RFactor.apply: unsupported recipe {recipe!r}")


def enumerate_rfactor_atoms(module: KernelModule) -> list[RFactor]:
    """Emit one :class:`RFactor` atom per (reducer block, valid divisor)."""
    atoms: list[RFactor] = []

    def walk(node: ForNode | SBlock, path: tuple[int, ...]) -> None:
        if isinstance(node, SBlock):
            if len(node.body) == 1 and node.body[0].op_cls.RFACTOR_RECIPE is not None:
                call = node.body[0]
                acc_dim = _accumulation_dim(call)
                if acc_dim is not None:
                    dim_info = module.dims.get(acc_dim)
                    if dim_info is not None:
                        num_t = _block_dim_trip(node, acc_dim)
                        if num_t is not None:
                            for factor in _strict_divisors(num_t):
                                atom = RFactor(reducer_block_path=path, outer_factor=factor)
                                if atom.is_legal(module):
                                    atoms.append(atom)
        else:
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms


def _strict_divisors(n: int) -> list[int]:
    """Return every ``d`` in ``(1, n)`` dividing ``n``."""
    return [d for d in range(2, n) if n % d == 0]


def _accumulation_dim(call: NKIOpCall) -> str | None:
    """Return the concrete dim id the op uses as its accumulation axis, or ``None``."""
    roles = getattr(call.op_cls, "AXIS_ROLES", {})
    for abstract, role in roles.items():
        if role == AxisRole.ACCUMULATION and abstract in call.axis_map:
            return call.axis_map[abstract]
    return None


def _is_legal_rmw(block: SBlock, acc_dim: str) -> bool:
    """rmw legality: block has RMW output + ACC dim iter var with ACCUMULATION role."""
    if not block.reads_writes:
        return False
    for iv in block.iter_vars:
        if iv.dim_id == acc_dim:
            return iv.role == AxisRole.ACCUMULATION
    return False


def _is_legal_slot(block: SBlock, acc_dim: str) -> bool:
    """slot legality: block has the ACC dim iter var (role already ACCUMULATION at canonical)."""
    for iv in block.iter_vars:
        if iv.dim_id == acc_dim:
            return iv.role == AxisRole.ACCUMULATION
    return False


def _fresh_dim_id(dims: dict[str, DimInfo]) -> str:
    """Pick a dim_id not yet in use: ``d<N>`` for the smallest free ``N``."""
    taken = set(dims.keys())
    i = 0
    while f"d{i}" in taken:
        i += 1
    return f"d{i}"


def _apply_rmw(module: KernelModule, block_path: tuple[int, ...], outer_factor: int) -> KernelModule:
    """Apply rmw recipe to a matmul reducer.

    Layout: dim_ids for ``psum_partials`` are ordered ``(P-dim, outer-dim,
    ...remaining-dims)`` — the outer dim sits as a middle slot (a
    per-iteration index), and the tensor's original F-axis stays last so
    :func:`place_buffers` gives the tensor full capacity.
    """
    matmul_block = resolve_node(module.body, block_path)
    assert isinstance(matmul_block, SBlock)
    assert len(matmul_block.body) == 1
    call = matmul_block.body[0]
    k_dim = _accumulation_dim(call)
    assert k_dim is not None

    """Identify the PSUM accumulator — the single RMW write."""
    psum_acc_name = _single_rmw_name(matmul_block)
    psum_acc = module.tensors[psum_acc_name]
    k_trip = _block_dim_trip(matmul_block, k_dim)
    if k_trip is None:
        raise AtomLegalityError("RFactor.apply (rmw): matmul block has no iter var on accumulation dim")
    inner_trip = k_trip // outer_factor

    """Declare the synthetic outer dim. The accumulation dim ``k_dim`` is
    split at the MATMUL iter-var level (Split-style) — its trip count
    stays unchanged on other users of the dim (loads, stores) so they
    keep iterating over the full K."""
    outer_dim_id = _fresh_dim_id(module.dims)
    new_dims = dict(module.dims)
    new_dims[outer_dim_id] = DimInfo(dim_id=outer_dim_id, total_size=outer_factor)

    """Materialise staging + local PSUM tensors. Ordering of
    ``psum_partials.dim_ids``: P first, outer next, then the remaining
    F-axis dims — keeps the original F dim as the trailing physical axis
    so ``place_buffers`` gives the full tile size rather than a slot
    count."""
    partials_name = "psum_partials"
    local_name = "psum_acc_local"
    p_dim = psum_acc.dim_ids[0]
    rest_dims = tuple(d for d in psum_acc.dim_ids[1:])
    partials_dim_ids = (p_dim, outer_dim_id, *rest_dims)
    partials_shape = tuple([psum_acc.shape[0], outer_factor] + list(psum_acc.shape[1:]))

    partials = Tensor(
        name=partials_name,
        dim_ids=partials_dim_ids,
        shape=partials_shape,
        dtype=psum_acc.dtype,
        origin="intermediate",
        location="sbuf",
    )
    acc_local = Tensor(
        name=local_name,
        dim_ids=psum_acc.dim_ids,
        shape=psum_acc.shape,
        dtype=psum_acc.dtype,
        origin="intermediate",
        location="psum",
    )
    new_tensors: dict[str, Tensor] = {k: v for k, v in module.tensors.items() if k != psum_acc_name}
    new_tensors[partials_name] = partials
    new_tensors[local_name] = acc_local

    """Locate sibling alloc / memset / drain trees. ``drain`` = the
    NKITensorCopy whose src reads ``psum_acc_name`` — its dst is the
    downstream SBUF product tensor."""
    alloc_idx = _find_alloc_root_for(module.body, psum_acc_name)
    memset_idx = _find_memset_root_for(module.body, psum_acc_name)
    drain_idx = _find_tensor_copy_root_src(module.body, psum_acc_name)
    if alloc_idx is None or memset_idx is None or drain_idx is None:
        raise AtomLegalityError("RFactor.apply (rmw): expected NKIAlloc + NKIMemset + NKITensorCopy root for psum_acc")

    drain_root = module.body[drain_idx]
    drain_dst_name = _tensor_copy_dst_name(drain_root, psum_acc_name)
    if drain_dst_name is None:
        raise AtomLegalityError("RFactor.apply (rmw): could not resolve drain target name")

    """Build new forest components — all require iter-var allocation via ``module``,
    so we do so AFTER ``new_dims`` is fixed but BEFORE returning. ``module``'s
    ``iter_var_counter`` is shared across the module; allocating on the
    original module is fine — the returned module will carry the incremented
    counter."""
    alloc_partials_block = _make_alloc_block(partials_name, partials)
    alloc_local_block = _make_alloc_block(local_name, acc_local)

    """Outer iter var — wraps the K_outer block. Role=PARALLEL because each
    outer step produces an independent partial result that the close-reduce
    sums associatively."""
    iv_k_outer = module.allocate_iter_var(dim_id=outer_dim_id, extent=outer_factor, role=AxisRole.PARALLEL)

    """Rebuild the memset tree: substitute ``psum_acc_name`` writes with
    ``local_name`` (PSUM local). The iter vars come from fresh allocations
    matched to the original memset block's iter var dim ids."""
    memset_local_tree = _rebuild_memset_tree_for_local(module, module.body[memset_idx], psum_acc_name, local_name)

    """Rebuild the matmul K-loop tree: split K into (K_outer, K_inner).
    The K-outer ForNode is placed outside; K_inner replaces the original
    K ForNode with reduced extent. The matmul writes to ``local_name``."""
    k_inner_tree = _rebuild_matmul_tree_with_inner_k(
        module, module.body[block_path[0]], block_path[1:], k_dim, inner_trip, local_name, iv_k_outer
    )

    """Rebuild the drain tree: NKITensorCopy(src=local_name → partials[outer])
    — sources ``psum_acc_local`` and writes a partials slot keyed by
    ``iv_k_outer``."""
    drain_local_tree = _rebuild_drain_tree_for_partials(
        module, module.body[drain_idx], psum_acc_name, drain_dst_name, local_name, partials_name, iv_k_outer
    )

    """Combine the four K-outer siblings under a single K-outer ForNode."""
    k_outer_forest = ForNode(
        iter_var=iv_k_outer, children=[alloc_local_block, memset_local_tree, k_inner_tree, drain_local_tree]
    )

    """Build close-reduce tree: iterate (P-dim, F-dims) excluding outer;
    tensor_reduce reduces the outer axis of partials into ``drain_dst_name``."""
    close_tree = _build_close_reduce_tree(
        module, module.body[drain_idx], partials_name, drain_dst_name, outer_dim_id, new_dims
    )

    """Reassemble the forest: original roots minus the four removed trees,
    with alloc_partials + k_outer_forest + close_tree spliced in where the
    matmul tree was."""
    matmul_root_idx = block_path[0]
    indices_to_remove = {alloc_idx, memset_idx, matmul_root_idx, drain_idx}
    new_body: TreeIR = []
    inserted = False
    for i, root in enumerate(module.body):
        if i in indices_to_remove:
            if i == matmul_root_idx and not inserted:
                new_body.append(alloc_partials_block)
                new_body.append(k_outer_forest)
                new_body.append(close_tree)
                inserted = True
            continue
        new_body.append(root)

    return dc_replace(module, tensors=new_tensors, dims=new_dims, body=new_body)


def _single_rmw_name(block: SBlock) -> str:
    """Return the tensor name of the block's single RMW-write operand."""
    if len(block.reads_writes) != 1:
        raise AtomLegalityError(f"RFactor (rmw): expected exactly 1 RMW operand, got {len(block.reads_writes)}")
    access = next(iter(block.reads_writes.values()))
    return access.tensor_name


def _find_alloc_root_for(body: TreeIR, tensor_name: str) -> int | None:
    """Forest-root index of the ``NKIAlloc`` SBlock writing ``tensor_name``."""
    for i, root in enumerate(body):
        if isinstance(root, SBlock) and len(root.body) == 1:
            call = root.body[0]
            if call.op_cls is NKIAlloc and call.kwargs.get("tensor_name") == tensor_name:
                return i
    return None


def _find_memset_root_for(body: TreeIR, tensor_name: str) -> int | None:
    """Forest-root index of the first tree whose SBlocks include a memset writing ``tensor_name``."""
    for i, root in enumerate(body):
        if _tree_has_memset_writing(root, tensor_name):
            return i
    return None


def _find_tensor_copy_root_src(body: TreeIR, tensor_name: str) -> int | None:
    """Forest-root index of the first tree whose SBlocks include a tensor_copy reading ``tensor_name``."""
    for i, root in enumerate(body):
        if _tree_has_tensor_copy_reading(root, tensor_name):
            return i
    return None


def _tree_has_memset_writing(node: ForNode | SBlock, tensor_name: str) -> bool:
    """Recursive predicate: does any SBlock under ``node`` memset-write ``tensor_name``?"""
    if isinstance(node, SBlock):
        if len(node.body) == 1 and node.body[0].op_cls is NKIMemset:
            return any(a.tensor_name == tensor_name for a in node.writes.values())
        return False
    return any(_tree_has_memset_writing(c, tensor_name) for c in node.children)


def _tree_has_tensor_copy_reading(node: ForNode | SBlock, tensor_name: str) -> bool:
    """Recursive predicate: does any SBlock under ``node`` tensor_copy-read ``tensor_name``?"""
    if isinstance(node, SBlock):
        if len(node.body) == 1 and node.body[0].op_cls is NKITensorCopy:
            return any(a.tensor_name == tensor_name for a in node.reads.values())
        return False
    return any(_tree_has_tensor_copy_reading(c, tensor_name) for c in node.children)


def _tensor_copy_dst_name(node: ForNode | SBlock, src_name: str) -> str | None:
    """Walk ``node``; return the first tensor_copy dst whose src is ``src_name``."""
    if isinstance(node, SBlock):
        if len(node.body) == 1 and node.body[0].op_cls is NKITensorCopy:
            reads_match = any(a.tensor_name == src_name for a in node.reads.values())
            if reads_match:
                for access in node.writes.values():
                    return access.tensor_name
        return None
    for c in node.children:
        got = _tensor_copy_dst_name(c, src_name)
        if got is not None:
            return got
    return None


def _make_alloc_block(name: str, tensor: Tensor) -> SBlock:
    """Build an empty-iter-var :class:`SBlock` that allocates ``name``.

    Kwargs follow the canonical builder's shape: the renderer reads
    ``location`` / ``dtype`` to emit the ``nl.ndarray(...)`` call. ``shape``
    is resolved at emit time by ``place_buffers``, but we pass the initial
    shape for completeness.
    """
    call = NKIOpCall(
        op_cls=NKIAlloc,
        kwargs={"tensor_name": name, "location": tensor.location, "shape": tensor.shape, "dtype": tensor.dtype},
        axis_map={},
        dim_role={},
    )
    return SBlock(
        iter_vars=[],
        reads={},
        writes={"output": BufferAccess(tensor_name=name, iter_var_ids=(), pattern=())},
        reads_writes={},
        body=[call],
    )


def _collect_loop_ancestors(root: ForNode | SBlock, path: tuple[int, ...]) -> tuple[list[IterVar], SBlock | None]:
    """Walk ``path`` from ``root``; return (ForNode iter vars, target SBlock)."""
    ivs: list[IterVar] = []
    node: ForNode | SBlock = root
    for idx in path:
        if isinstance(node, ForNode):
            ivs.append(node.iter_var)
            node = node.children[idx]
        else:
            return ivs, None
    if isinstance(node, SBlock):
        return ivs, node
    if isinstance(node, ForNode):
        """Target inside the subtree — recurse descending children[0]."""
        while isinstance(node, ForNode):
            ivs.append(node.iter_var)
            node = node.children[0]
        assert isinstance(node, SBlock)
        return ivs, node
    return ivs, None


def _find_inner_block(root: ForNode | SBlock) -> SBlock:
    """Return the single SBlock descendant of ``root`` (linear ForNode chain assumed)."""
    node: ForNode | SBlock = root
    while isinstance(node, ForNode):
        if len(node.children) != 1:
            raise AtomLegalityError("RFactor: expected a linear ForNode→SBlock chain")
        node = node.children[0]
    assert isinstance(node, SBlock)
    return node


def _forest_iter_vars(root: ForNode | SBlock) -> list[IterVar]:
    """Collect enclosing iter vars of the unique SBlock under ``root`` (outermost → innermost)."""
    ivs: list[IterVar] = []
    node: ForNode | SBlock = root
    while isinstance(node, ForNode):
        ivs.append(node.iter_var)
        if len(node.children) != 1:
            raise AtomLegalityError("RFactor: expected a linear ForNode→SBlock chain in sibling tree")
        node = node.children[0]
    return ivs


def _rebuild_memset_tree_for_local(
    module: KernelModule, root: ForNode | SBlock, old_name: str, new_name: str
) -> ForNode | SBlock:
    """Clone the memset tree with writes retargeted from ``old_name`` to ``new_name``.

    Allocates fresh iter vars so the new tree does not share IterVar
    identity with the original (identity leaks distort the dep-cache
    signature check).
    """
    orig_block = _find_inner_block(root)
    old_ivs = _forest_iter_vars(root)
    iv_map = {iv.var_id: module.allocate_iter_var(dim_id=iv.dim_id, extent=iv.extent, role=iv.role) for iv in old_ivs}
    new_ivs = [iv_map[iv.var_id] for iv in old_ivs]

    """Rewrite the writes BufferAccess: swap tensor name + iter var ids."""
    new_writes = {
        slot: _rewrite_access_tensor_and_ivs(access, old_name, new_name, iv_map)
        for slot, access in orig_block.writes.items()
    }
    new_reads = {
        slot: _rewrite_access_tensor_and_ivs(access, old_name, new_name, iv_map)
        for slot, access in orig_block.reads.items()
    }
    new_reads_writes = {
        slot: _rewrite_access_tensor_and_ivs(access, old_name, new_name, iv_map)
        for slot, access in orig_block.reads_writes.items()
    }
    new_block = SBlock(
        iter_vars=new_ivs,
        reads=new_reads,
        writes=new_writes,
        reads_writes=new_reads_writes,
        body=list(orig_block.body),
        annotations=dict(orig_block.annotations),
    )
    return _wrap_in_fornodes(new_block, new_ivs)


def _rebuild_matmul_tree_with_inner_k(
    module: KernelModule,
    root: ForNode | SBlock,
    body_path: tuple[int, ...],
    k_dim: str,
    inner_trip: int,
    local_name: str,
    iv_k_outer: IterVar,
) -> ForNode | SBlock:
    """Rebuild the matmul tree with K split into (K_outer, K_inner).

    K_outer is shared with the enclosing RFactor forest (the
    ``iv_k_outer`` passed in); this function wraps the matmul block in
    per-dim ForNodes for NON-K dims + a K_inner ForNode with role=ACC.
    K accesses in ``lhs_T_sbuf`` / ``rhs_sbuf`` become
    ``k_outer * inner_trip + k_inner``.

    The RMW write on ``psum_acc`` is retargeted to ``local_name``; no
    K iter var participates in the RMW pattern (the original matmul
    write pattern only references M, N — K is implicit via
    accumulation).
    """
    _ = body_path
    orig_block = _find_inner_block(root)
    old_ivs = _forest_iter_vars(root)
    psum_acc_name = _single_rmw_name(orig_block)
    iv_map: dict[int, IterVar] = {}
    new_outer_ivs: list[IterVar] = []
    k_inner: IterVar | None = None
    old_k_iv_id: int | None = None
    for iv in old_ivs:
        if iv.dim_id == k_dim:
            old_k_iv_id = iv.var_id
            k_inner = module.allocate_iter_var(dim_id=iv.dim_id, extent=inner_trip, role=AxisRole.ACCUMULATION)
            iv_map[iv.var_id] = k_inner
        else:
            fresh = module.allocate_iter_var(dim_id=iv.dim_id, extent=iv.extent, role=iv.role)
            iv_map[iv.var_id] = fresh
            new_outer_ivs.append(fresh)

    if k_inner is None or old_k_iv_id is None:
        raise AtomLegalityError("RFactor (rmw): matmul tree does not bind the accumulation dim")

    """Rewrite block iter_vars + accesses. For LHS/RHS reads (which include
    the K dim), rewrite their affine pattern to
    ``k_outer * inner_trip + k_inner`` (Split-style expansion)."""
    new_block_ivs: list[IterVar] = []
    for iv in orig_block.iter_vars:
        new_block_ivs.append(iv_map[iv.var_id])
    new_reads: dict[str, BufferAccess] = {}
    for slot, access in orig_block.reads.items():
        new_reads[slot] = _rewrite_k_split_access(
            access, old_k_iv_id, iv_k_outer.var_id, k_inner.var_id, inner_trip, iv_map
        )
    new_writes = {slot: _remap_access_ivs(access, iv_map) for slot, access in orig_block.writes.items()}
    new_reads_writes = {
        slot: _rewrite_access_tensor_and_ivs(access, psum_acc_name, local_name, iv_map)
        for slot, access in orig_block.reads_writes.items()
    }
    new_block = SBlock(
        iter_vars=new_block_ivs,
        reads=new_reads,
        writes=new_writes,
        reads_writes=new_reads_writes,
        body=list(orig_block.body),
        annotations=dict(orig_block.annotations),
    )
    """Wrap in outer non-K ForNodes + K_inner ForNode. We preserve the
    original loop nesting order (outer non-K dims first, then K_inner
    innermost) — identical to the canonical matmul tree's order modulo
    replacing K with K_inner."""
    node: ForNode | SBlock = new_block
    """Nest K_inner (innermost); but if the original K was in the middle of
    the loop nest, the outer_ivs that come AFTER K in the original order
    must nest INSIDE K_inner. Walk in reverse to preserve order."""
    ordered_new_ivs: list[IterVar] = []
    for iv in old_ivs:
        ordered_new_ivs.append(iv_map[iv.var_id])
    for iv in reversed(ordered_new_ivs):
        node = ForNode(iter_var=iv, children=[node])
    return node


def _rewrite_k_split_access(
    access: BufferAccess,
    old_k_iv_id: int,
    k_outer_id: int,
    k_inner_id: int,
    inner_trip: int,
    iv_map: dict[int, IterVar],
) -> BufferAccess:
    """Rewrite an access where the K iter var decomposes into
    ``k_outer * inner_trip + k_inner``.

    All non-K iter vars remap via ``iv_map``; the K iter var splits into
    a sum of (outer, coeff=inner_trip) + (inner, coeff=1).
    """
    new_ranges: list[AccessRange] = []
    k_pattern_touched = False
    for ar in access.pattern:
        new_coeffs: dict[int, int] = {}
        for iv_id, c in ar.iter_var_coeffs:
            if iv_id == old_k_iv_id:
                new_coeffs[k_outer_id] = new_coeffs.get(k_outer_id, 0) + c * inner_trip
                new_coeffs[k_inner_id] = new_coeffs.get(k_inner_id, 0) + c
                k_pattern_touched = True
            elif iv_id in iv_map:
                new_coeffs[iv_map[iv_id].var_id] = c
            else:
                new_coeffs[iv_id] = c
        new_ranges.append(AccessRange.make(new_coeffs, ar.const_offset, ar.extent))

    new_ids: list[int] = []
    seen: set[int] = set()
    for iv_id in access.iter_var_ids:
        mapped = iv_map[iv_id].var_id if iv_id in iv_map and iv_id != old_k_iv_id else iv_id
        if iv_id == old_k_iv_id:
            for sub in (k_outer_id, k_inner_id):
                if sub not in seen:
                    seen.add(sub)
                    new_ids.append(sub)
        else:
            if mapped not in seen:
                seen.add(mapped)
                new_ids.append(mapped)
    """Append k_outer id if not yet present (K-less accesses keep unchanged)."""
    if k_pattern_touched:
        for req in (k_outer_id, k_inner_id):
            if req not in seen:
                seen.add(req)
                new_ids.append(req)
    return BufferAccess(tensor_name=access.tensor_name, iter_var_ids=tuple(new_ids), pattern=tuple(new_ranges))


def _rebuild_drain_tree_for_partials(
    module: KernelModule,
    root: ForNode | SBlock,
    old_src_name: str,
    old_dst_name: str,
    new_src_name: str,
    new_dst_name: str,
    iv_k_outer: IterVar,
) -> ForNode | SBlock:
    """Clone the drain tree: tensor_copy retargeted from (old_src, old_dst) → (new_src, new_dst[outer])."""
    orig_block = _find_inner_block(root)
    old_ivs = _forest_iter_vars(root)
    iv_map = {iv.var_id: module.allocate_iter_var(dim_id=iv.dim_id, extent=iv.extent, role=iv.role) for iv in old_ivs}
    new_ivs = [iv_map[iv.var_id] for iv in old_ivs]

    new_reads = {
        slot: _rewrite_access_tensor_and_ivs(access, old_src_name, new_src_name, iv_map)
        for slot, access in orig_block.reads.items()
    }
    new_writes = {
        slot: _rewrite_drain_write_with_outer(access, old_dst_name, new_dst_name, iv_map, iv_k_outer)
        for slot, access in orig_block.writes.items()
    }
    new_block_ivs = [iv_map[iv.var_id] for iv in orig_block.iter_vars]

    new_block = SBlock(
        iter_vars=new_block_ivs,
        reads=new_reads,
        writes=new_writes,
        reads_writes={},
        body=list(orig_block.body),
        annotations=dict(orig_block.annotations),
    )
    return _wrap_in_fornodes(new_block, new_ivs)


def _rewrite_drain_write_with_outer(
    access: BufferAccess, old_name: str, new_name: str, iv_map: dict[int, IterVar], iv_outer: IterVar
) -> BufferAccess:
    """Retarget write access's tensor name; insert an outer-dim slot range after the partition dim.

    The staging tensor ``psum_partials`` has dim_ids ``(P, outer, ...rest)``.
    The original drain writes accessed ``(P, ...rest)``; we insert a
    one-slot access for the ``outer`` dim between them.
    """
    if access.tensor_name != old_name:
        return _rewrite_access_tensor_and_ivs(access, old_name, new_name, iv_map)
    """Map the original pattern through iv_map first."""
    remapped = _remap_pattern_ivs(access.pattern, iv_map)
    if not remapped:
        raise AtomLegalityError("RFactor (rmw): drain access has empty pattern")
    """Splice in an outer-dim AccessRange right after the partition (first) dim."""
    outer_ar = AccessRange.make({iv_outer.var_id: 1}, 0, 1)
    new_pattern = (remapped[0], outer_ar, *remapped[1:])
    """Rebuild iter_var_ids: include the original ids plus iv_outer."""
    orig_ids = tuple(_remap_iv_ids(access.iter_var_ids, iv_map))
    new_ids = (*orig_ids, iv_outer.var_id)
    return BufferAccess(tensor_name=new_name, iter_var_ids=new_ids, pattern=new_pattern)


def _build_close_reduce_tree(
    module: KernelModule,
    drain_root: ForNode | SBlock,
    partials_name: str,
    original_dst_name: str,
    outer_dim_id: str,
    new_dims: dict[str, DimInfo],
) -> ForNode | SBlock:
    """Build the close-reduce tree: NKITensorReduce reads ``partials`` (full outer slice) and writes ``original_dst_name``.

    Iter vars mirror the original drain's iter vars (per-(p, f) tile).
    The outer dim is NOT bound to an iter var in this block — its
    :class:`AccessRange` carries empty coefficients and ``extent =
    outer_factor`` so the renderer emits ``0:outer_factor`` along that
    physical dim.
    """
    orig_block = _find_inner_block(drain_root)
    old_ivs = _forest_iter_vars(drain_root)
    iv_map = {iv.var_id: module.allocate_iter_var(dim_id=iv.dim_id, extent=iv.extent, role=iv.role) for iv in old_ivs}
    new_ivs = [iv_map[iv.var_id] for iv in old_ivs]

    """Derive the original per-dim write access for the drain target;
    re-use it for the reduce's dst (same iteration footprint)."""
    drain_write_access = _find_access_for_tensor(orig_block.writes, original_dst_name)
    if drain_write_access is None:
        raise AtomLegalityError("RFactor (rmw): drain target write access not found")
    dst_access = _remap_access_ivs(drain_write_access, iv_map)

    """Partial read access — same (P, ...rest) footprint as the drain's
    write, but with an extra no-iter-var AccessRange for the outer dim
    spanning its full extent."""
    outer_info = new_dims[outer_dim_id]
    outer_ar = AccessRange.make({}, 0, outer_info.total_size)
    orig_dst_pattern = dst_access.pattern
    new_pattern = (orig_dst_pattern[0], outer_ar, *orig_dst_pattern[1:])
    new_iv_ids = tuple(_remap_iv_ids(drain_write_access.iter_var_ids, iv_map))
    reduce_read_access = BufferAccess(tensor_name=partials_name, iter_var_ids=new_iv_ids, pattern=new_pattern)

    """Determine the axis (0-indexed) of the outer dim within the
    SLICED tile dimensions that ``NKITensorReduce`` observes. When the
    close-reduce slices ``psum_partials[0:P_tile, m_slot, 0:outer_factor,
    f_start:f_end]``, the resulting tile has shape ``(P_tile,
    outer_factor, F_tile)`` — the outer dim becomes axis=1 in the local
    tile view. The block-level slot index (m_slot) projects away.
    """
    reduce_axis = 1

    call = NKIOpCall(op_cls=NKITensorReduce, kwargs={"axis": reduce_axis, "op": "add"}, axis_map={}, dim_role={})
    new_block = SBlock(
        iter_vars=new_ivs, reads={"data": reduce_read_access}, writes={"dst": dst_access}, reads_writes={}, body=[call]
    )
    return _wrap_in_fornodes(new_block, new_ivs)


def _rewrite_access_tensor_and_ivs(
    access: BufferAccess, old_name: str, new_name: str, iv_map: dict[int, IterVar]
) -> BufferAccess:
    """Return a fresh BufferAccess — tensor rebinding + iter-var remap."""
    remapped = _remap_pattern_ivs(access.pattern, iv_map)
    new_ids = tuple(_remap_iv_ids(access.iter_var_ids, iv_map))
    tname = new_name if access.tensor_name == old_name else access.tensor_name
    return BufferAccess(tensor_name=tname, iter_var_ids=new_ids, pattern=remapped)


def _remap_access_ivs(access: BufferAccess, iv_map: dict[int, IterVar]) -> BufferAccess:
    """Return a fresh BufferAccess with only iter-var ids remapped."""
    remapped = _remap_pattern_ivs(access.pattern, iv_map)
    new_ids = tuple(_remap_iv_ids(access.iter_var_ids, iv_map))
    return BufferAccess(tensor_name=access.tensor_name, iter_var_ids=new_ids, pattern=remapped)


def _remap_pattern_ivs(pattern: tuple[AccessRange, ...], iv_map: dict[int, IterVar]) -> tuple[AccessRange, ...]:
    """Remap iter-var ids inside every :class:`AccessRange`."""
    new_ranges: list[AccessRange] = []
    for ar in pattern:
        new_coeffs: dict[int, int] = {}
        for iv_id, c in ar.iter_var_coeffs:
            if iv_id in iv_map:
                new_coeffs[iv_map[iv_id].var_id] = c
            else:
                new_coeffs[iv_id] = c
        new_ranges.append(AccessRange.make(new_coeffs, ar.const_offset, ar.extent))
    return tuple(new_ranges)


def _remap_iv_ids(ids: tuple[int, ...], iv_map: dict[int, IterVar]) -> list[int]:
    """Remap an ordered list of iter-var ids, deduping while preserving order."""
    out: list[int] = []
    seen: set[int] = set()
    for iv_id in ids:
        mapped = iv_map[iv_id].var_id if iv_id in iv_map else iv_id
        if mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out


def _find_access_for_tensor(access_map: dict[str, BufferAccess], tensor_name: str) -> BufferAccess | None:
    """Return the first access in ``access_map`` that targets ``tensor_name``."""
    for access in access_map.values():
        if access.tensor_name == tensor_name:
            return access
    return None


def _wrap_in_fornodes(block: SBlock, ivs: list[IterVar]) -> ForNode | SBlock:
    """Wrap ``block`` in one ForNode per entry of ``ivs`` (outermost first)."""
    node: ForNode | SBlock = block
    for iv in reversed(ivs):
        node = ForNode(iter_var=iv, children=[node])
    return node


def _apply_slot(module: KernelModule, block_path: tuple[int, ...], outer_factor: int) -> KernelModule:
    """Apply slot recipe to an :class:`NKIActivationReduce` reducer.

    Transforms the canonical form:

    ::

        F_loop(F, ACC) {
            NKIActivationReduce(data, dst=scratch, reduce_res=sum_acc)
        }

    into:

    ::

        F_outer_loop(outer_factor, PARALLEL) {
            F_inner_loop(inner_F, ACC) {
                NKIActivationReduce(data, dst=scratch_local, reduce_res=partials[F_outer])
            }
        }
        NKITensorReduce(partials → sum_acc, axis=last)
    """
    ar_block = resolve_node(module.body, block_path)
    assert isinstance(ar_block, SBlock)
    assert len(ar_block.body) == 1
    call = ar_block.body[0]
    f_dim = _accumulation_dim(call)
    assert f_dim is not None
    f_trip = _block_dim_trip(ar_block, f_dim)
    if f_trip is None:
        raise AtomLegalityError("RFactor.apply (slot): activation_reduce block has no iter var on accumulation dim")
    inner_f_trip = f_trip // outer_factor

    """Identify operand tensor names. ``NKIActivationReduce`` declares
    OPERAND_AXES in the order (data, dst, reduce_res); the block puts
    ``dst`` + ``reduce_res`` into ``writes``."""
    scratch_name = _writes_name(ar_block, "dst")
    sum_acc_name = _writes_name(ar_block, "reduce_res")
    if scratch_name is None or sum_acc_name is None:
        raise AtomLegalityError("RFactor.apply (slot): expected dst + reduce_res writes on activation_reduce")
    sum_acc = module.tensors[sum_acc_name]
    scratch = module.tensors[scratch_name]

    """Declare the synthetic outer dim. Like the rmw recipe, we split the
    F iter var at the block level (Split-style) instead of mutating the
    F dim's trip count — other users of ``f_dim`` (the outer load)
    continue to iterate the full F extent."""
    outer_dim_id = _fresh_dim_id(module.dims)
    new_dims = dict(module.dims)
    new_dims[outer_dim_id] = DimInfo(dim_id=outer_dim_id, total_size=outer_factor)

    """Staging tensors: ``partials`` holds a per-outer-step scalar per P
    row (1D → 2D w/ outer); ``scratch_local`` per-outer activation tile
    — keeps same dim_ids but the F dim now covers ``inner_f_trip`` tiles."""
    partials_name = "partials"
    local_name = "scratch_local"
    partials = Tensor(
        name=partials_name,
        dim_ids=(*sum_acc.dim_ids, outer_dim_id),
        shape=(*sum_acc.shape, outer_factor),
        dtype=sum_acc.dtype,
        origin="intermediate",
        location="sbuf",
    )
    scratch_local = Tensor(
        name=local_name,
        dim_ids=scratch.dim_ids,
        shape=scratch.shape,
        dtype=scratch.dtype,
        origin="intermediate",
        location="sbuf",
    )
    new_tensors: dict[str, Tensor] = {k: v for k, v in module.tensors.items() if k != scratch_name}
    new_tensors[partials_name] = partials
    new_tensors[local_name] = scratch_local

    alloc_partials_block = _make_alloc_block(partials_name, partials)
    alloc_local_block = _make_alloc_block(local_name, scratch_local)

    """F-outer iter var wraps the F-inner tree. Role=PARALLEL because each
    outer step writes a distinct slot of partials."""
    iv_f_outer = module.allocate_iter_var(dim_id=outer_dim_id, extent=outer_factor, role=AxisRole.PARALLEL)

    """Rebuild the ar-tree: iterate over original dims (mapping the F dim's
    iter var to an inner-F iter var with reduced extent). The block's
    ``reduce_res`` slot redirects to ``partials`` with an extra
    ``iv_f_outer`` slot index; the ``dst`` slot redirects to
    ``scratch_local``."""
    ar_tree = _rebuild_ar_tree_with_inner_f(
        module,
        module.body[block_path[0]],
        f_dim,
        inner_f_trip,
        scratch_name,
        sum_acc_name,
        local_name,
        partials_name,
        iv_f_outer,
    )

    f_outer_forest = ForNode(iter_var=iv_f_outer, children=[ar_tree])

    """Build close-reduce tree: tensor_reduce over outer axis of partials
    into sum_acc."""
    close_tree = _build_close_reduce_slot_tree(module, ar_block, partials_name, sum_acc_name, outer_dim_id, new_dims)

    """Drop: the scratch NKIAlloc root + the AR-root (replaced above).
    Insert alloc_partials + alloc_local + f_outer + close_tree at AR-root's position."""
    scratch_alloc_idx = _find_alloc_root_for(module.body, scratch_name)
    ar_root_idx = block_path[0]
    indices_to_remove: set[int] = {ar_root_idx}
    if scratch_alloc_idx is not None:
        indices_to_remove.add(scratch_alloc_idx)

    new_body: TreeIR = []
    inserted = False
    for i, root in enumerate(module.body):
        if i in indices_to_remove:
            if i == ar_root_idx and not inserted:
                new_body.append(alloc_partials_block)
                new_body.append(alloc_local_block)
                new_body.append(f_outer_forest)
                new_body.append(close_tree)
                inserted = True
            continue
        new_body.append(root)

    return dc_replace(module, tensors=new_tensors, dims=new_dims, body=new_body)


def _writes_name(block: SBlock, slot: str) -> str | None:
    """Return the tensor name bound to ``slot`` in ``block.writes``, or ``None``."""
    access = block.writes.get(slot)
    if access is None:
        return None
    return access.tensor_name


def _rebuild_ar_tree_with_inner_f(
    module: KernelModule,
    root: ForNode | SBlock,
    f_dim: str,
    inner_f_trip: int,
    scratch_name: str,
    sum_acc_name: str,
    local_name: str,
    partials_name: str,
    iv_f_outer: IterVar,
) -> ForNode | SBlock:
    """Clone the activation_reduce tree with F split into (F_outer, F_inner).

    F_outer is shared with the enclosing RFactor forest (``iv_f_outer``);
    F_inner replaces the block's F iter var at the block level.
    Accesses that referenced the F iter var expand via
    ``f_outer * inner_f_trip + f_inner``.
    """
    orig_block = _find_inner_block(root)
    old_ivs = _forest_iter_vars(root)
    iv_map: dict[int, IterVar] = {}
    old_f_iv_id: int | None = None
    f_inner: IterVar | None = None
    new_outer_ivs: list[IterVar] = []
    for iv in old_ivs:
        if iv.dim_id == f_dim:
            old_f_iv_id = iv.var_id
            f_inner = module.allocate_iter_var(dim_id=iv.dim_id, extent=inner_f_trip, role=AxisRole.ACCUMULATION)
            iv_map[iv.var_id] = f_inner
        else:
            fresh = module.allocate_iter_var(dim_id=iv.dim_id, extent=iv.extent, role=iv.role)
            iv_map[iv.var_id] = fresh
            new_outer_ivs.append(fresh)

    if old_f_iv_id is None or f_inner is None:
        raise AtomLegalityError("RFactor (slot): activation_reduce tree does not bind the accumulation dim")

    new_block_ivs = [iv_map[iv.var_id] for iv in orig_block.iter_vars]
    new_reads = {
        slot: _rewrite_k_split_access(access, old_f_iv_id, iv_f_outer.var_id, f_inner.var_id, inner_f_trip, iv_map)
        for slot, access in orig_block.reads.items()
    }
    """Redirect writes: dst → scratch_local (F-split expansion); reduce_res →
    partials with an extra outer slot indexed by iv_f_outer (no F in
    reduce_res, so no Split-expansion there)."""
    new_writes: dict[str, BufferAccess] = {}
    for slot, access in orig_block.writes.items():
        if access.tensor_name == scratch_name:
            """``dst`` mirrors ``data`` axis-wise (both ``(P, F)``) — F-split."""
            expanded = _rewrite_k_split_access(
                access, old_f_iv_id, iv_f_outer.var_id, f_inner.var_id, inner_f_trip, iv_map
            )
            new_writes[slot] = BufferAccess(
                tensor_name=local_name, iter_var_ids=expanded.iter_var_ids, pattern=expanded.pattern
            )
        elif access.tensor_name == sum_acc_name:
            """``reduce_res`` shape = (P,); partials = (P, outer)."""
            remapped = _remap_pattern_ivs(access.pattern, iv_map)
            outer_ar = AccessRange.make({iv_f_outer.var_id: 1}, 0, 1)
            new_pattern = (*remapped, outer_ar)
            orig_ids = tuple(_remap_iv_ids(access.iter_var_ids, iv_map))
            new_ids = (*orig_ids, iv_f_outer.var_id)
            new_writes[slot] = BufferAccess(tensor_name=partials_name, iter_var_ids=new_ids, pattern=new_pattern)
        else:
            new_writes[slot] = _remap_access_ivs(access, iv_map)

    new_block = SBlock(
        iter_vars=new_block_ivs,
        reads=new_reads,
        writes=new_writes,
        reads_writes={},
        body=list(orig_block.body),
        annotations=dict(orig_block.annotations),
    )
    ordered_new_ivs = [iv_map[iv.var_id] for iv in old_ivs]
    node: ForNode | SBlock = new_block
    for iv in reversed(ordered_new_ivs):
        node = ForNode(iter_var=iv, children=[node])
    return node


def _build_close_reduce_slot_tree(
    module: KernelModule,
    orig_ar_block: SBlock,
    partials_name: str,
    sum_acc_name: str,
    outer_dim_id: str,
    new_dims: dict[str, DimInfo],
) -> ForNode | SBlock:
    """Build the close-reduce tree for slot: iterate only the P dim (shared w/ sum_acc), reduce outer axis of partials."""
    sum_acc = module.tensors[sum_acc_name]
    p_dim = sum_acc.dim_ids[0]

    p_trip = _block_dim_trip(orig_ar_block, p_dim)
    p_tile = _block_dim_tile(module, orig_ar_block, p_dim)
    if p_trip is None or p_tile is None:
        raise AtomLegalityError("RFactor.apply (slot): activation_reduce block has no iter var/access on P dim")

    iv_p = module.allocate_iter_var(dim_id=p_dim, extent=p_trip, role=AxisRole.PARALLEL)
    outer_info = new_dims[outer_dim_id]
    outer_ar = AccessRange.make({}, 0, outer_info.total_size)

    """sum_acc's logical shape is 1D (P,) — pattern has one entry."""
    dst_pattern = (AccessRange.make({iv_p.var_id: 1}, 0, p_tile),)
    dst_access = BufferAccess(tensor_name=sum_acc_name, iter_var_ids=(iv_p.var_id,), pattern=dst_pattern)

    partials_pattern = (AccessRange.make({iv_p.var_id: 1}, 0, p_tile), outer_ar)
    data_access = BufferAccess(tensor_name=partials_name, iter_var_ids=(iv_p.var_id,), pattern=partials_pattern)

    """partials has logical shape ``(P, outer)`` — 2D; place_buffers promotes
    to physical ``(P_tile, num_P_slots, F_tile*num_F_tiles)`` with outer
    as the F axis. The close-reduce block slices
    ``partials[0:P_tile, p_slot, 0:outer]`` giving a 2D ``(P_tile, outer)``
    tile where outer is the last axis — reduce axis=1."""
    reduce_axis = 1

    call = NKIOpCall(op_cls=NKITensorReduce, kwargs={"axis": reduce_axis, "op": "add"}, axis_map={}, dim_role={})
    block = SBlock(
        iter_vars=[iv_p], reads={"data": data_access}, writes={"dst": dst_access}, reads_writes={}, body=[call]
    )
    return _wrap_in_fornodes(block, [iv_p])
