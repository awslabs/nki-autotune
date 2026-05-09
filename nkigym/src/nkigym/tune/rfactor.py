"""RFactor rewrite — fission a reducer into staging-buffer decomposition.

Takes a reducer leaf (matmul or activation_reduce) and an outer factor;
emits a staging buffer plus either a per-outer-iteration PSUM accumulator
(recipe "rmw") or slot-indexed writes (recipe "slot"), closed by a
tensor_reduce over the outer axis.

See ``docs/superpowers/specs/2026-05-09-first-class-buffers-and-rfactor-design.md``.
"""

from dataclasses import dataclass

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, resolve_node
from nkigym.ops.base import AxisRole
from nkigym.tune import AtomLegalityError


@dataclass(frozen=True)
class RFactor:
    """Fission a reducer into outer-split + staging + close.

    Attributes:
        reducer_leaf_path: Path to the reducer's ``BodyLeaf``
            (``NKIMatmul`` for recipe "rmw", ``NKIActivationReduce`` for
            recipe "slot").
        outer_factor: The outer loop's trip count post-split. Must divide
            the accumulation dim's ``num_tiles``, and be strictly between
            1 and ``num_tiles``.
    """

    reducer_leaf_path: tuple[int, ...]
    outer_factor: int

    def is_legal(self, module: KernelModule) -> bool:
        """Check structural + dataflow preconditions."""
        leaf = resolve_node(module.body, self.reducer_leaf_path)
        if not isinstance(leaf, BodyLeaf):
            return False
        recipe = leaf.op_cls.RFACTOR_RECIPE
        if recipe is None:
            return False
        acc_dim = _accumulation_dim(leaf, recipe)
        if acc_dim is None:
            return False
        num_t = module.dims[acc_dim].num_tiles
        if num_t <= 1:
            return False
        if self.outer_factor <= 1 or self.outer_factor >= num_t:
            return False
        if num_t % self.outer_factor != 0:
            return False
        if recipe == "rmw":
            return _is_legal_rmw(module, self.reducer_leaf_path, leaf)
        if recipe == "slot":
            return _is_legal_slot(module, self.reducer_leaf_path, leaf)
        return False

    def apply(self, module: KernelModule) -> KernelModule:
        """Apply the recipe-specific rewrite; rename canonical loop vars."""
        if not self.is_legal(module):
            raise AtomLegalityError(f"RFactor.apply: atom illegal on current state — {self}")
        leaf = resolve_node(module.body, self.reducer_leaf_path)
        assert isinstance(leaf, BodyLeaf)
        recipe = leaf.op_cls.RFACTOR_RECIPE
        if recipe == "rmw":
            return _apply_rmw(module, self.reducer_leaf_path, self.outer_factor)
        if recipe == "slot":
            return _apply_slot(module, self.reducer_leaf_path, self.outer_factor)
        raise AtomLegalityError(f"RFactor.apply: unknown recipe {recipe!r}")


def _accumulation_dim(leaf: BodyLeaf, recipe: str) -> str | None:
    """Find the reducer's accumulation dim id from its axis_map + AXIS_ROLES."""
    _ = recipe
    abstract_roles = leaf.op_cls.AXIS_ROLES
    for abstract, role in abstract_roles.items():
        if role == AxisRole.ACCUMULATION and abstract in leaf.axis_map:
            return leaf.axis_map[abstract]
    return None


def _is_legal_rmw(module: KernelModule, leaf_path: tuple[int, ...], leaf: BodyLeaf) -> bool:
    """Recipe "rmw": accumulation dim's loop must have ACCUMULATION role; dst must be RMW."""
    if not leaf.reads_writes:
        return False
    acc_dim = _accumulation_dim(leaf, "rmw")
    if acc_dim is None:
        return False
    """Find the loop ancestor corresponding to the accumulation dim and check its role."""
    for i in range(len(leaf_path)):
        ancestor = resolve_node(module.body, leaf_path[: i + 1])
        if isinstance(ancestor, LoopNode) and ancestor.dim_id == acc_dim:
            return ancestor.role == AxisRole.ACCUMULATION
    return False


def _is_legal_slot(module: KernelModule, leaf_path: tuple[int, ...], leaf: BodyLeaf) -> bool:
    """Recipe "slot": reduction axis must be in leaf.axis_map with ACCUMULATION role."""
    _ = module, leaf_path
    return _accumulation_dim(leaf, "slot") is not None


def _apply_rmw(module: KernelModule, leaf_path: tuple[int, ...], outer_factor: int) -> KernelModule:
    """Recipe "rmw": matmul-style.

    Transforms:
        [..., NKIAlloc(psum_acc), NKIMemset(psum_acc), K-loop { NKIMatmul(dst=psum_acc) },
         NKITensorCopy(psum_acc → sbuf_prod), ...]
    into:
        [..., NKIAlloc(psum_partials),
              K_outer_loop {
                  NKIAlloc(psum_acc_local), NKIMemset(psum_acc_local),
                  K_inner_loop { NKIMatmul(dst=psum_acc_local) },
                  NKITensorCopy(psum_acc_local → psum_partials[K_outer])
              },
              NKITensorReduce(psum_partials → sbuf_prod, axis=K_outer),
         ...]
    """
    from dataclasses import replace as dc_replace

    from nkigym.codegen.ir import DimInfo, Tensor
    from nkigym.tune.compute_at import _rename_canonical

    matmul_leaf = resolve_node(module.body, leaf_path)
    assert isinstance(matmul_leaf, BodyLeaf)
    psum_acc_name = matmul_leaf.reads_writes[0]
    psum_acc = module.tensors[psum_acc_name]
    k_dim = _accumulation_dim(matmul_leaf, "rmw")
    assert k_dim is not None
    k_info = module.dims[k_dim]
    inner_trip = k_info.num_tiles // outer_factor

    """Introduce a new dim for the outer split."""
    outer_dim_id = _fresh_dim_id(module.dims)
    module_dims = dict(module.dims)
    module_dims[outer_dim_id] = DimInfo(
        dim_id=outer_dim_id, total_size=outer_factor, tile_size=1, num_tiles=outer_factor
    )
    module_dims[k_dim] = dc_replace(k_info, num_tiles=inner_trip)

    """New tensors: psum_partials (SBUF slot vector), psum_acc_local (per-outer PSUM)."""
    partials_name = "psum_partials"
    local_name = "psum_acc_local"
    partials = Tensor(
        name=partials_name,
        dim_ids=psum_acc.dim_ids + (outer_dim_id,),
        shape=psum_acc.shape + (outer_factor,),
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
    new_tensors = {k: v for k, v in module.tensors.items() if k != psum_acc_name}
    new_tensors[partials_name] = partials
    new_tensors[local_name] = acc_local

    """Find the existing memset + tensor_copy siblings and remove them
    from the forest. We rebuild the forest by walking and replacing."""
    new_body = _rebuild_forest_rmw(
        module.body,
        leaf_path,
        psum_acc_name,
        outer_dim_id,
        outer_factor,
        inner_trip,
        k_dim,
        k_info.num_tiles,
        partials_name,
        local_name,
    )
    new_body = _rename_canonical(new_body)

    return dc_replace(module, tensors=new_tensors, dims=module_dims, body=new_body)


def _fresh_dim_id(dims: dict) -> str:
    """Pick a dim_id not yet in use: ``d<N>`` for the smallest available N."""
    taken = set(dims.keys())
    i = 0
    while f"d{i}" in taken:
        i += 1
    return f"d{i}"


def _rebuild_forest_rmw(
    body,
    leaf_path,
    psum_acc_name,
    outer_dim_id,
    outer_factor,
    inner_trip,
    k_dim,
    _original_k_trips,
    partials_name,
    local_name,
):
    """Surgical rewrite of the forest to the rmw-rfactor shape.

    The high-level steps:
    1. Walk the forest; find every node touched by the matmul's subtree.
    2. Remove the psum_acc memset root (may be a LoopNode wrapping the memset leaf).
    3. Remove the NKITensorCopy root that drains psum_acc → sbuf_prod
       (the old drain target). Record its dst tensor name.
    4. Locate the K-loop LoopNode (the leaf_path's parent).
    5. Replace the K-loop with: K_outer { NKIAlloc(local), NKIMemset(local),
       K_inner { matmul(dst=local) }, NKITensorCopy(local → partials[outer]) }.
    6. Prepend NKIAlloc(partials) before the K_outer loop.
    7. Append NKITensorReduce(partials → original_drain_dst, axis=outer) after K_outer.
    """
    from nkigym.codegen.ir import BodyLeaf, LoopNode
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.tensor_copy import NKITensorCopy
    from nkigym.ops.tensor_reduce import NKITensorReduce

    """Scan the forest roots to locate the relevant subtrees."""
    original_drain_dst = None
    alloc_root_idx = None
    memset_root_idx = None
    drain_root_idx = None

    def find_leaf_in_tree(node, target_op_cls, target_field, target_value):
        """Recursively search for a leaf matching the given criteria."""
        if isinstance(node, BodyLeaf):
            if node.op_cls is target_op_cls:
                if target_field == "writes" and target_value in node.writes:
                    return node
                if target_field == "reads.src" and node.reads.get("src") == target_value:
                    return node
        elif isinstance(node, LoopNode):
            for child in node.children:
                result = find_leaf_in_tree(child, target_op_cls, target_field, target_value)
                if result:
                    return result
        return None

    for i, root in enumerate(body):
        if find_leaf_in_tree(root, NKIAlloc, "writes", psum_acc_name):
            alloc_root_idx = i
        if find_leaf_in_tree(root, NKIMemset, "writes", psum_acc_name):
            memset_root_idx = i
        leaf = find_leaf_in_tree(root, NKITensorCopy, "reads.src", psum_acc_name)
        if leaf:
            drain_root_idx = i
            original_drain_dst = leaf.writes[0]

    if alloc_root_idx is None or memset_root_idx is None or drain_root_idx is None or original_drain_dst is None:
        raise AtomLegalityError(
            "RFactor.apply (rmw): expected sibling NKIAlloc + NKIMemset + NKITensorCopy for psum_acc"
        )

    """Locate the matmul leaf + its K loop parent path at forest root level.
    For the canonical form, the K loop is a root in the forest."""
    matmul_root_idx = leaf_path[0]
    matmul_root = body[matmul_root_idx]
    if not isinstance(matmul_root, LoopNode) or matmul_root.dim_id != k_dim:
        raise AtomLegalityError("RFactor.apply (rmw): expected K-loop LoopNode at forest root as matmul ancestor")

    """Navigate down the leaf_path to find the matmul leaf and update it."""

    def clone_subtree_with_replacement(node, path_from_here, new_dst):
        """Recursively clone the subtree, replacing the matmul leaf's dst."""
        if not path_from_here:
            """We've reached the matmul leaf."""
            assert isinstance(node, BodyLeaf) and node.op_cls is NKIMatmul
            return BodyLeaf(
                op_cls=NKIMatmul,
                reads=dict(node.reads),
                writes=(),
                reads_writes=(new_dst,),
                kwargs=dict(node.kwargs),
                axis_map=dict(node.axis_map),
                dim_role=dict(node.dim_role),
            )
        """We're at an intermediate LoopNode; clone it with the updated child."""
        assert isinstance(node, LoopNode)
        child_idx = path_from_here[0]
        new_children = []
        for i, c in enumerate(node.children):
            if i == child_idx:
                new_children.append(clone_subtree_with_replacement(c, path_from_here[1:], new_dst))
            else:
                new_children.append(c)
        return LoopNode(
            dim_id=node.dim_id,
            trip_count=node.trip_count,
            role=node.role,
            children=new_children,
            reduce_op=node.reduce_op,
            name=node.name,
            pipeline_depth=node.pipeline_depth,
        )

    """Clone the subtree under the K-loop, replacing psum_acc with psum_acc_local."""
    k_inner_subtree = clone_subtree_with_replacement(matmul_root, leaf_path[1:], local_name)
    """k_inner_subtree is now a LoopNode with dim_id=k_dim and the modified matmul."""
    k_inner = LoopNode(
        dim_id=k_dim,
        trip_count=inner_trip,
        role=AxisRole.ACCUMULATION,
        children=k_inner_subtree.children,
        reduce_op=k_inner_subtree.reduce_op,
        name=k_inner_subtree.name,
        pipeline_depth=k_inner_subtree.pipeline_depth,
    )

    """Clone the memset loop structure, replacing psum_acc with psum_acc_local."""
    memset_root = body[memset_root_idx]

    def clone_memset_tree(node, old_name, new_name):
        """Recursively clone, replacing old_name with new_name in writes."""
        if isinstance(node, BodyLeaf):
            if node.op_cls is NKIMemset and old_name in node.writes:
                return BodyLeaf(
                    op_cls=NKIMemset,
                    reads={},
                    writes=(new_name,),
                    reads_writes=(),
                    kwargs=dict(node.kwargs),
                    axis_map=dict(node.axis_map),
                    dim_role=dict(node.dim_role),
                )
            return node
        assert isinstance(node, LoopNode)
        new_children = [clone_memset_tree(c, old_name, new_name) for c in node.children]
        return LoopNode(
            dim_id=node.dim_id,
            trip_count=node.trip_count,
            role=node.role,
            children=new_children,
            reduce_op=node.reduce_op,
            name=node.name,
            pipeline_depth=node.pipeline_depth,
        )

    memset_local_tree = clone_memset_tree(memset_root, psum_acc_name, local_name)

    """Clone the drain loop structure, replacing psum_acc → psum_partials."""
    drain_root = body[drain_root_idx]

    def clone_drain_tree(node, old_src, new_src, old_dst, new_dst):
        """Recursively clone, replacing old_src/dst with new_src/dst."""
        if isinstance(node, BodyLeaf):
            if node.op_cls is NKITensorCopy and node.reads.get("src") == old_src and old_dst in node.writes:
                return BodyLeaf(
                    op_cls=NKITensorCopy,
                    reads={"src": new_src},
                    writes=(new_dst,),
                    reads_writes=(),
                    kwargs=dict(node.kwargs),
                    axis_map=dict(node.axis_map),
                    dim_role=dict(node.dim_role),
                )
            return node
        assert isinstance(node, LoopNode)
        new_children = [clone_drain_tree(c, old_src, new_src, old_dst, new_dst) for c in node.children]
        return LoopNode(
            dim_id=node.dim_id,
            trip_count=node.trip_count,
            role=node.role,
            children=new_children,
            reduce_op=node.reduce_op,
            name=node.name,
            pipeline_depth=node.pipeline_depth,
        )

    drain_local_tree = clone_drain_tree(drain_root, psum_acc_name, local_name, original_drain_dst, partials_name)

    alloc_local = BodyLeaf(
        op_cls=NKIAlloc,
        reads={},
        writes=(local_name,),
        reads_writes=(),
        kwargs={"tensor_name": local_name},
        axis_map={},
        dim_role={},
    )

    k_outer = LoopNode(
        dim_id=outer_dim_id,
        trip_count=outer_factor,
        role=AxisRole.PARALLEL,
        children=[alloc_local, memset_local_tree, k_inner, drain_local_tree],
    )
    alloc_partials = BodyLeaf(
        op_cls=NKIAlloc,
        reads={},
        writes=(partials_name,),
        reads_writes=(),
        kwargs={"tensor_name": partials_name},
        axis_map={},
        dim_role={},
    )

    """Clone the drain loop structure for the closing tensor_reduce."""

    def clone_drain_for_reduce(node, partials_name, original_dst):
        """Recursively clone drain structure, replacing TensorCopy with TensorReduce."""
        if isinstance(node, BodyLeaf):
            if node.op_cls is NKITensorCopy:
                return BodyLeaf(
                    op_cls=NKITensorReduce,
                    reads={"data": partials_name},
                    writes=(original_dst,),
                    reads_writes=(),
                    kwargs={"axis": 2, "op": "add"},
                    axis_map=dict(node.axis_map),
                    dim_role=dict(node.dim_role),
                )
            return node
        assert isinstance(node, LoopNode)
        new_children = [clone_drain_for_reduce(c, partials_name, original_dst) for c in node.children]
        return LoopNode(
            dim_id=node.dim_id,
            trip_count=node.trip_count,
            role=node.role,
            children=new_children,
            reduce_op=node.reduce_op,
            name=node.name,
            pipeline_depth=node.pipeline_depth,
        )

    close_reduce_tree = clone_drain_for_reduce(body[drain_root_idx], partials_name, original_drain_dst)

    """Rebuild the forest: remove alloc, memset, matmul-K-loop, drain at their
    original indices; insert alloc_partials + k_outer + close_reduce_tree in place
    of the K-loop's slot."""
    indices_to_remove = {alloc_root_idx, memset_root_idx, matmul_root_idx, drain_root_idx}
    new_body = []
    inserted = False
    for i, root in enumerate(body):
        if i in indices_to_remove:
            if i == matmul_root_idx and not inserted:
                new_body.append(alloc_partials)
                new_body.append(k_outer)
                new_body.append(close_reduce_tree)
                inserted = True
            continue
        new_body.append(root)
    return new_body


def _apply_slot(module: KernelModule, leaf_path: tuple[int, ...], outer_factor: int) -> KernelModule:
    """Recipe "slot": activation_reduce-style.

    Transforms:
        [..., NKIAlloc(sum_acc), NKIAlloc(scratch),
              F-LoopNode(F_orig, ACC){ NKIActivationReduce(dst=scratch, reduce_res=sum_acc) },
         ...]
    into:
        [..., NKIAlloc(sum_acc),          # unchanged
              NKIAlloc(partials),         # (P, outer_factor) staging
              NKIAlloc(scratch_local),    # (P, F_inner) per-outer scratch
              F_outer_loop(outer_factor, PARALLEL){
                  F_inner_loop(F_inner, ACC){
                      NKIActivationReduce(data, dst=scratch_local, reduce_res=partials[F_outer])
                  }
              },
              NKITensorReduce(partials → sum_acc, axis=1),
         ...]
    """
    from dataclasses import replace as dc_replace

    from nkigym.codegen.ir import DimInfo, Tensor
    from nkigym.tune.compute_at import _rename_canonical

    ar_leaf = resolve_node(module.body, leaf_path)
    assert isinstance(ar_leaf, BodyLeaf)
    f_dim = _accumulation_dim(ar_leaf, "slot")
    assert f_dim is not None
    f_info = module.dims[f_dim]
    inner_f_trip = f_info.num_tiles // outer_factor

    """Identify operand names: dst (scratch, to be replaced by scratch_local)
    and reduce_res (original sum_acc, final output of the rfactored reduction)."""
    """writes tuple order mirrors OPERAND_AXES insertion: (dst, reduce_res)."""
    scratch_name = ar_leaf.writes[0]
    sum_acc_name = ar_leaf.writes[1]
    sum_acc = module.tensors[sum_acc_name]
    scratch = module.tensors[scratch_name]

    """Introduce a new dim for the outer split."""
    outer_dim_id = _fresh_dim_id(module.dims)
    module_dims = dict(module.dims)
    module_dims[outer_dim_id] = DimInfo(
        dim_id=outer_dim_id, total_size=outer_factor, tile_size=1, num_tiles=outer_factor
    )
    module_dims[f_dim] = dc_replace(f_info, num_tiles=inner_f_trip)

    partials_name = "partials"
    local_name = "scratch_local"
    """partials shape: (sum_acc.shape..., outer_factor). For 1D sum_acc (P,),
    partials is 2D (P, outer_factor). dim_ids inherit P dim from sum_acc plus
    the new outer_dim."""
    partials = Tensor(
        name=partials_name,
        dim_ids=sum_acc.dim_ids + (outer_dim_id,),
        shape=sum_acc.shape + (outer_factor,),
        dtype=sum_acc.dtype,
        origin="intermediate",
        location="sbuf",
    )
    scratch_local = Tensor(
        name=local_name,
        dim_ids=scratch.dim_ids,
        shape=(scratch.shape[0], scratch.shape[1] // outer_factor),
        dtype=scratch.dtype,
        origin="intermediate",
        location="sbuf",
    )
    new_tensors = {k: v for k, v in module.tensors.items() if k != scratch_name}
    new_tensors[partials_name] = partials
    new_tensors[local_name] = scratch_local

    """Rebuild the forest: replace the F-LoopNode + activation_reduce with
    the new F_outer→F_inner→modified_ar structure plus closing TensorReduce.
    Also: remove the original `scratch` NKIAlloc (replaced by scratch_local);
    add NKIAlloc entries for partials and scratch_local."""
    new_body = _rebuild_forest_slot(
        module.body,
        leaf_path,
        ar_leaf,
        scratch_name,
        sum_acc_name,
        outer_dim_id,
        outer_factor,
        inner_f_trip,
        f_dim,
        partials_name,
        local_name,
    )
    new_body = _rename_canonical(new_body)

    return dc_replace(module, tensors=new_tensors, dims=module_dims, body=new_body)


def _rebuild_forest_slot(
    body,
    leaf_path,
    ar_leaf,
    scratch_name,
    sum_acc_name,
    outer_dim_id,
    outer_factor,
    inner_f_trip,
    f_dim,
    partials_name,
    local_name,
):
    """Surgical rewrite for the slot recipe.

    Steps:
    1. Locate and remove the scratch_name NKIAlloc leaf (forest root).
    2. Locate the F-LoopNode holding the activation_reduce leaf.
    3. Replace the F-LoopNode with:
       - NKIAlloc(partials_name), NKIAlloc(local_name) before
       - F_outer(outer_factor) { F_inner(inner_f_trip) { modified_ar } }
       - NKITensorReduce(partials → sum_acc_name, axis=1) after
    The `modified_ar` has:
       - dst rewritten to local_name
       - reduce_res rewritten to partials_name
       - dim_role updated (F becomes ACC but under F_inner; outer is PARALLEL)
    """
    from nkigym.codegen.ir import BodyLeaf, LoopNode
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.tensor_reduce import NKITensorReduce

    """Find the root index containing the activation_reduce leaf's F-LoopNode ancestor."""
    ar_root_idx = leaf_path[0]
    ar_root = body[ar_root_idx]
    if not isinstance(ar_root, LoopNode) or ar_root.dim_id != f_dim:
        raise AtomLegalityError("RFactor.apply (slot): expected F-LoopNode at forest root containing activation_reduce")

    """Find and remove the scratch NKIAlloc at forest root level."""
    scratch_root_idx = None
    for i, root in enumerate(body):
        if isinstance(root, BodyLeaf) and root.op_cls is NKIAlloc and scratch_name in root.writes:
            scratch_root_idx = i
            break

    """Build the modified activation_reduce leaf with new operand names."""
    new_ar_leaf = BodyLeaf(
        op_cls=NKIActivationReduce,
        reads=dict(ar_leaf.reads),
        writes=(local_name, partials_name),
        reads_writes=ar_leaf.reads_writes,
        kwargs=dict(ar_leaf.kwargs),
        axis_map=dict(ar_leaf.axis_map),
        dim_role=dict(ar_leaf.dim_role),
    )

    """Clone the subtree under F-loop (excluding the F-loop itself), replacing ar_leaf."""

    def clone_subtree(node, path_from_here, replacement_leaf):
        """Recursively clone, replacing the target leaf."""
        if not path_from_here:
            """Reached the target leaf."""
            return replacement_leaf
        assert isinstance(node, LoopNode)
        child_idx = path_from_here[0]
        new_children = []
        for i, c in enumerate(node.children):
            if i == child_idx:
                new_children.append(clone_subtree(c, path_from_here[1:], replacement_leaf))
            else:
                new_children.append(c)
        return LoopNode(
            dim_id=node.dim_id,
            trip_count=node.trip_count,
            role=node.role,
            children=new_children,
            reduce_op=node.reduce_op,
            name=node.name,
            pipeline_depth=node.pipeline_depth,
        )

    """Clone the F-loop's children, replacing the ar_leaf."""
    cloned_children = []
    for i, child in enumerate(ar_root.children):
        if i == leaf_path[1]:
            cloned_children.append(clone_subtree(child, leaf_path[2:], new_ar_leaf))
        else:
            cloned_children.append(child)

    """Build inner F loop (trip = inner_f_trip, still ACCUMULATION role)."""
    f_inner = LoopNode(dim_id=f_dim, trip_count=inner_f_trip, role=AxisRole.ACCUMULATION, children=cloned_children)
    """Build outer F loop (trip = outer_factor, PARALLEL role)."""
    f_outer = LoopNode(dim_id=outer_dim_id, trip_count=outer_factor, role=AxisRole.PARALLEL, children=[f_inner])
    alloc_partials = BodyLeaf(
        op_cls=NKIAlloc,
        reads={},
        writes=(partials_name,),
        kwargs={"tensor_name": partials_name},
        axis_map={},
        dim_role={},
    )
    alloc_local = BodyLeaf(
        op_cls=NKIAlloc, reads={}, writes=(local_name,), kwargs={"tensor_name": local_name}, axis_map={}, dim_role={}
    )
    close_reduce = BodyLeaf(
        op_cls=NKITensorReduce,
        reads={"data": partials_name},
        writes=(sum_acc_name,),
        kwargs={"axis": 1, "op": ar_leaf.kwargs.get("reduce_op", "add")},
        axis_map={},
        dim_role={},
    )

    """Rebuild forest: drop scratch_alloc and f_root; insert alloc_partials +
    alloc_local + f_outer + close_reduce in f_root's slot."""
    new_body = []
    indices_to_remove = {ar_root_idx}
    if scratch_root_idx is not None:
        indices_to_remove.add(scratch_root_idx)
    inserted = False
    for i, root in enumerate(body):
        if i in indices_to_remove:
            if i == ar_root_idx and not inserted:
                new_body.append(alloc_partials)
                new_body.append(alloc_local)
                new_body.append(f_outer)
                new_body.append(close_reduce)
                inserted = True
            continue
        new_body.append(root)
    return new_body


def enumerate_rfactor_atoms(module: KernelModule) -> list[RFactor]:
    """Emit one atom per (reducer leaf, valid divisor of accumulation dim)."""
    atoms: list[RFactor] = []

    def walk(node, path: tuple[int, ...]) -> None:
        if isinstance(node, BodyLeaf):
            if node.op_cls.RFACTOR_RECIPE is not None:
                acc_dim = _accumulation_dim(node, node.op_cls.RFACTOR_RECIPE)
                if acc_dim is not None:
                    num_t = module.dims[acc_dim].num_tiles
                    for factor in _divisors_strict(num_t):
                        atom = RFactor(reducer_leaf_path=path, outer_factor=factor)
                        if atom.is_legal(module):
                            atoms.append(atom)
        else:
            for i, c in enumerate(node.children):
                walk(c, path + (i,))

    for i, root in enumerate(module.body):
        walk(root, (i,))
    return atoms


def _divisors_strict(n: int) -> list[int]:
    """Return every divisor d of n satisfying 1 < d < n."""
    return [d for d in range(2, n) if n % d == 0]
