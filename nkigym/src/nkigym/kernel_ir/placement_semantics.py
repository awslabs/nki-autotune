"""Semantics of ``BufferPlacement`` within a fusion group's loop nest.

Given a group's ``dim_order`` and a buffer's own ``dim_ids``, one
``BufferPlacement`` value determines:

* the alloc depth — where in the pair-interleaved ``before/after``
  slot sequence the buffer's ``nl.ndarray(...)`` declaration goes;
* per-dim ``loop_open`` state — whether the block and ltile loops of
  each buffer dim are open inside the alloc scope (used to derive
  list-axis extents and access patterns).

The 3 placements are::

    OUTER:  alloc at depth 0  — NEITHER block loop is open
    MIDDLE: alloc at block_depth(d_outer_pos) + 1 — only d_outer's is
    INNER:  alloc at block_depth(d_inner_pos) + 1 — BOTH are

For a 1-D buffer, ``MIDDLE`` collapses to ``INNER``. Dims in
``dim_order`` not present in the buffer's ``dim_ids`` are "irrelevant":
placement is hoisted outside them.
"""

from nkigym.kernel_ir.fusion_group import BufferPlacement


def buffer_dim_positions(dim_ids: tuple[str, ...], dim_order: list[str]) -> list[int]:
    """Return the buffer dims' positions in ``dim_order``, in ascending order.

    Dims not in ``dim_order`` are dropped (they can't participate in
    placement — the group's loop nest doesn't iterate them).
    """
    positions = [dim_order.index(d) for d in dim_ids if d in dim_order]
    positions.sort()
    return positions


def alloc_depth(placement: BufferPlacement, dim_positions: list[int]) -> int:
    """Return the slot depth at which the buffer should be declared.

    The depth uses the pair-interleaved convention:
    ``block_depth(pos) = 2*pos`` is the slot BEFORE dim ``pos``'s
    block loop opens; alloc at ``block_depth(pos) + 1`` puts the
    declaration INSIDE the block loop.

    For a 1-D buffer ``MIDDLE`` collapses to ``INNER``.
    """
    if not dim_positions:
        return 0
    if placement is BufferPlacement.OUTER:
        depth = 0
    elif placement is BufferPlacement.MIDDLE:
        depth = 2 * dim_positions[0] + 1
    elif placement is BufferPlacement.INNER:
        depth = 2 * dim_positions[-1] + 1
    else:
        raise ValueError(f"unknown BufferPlacement {placement!r}")
    return depth


def block_loop_open(placement: BufferPlacement, dim_position: int, dim_positions: list[int]) -> bool:
    """True iff the block loop of dim at ``dim_position`` is open inside alloc scope.

    A buffer's dim is "open" iff placement puts the alloc INSIDE
    that dim's block loop. That happens when:

    * ``INNER``: every buffer dim's block loop is open.
    * ``MIDDLE``: only the outermost buffer dim's block loop is open.
    * ``OUTER``: none.
    """
    if not dim_positions or dim_position not in dim_positions:
        return False
    if placement is BufferPlacement.OUTER:
        is_open = False
    elif placement is BufferPlacement.MIDDLE:
        is_open = dim_position == dim_positions[0]
    elif placement is BufferPlacement.INNER:
        is_open = True
    else:
        raise ValueError(f"unknown BufferPlacement {placement!r}")
    return is_open


def ltile_loop_open(placement: BufferPlacement, dim_position: int, dim_positions: list[int]) -> bool:
    """True iff the ltile loop of dim at ``dim_position`` is open inside alloc scope.

    Under the 3-way placement scheme placements are ALWAYS at
    block-loop granularity, never inside an ltile loop. So the
    ltile loop of any buffer dim is closed at alloc — the list
    level retains ``ltiles_per_block`` entries on that axis.
    """
    _ = placement, dim_position, dim_positions
    return False


def effective_placement(ir: object, group_idx: int, tensor_name: str) -> BufferPlacement:
    """Return the ``BufferPlacement`` to use for ``tensor_name`` in group ``group_idx``.

    * If the group explicitly stores a placement for this tensor
      (Load-destination physical_buffer that is an external input
      to the FG), return that.
    * If the tensor is touched by 2+ FGs (cross-group), force
      ``OUTER`` so its data persists across the boundary.
    * Otherwise derive from the writing ops' reduction dims:
      tightest placement that allocates OUTSIDE every reduction
      block loop of any op writing this tensor in this group.
    """
    from nkigym.kernel_ir.fusion_group import FusionGroup

    group: FusionGroup = ir.groups[group_idx]
    key = ("sbuf", tensor_name)
    if key in group.buffer_placements:
        return group.buffer_placements[key]
    if _touched_by_multiple_groups(ir, tensor_name):
        return BufferPlacement.OUTER
    tinfo = ir.tensor_info(tensor_name)
    positions = buffer_dim_positions(tinfo.dim_ids, group.dim_order)
    reduction_positions = _writing_ops_reduction_positions(ir, group_idx, tensor_name)
    return derive_placement(positions, reduction_positions)


def _touched_by_multiple_groups(ir: object, tensor_name: str) -> bool:
    """True iff ``tensor_name`` is referenced by ops in 2+ fusion groups."""
    touching: set[int] = set()
    for gi, group in enumerate(ir.groups):
        for op in group.ops:
            names = [*ir.op_inputs.get(op, {}).values(), *ir.op_outputs.get(op, [])]
            if tensor_name in names:
                touching.add(gi)
                break
        if len(touching) >= 2:
            break
    return len(touching) >= 2


def _writing_ops_reduction_positions(ir: object, group_idx: int, tensor_name: str) -> set[int]:
    """Positions (in group's ``dim_order``) of dims any writing op reduces over.

    A reduction dim of a writing op is a blocking dim the op
    doesn't carry to this output tensor — e.g. matmul's K dim
    is blocking on the op but absent from its output.
    """
    group = ir.groups[group_idx]
    dim_order = group.dim_order
    tinfo = ir.tensor_info(tensor_name)
    output_dims = set(tinfo.dim_ids)
    positions: set[int] = set()
    for op in group.ops:
        if tensor_name not in ir.op_outputs.get(op, []):
            continue
        for dim_id in ir.op_blocking_dims.get(op, set()):
            if dim_id in output_dims:
                continue
            if dim_id in dim_order:
                positions.add(dim_order.index(dim_id))
    return positions


def derive_placement(dim_positions: list[int], reduction_positions: set[int]) -> BufferPlacement:
    """Derive a ``BufferPlacement`` as the TIGHTEST placement that keeps the
    buffer allocated OUTSIDE every reduction-dim block loop.

    A buffer that's written across multiple iterations of a
    reduction dim (e.g. a matmul output accumulated across K)
    must be allocated OUTSIDE that reduction dim's block loop so
    it persists. Within that constraint, tightest placement =
    smallest buffer = best SBUF residency.

    ``dim_positions``: buffer's own dim positions in ``dim_order``.
    ``reduction_positions``: positions of dims the writing ops
    reduce over (typically ``op_blocking_dims`` that aren't in
    the buffer's output).

    Tightest valid choice (INNER > MIDDLE > OUTER by tightness):

    * ``INNER`` if every reduction position is outside all buffer dims.
    * ``MIDDLE`` if every reduction position is outside the buffer's
      inner dim (reductions may be inside the outer).
    * ``OUTER`` otherwise.
    """
    if not dim_positions:
        return BufferPlacement.INNER
    if not reduction_positions:
        return BufferPlacement.INNER
    outer = dim_positions[0]
    inner = dim_positions[-1]
    if all(r > inner for r in reduction_positions):
        result = BufferPlacement.INNER
    elif all(r > outer for r in reduction_positions):
        result = BufferPlacement.MIDDLE
    else:
        result = BufferPlacement.OUTER
    return result
