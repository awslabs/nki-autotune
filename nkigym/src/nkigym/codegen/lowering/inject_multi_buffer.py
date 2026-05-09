"""``InjectMultiBuffer`` pass: slot-index expressions for multi-buffered tiles.

Runs as part of source emission. Body emitters in
:mod:`nkigym.codegen.lowering.emit_ops` consume this module's
functions to build the ``[p_tile, p_slot, f_range]`` style slice
expressions that read and write the 3D SBUF allocations produced by
:mod:`nkigym.codegen.lowering.place_buffers`.

A tensor's per-dim ``total_slots`` is ``required_tiles * buffer_degree``.
When ``total_slots == 1`` the slot collapses to the literal ``"0"``; when
it equals the product of ancestor trip counts, the modulo is identity
and the raw slot is emitted. Otherwise the expression includes
``(raw) % total_slots``. Software pipelining passes a ``stage_offset``
through the body emitter so slot expressions for a pipelined dim
substitute the innermost ancestor with ``(loop_var + stage_offset)``.
"""


def slot_expr(
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    dim_id: str,
    total_slots: int,
    stage_offset: int = 0,
) -> str:
    """Return the slot expression for ``dim_id``.

    For dim ``d`` with ``k`` same-dim ancestors on the current path and
    loop variable names ``path_names[d] = [n_0, n_1, ..., n_{k-1}]``
    (outermost->innermost), the raw slot is ``sum_{idx} n_idx *
    prod_of_tail_trips``. The final expression is
    ``(raw_slot) % total_slots``, with two simplifications:

    * ``total_slots == 1`` — slot is literal ``"0"``.
    * ``total_slots == product_of_ancestor_trips`` (raw slot never
      exceeds ``total_slots``) — modulo is identity; emit the raw slot.

    ``stage_offset`` adds an integer offset to the innermost ancestor
    (used by the software-pipelined body emission in Task 8). Default 0.

    Uses each ancestor's persisted ``LoopNode.name`` so loop identity
    survives structural rewrites — post-swap, the same loop prints the
    same variable name regardless of its tree position.

    Raises:
        ValueError: ``dim_id`` has no open ancestor loops on the path.
    """
    names = path_names.get(dim_id, [])
    k = len(names)
    if k == 0:
        raise ValueError(f"No open LoopNode on path for dim {dim_id!r}")
    if total_slots == 1:
        return "0"
    trips = path_trips[dim_id]
    raw_trip_product = 1
    for t in trips:
        raw_trip_product *= t
    terms: list[str] = []
    for idx in range(k):
        tail_prod = 1
        for t in trips[idx + 1 :]:
            tail_prod *= t
        innermost = idx == k - 1
        if innermost and stage_offset != 0:
            sign = "+" if stage_offset > 0 else "-"
            token = f"({names[idx]} {sign} {abs(stage_offset)})"
        else:
            token = names[idx]
        if tail_prod == 1:
            terms.append(token)
        else:
            terms.append(f"{token} * {tail_prod}")
    raw = " + ".join(terms)
    if total_slots == raw_trip_product and stage_offset == 0:
        return raw
    return f"({raw}) % {total_slots}"


def sbuf_tile_slice(
    name: str,
    dim_ids: tuple[str, ...],
    p_tile: int,
    f_tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    total_slots_p: int,
    total_slots_f: int,
    stage_offset_p: int = 0,
    stage_offset_f: int = 0,
) -> str:
    """Return the SBUF ``[p_tile, p_slot, f_range]`` slice expression."""
    p_axis = dim_ids[0]
    p_slot = slot_expr(path_names, path_trips, p_axis, total_slots_p, stage_offset_p)
    if len(dim_ids) == 1:
        return f"{name}[0:{p_tile}, {p_slot}, 0:1]"
    f_axis = dim_ids[1]
    f_slot_inner = slot_expr(path_names, path_trips, f_axis, total_slots_f, stage_offset_f)
    f_slot = f"({f_slot_inner})"
    return f"{name}[0:{p_tile}, {p_slot}, {f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"


def hbm_tile_slice(
    name: str,
    dim_ids: tuple[str, ...],
    p_tile: int,
    f_tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    total_slots_p: int,
    total_slots_f: int,
    stage_offset_p: int = 0,
    stage_offset_f: int = 0,
) -> str:
    """Return the HBM ``[p_range, f_range]`` slice expression."""
    p_axis = dim_ids[0]
    p_slot_inner = slot_expr(path_names, path_trips, p_axis, total_slots_p, stage_offset_p)
    p_slot = f"({p_slot_inner})"
    if len(dim_ids) == 1:
        return f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}]"
    f_axis = dim_ids[1]
    f_slot_inner = slot_expr(path_names, path_trips, f_axis, total_slots_f, stage_offset_f)
    f_slot = f"({f_slot_inner})"
    return (
        f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}, "
        f"{f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"
    )


def swapped_dst_tile_slice(
    dst_name: str,
    src_p_axis: str,
    src_f_axis: str,
    tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    total_slots_p: int,
    total_slots_f: int,
    stage_offset_dst_p: int = 0,
    stage_offset_dst_f: int = 0,
) -> str:
    """SBUF slice for a transpose's dst tensor (swapped axes).

    The dst's partition slot uses the source's free-axis ordinals; the
    dst's free slot uses the source's partition-axis ordinals. Transpose
    ops enforce square tiles (p_tile == f_tile), so a single ``tile``
    parameter suffices. ``total_slots_p`` is the slot count for the
    dst's partition axis (= src's free axis); ``total_slots_f`` is for
    the dst's free axis (= src's partition axis).

    ``stage_offset_dst_p`` applies to the innermost ancestor of
    ``src_f_axis`` (used by the dst P slot); ``stage_offset_dst_f``
    applies to the innermost ancestor of ``src_p_axis`` (used by the
    dst F slot).
    """
    p_slot = slot_expr(path_names, path_trips, src_f_axis, total_slots_p, stage_offset_dst_p)
    f_slot_inner = slot_expr(path_names, path_trips, src_p_axis, total_slots_f, stage_offset_dst_f)
    f_slot = f"({f_slot_inner})"
    return f"{dst_name}[0:{tile}, {p_slot}, {f_slot} * {tile} : {f_slot} * {tile} + {tile}]"
