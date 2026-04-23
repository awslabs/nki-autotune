"""KernelIR legality checks on the ``(ir, ir)`` shape."""

from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.validate.emission import Placement, block_depth, body_depth, ltile_depth, op_emission_placement
from nkigym.ops.base import NKIOp
from nkigym.ops.matmul import NKIMatmul

_TIER_RANK = {"per_tile": 0, "per_block": 1, "full": 2}


def validate(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> bool:
    """Return True iff every legality rule passes for ``ir``."""
    tensor_to_groups = _build_tensor_to_groups(ir)
    return (
        _check_cross_group_placements(ir, tensor_to_groups)
        and _check_blocking_innermost(ir)
        and _check_placement_feasibility(ir)
        and _check_emission_feasibility(ir, op_to_group, staged)
    )


def _build_tensor_to_groups(ir: KernelIR) -> dict[str, set[int]]:
    """Map tensor names → set of group indices whose ops touch them."""
    result: dict[str, set[int]] = {}
    for gi, group in enumerate(ir.groups):
        for op in group.ops:
            for name in _op_tensors(ir, op):
                if name in ir.logical_tensors:
                    result.setdefault(name, set()).add(gi)
    return result


def _op_tensors(ir: KernelIR, op: NKIOp) -> list[str]:
    """Return inputs + outputs tensor names for an op."""
    return [*ir.op_inputs.get(op, {}).values(), *ir.op_outputs.get(op, [])]


def _check_emission_feasibility(ir: KernelIR, op_to_group: dict[int, int], staged: set[str]) -> bool:
    """Every op must have a legal emission slot."""
    memo: dict[int, Placement] = {}
    return all(
        _placement_ok(ir, op, gi, op_to_group, staged, memo) for gi, group in enumerate(ir.groups) for op in group.ops
    )


def _placement_ok(
    ir: KernelIR, op: NKIOp, gi: int, op_to_group: dict[int, int], staged: set[str], memo: dict[int, Placement]
) -> bool:
    """True iff ``op_emission_placement`` returns a legal slot."""
    ok = True
    try:
        op_emission_placement(ir, op, gi, op_to_group, staged, memo)
    except ValueError:
        ok = False
    return ok


def _check_blocking_innermost(ir: KernelIR) -> bool:
    """Blocking dims must be innermost (after non-blocking) per op."""
    return all(_op_blocking_innermost_ok(ir, gi, op) for gi, group in enumerate(ir.groups) for op in group.ops)


def _op_blocking_innermost_ok(ir: KernelIR, group_idx: int, op: NKIOp) -> bool:
    """Every effective-blocking dim inner to every non-blocking dim this op touches.

    Dims whose ltile iteration is absorbed by a gadget (e.g.
    ``matmul_block`` iterates K / M / N internally) are NOT
    treated as blocking for this check — the surrounding loop
    nest never opens their ltile loop, so their position in
    ``dim_order`` doesn't constrain PSUM accumulation order the
    way a renderer-emitted blocking loop would.
    """
    ir = ir
    blocking = ir.op_blocking_dims.get(op, set()) - _gadget_absorbed_blocking_dims(ir, group_idx)
    dim_order = ir.groups[group_idx].dim_order
    op_dims: set[str] = set()
    for name in _op_tensors(ir, op):
        tinfo = ir.logical_tensors.get(name)
        if tinfo is not None:
            op_dims.update(tinfo.dim_ids)
    op_dims &= set(dim_order)
    blocking_positions = [dim_order.index(d) for d in blocking if d in op_dims]
    non_blocking_positions = [dim_order.index(d) for d in op_dims if d not in blocking]
    return not (blocking_positions and non_blocking_positions and min(blocking_positions) < max(non_blocking_positions))


def _gadget_absorbed_blocking_dims(ir: KernelIR, group_idx: int) -> set[str]:
    """All dims absorbed by a gadget in ``group_idx`` (K, M, N for matmul_block).

    Mirrors ``codegen.matmul_block_detect.gadget_absorbed_dims``
    — named ``_blocking`` for historical reasons; includes
    non-blocking M/N too since all three are absorbed by the
    gadget's internal iteration. Kept local here to avoid a
    ``validate`` → ``codegen`` → ``kernel_ir`` import cycle.
    """
    result: set[str] = set()
    for op in ir.groups[group_idx].ops:
        if _is_matmul_block_candidate(ir, op, group_idx):
            for abstract in ("K", "M", "N"):
                dim_id = ir.op_axis_map.get(op, {}).get(abstract)
                if dim_id is not None:
                    result.add(dim_id)
    return result


def _is_matmul_block_candidate(ir: KernelIR, op: NKIOp, group_idx: int) -> bool:
    """Local copy of the codegen-level matmul_block candidate predicate."""
    ok = type(op) is NKIMatmul
    if ok:
        axis_map = ir.op_axis_map.get(op, {})
        k_dim = axis_map.get("K")
        if k_dim is None:
            ok = False
        else:
            di = ir.dimensions[k_dim]
            tpb = ir.ltiles_per_block.get(k_dim, 1)
            num_blocks = di.dim_size // (tpb * di.logical_tile_size)
            if num_blocks <= 1:
                ok = False
            else:
                ok = (
                    _k_inputs_per_block(ir, op, group_idx, k_dim)
                    and _inputs_have_block_slab(ir, op, group_idx)
                    and _output_has_block_slab(ir, op, group_idx)
                )
    return ok


def _inputs_have_block_slab(ir: KernelIR, op: NKIOp, group_idx: int) -> bool:
    """True iff matmul inputs' free-axis tiers are ``per_block`` or ``full``."""
    inputs = ir.op_inputs.get(op, {})
    placements = ir.groups[group_idx].tensor_placements
    axis_map = ir.op_axis_map.get(op, {})
    result = True
    for role, abstract in (("stationary", "M"), ("moving", "N")):
        tensor = inputs.get(role)
        dim_id = axis_map.get(abstract)
        if tensor is None or dim_id is None:
            result = False
            break
        tier = placements.get(("sbuf", tensor, dim_id), "per_tile")
        if _TIER_RANK[tier] < _TIER_RANK["per_block"]:
            result = False
            break
    return result


def _k_inputs_per_block(ir: KernelIR, op: NKIOp, group_idx: int, k_dim: str) -> bool:
    """True iff both matmul SBUF inputs have ``per_block`` tier on ``k_dim``."""
    inputs = ir.op_inputs.get(op, {})
    placements = ir.groups[group_idx].tensor_placements
    result = True
    for role in ("stationary", "moving"):
        tensor = inputs.get(role)
        tier = placements.get(("sbuf", tensor, k_dim), "per_tile") if tensor else "full"
        if tensor is None or tier != "per_block":
            result = False
            break
    return result


def _output_has_block_slab(ir: KernelIR, op: NKIOp, group_idx: int) -> bool:
    """True iff the matmul's output has ``per_block`` or ``full`` tier on its M and N dims."""
    outputs = ir.op_outputs.get(op, [])
    placements = ir.groups[group_idx].tensor_placements
    axis_map = ir.op_axis_map.get(op, {})
    result = bool(outputs)
    if result:
        out_name = outputs[0]
        for abstract in ("M", "N"):
            dim_id = axis_map.get(abstract)
            if dim_id is None:
                continue
            tier = placements.get(("sbuf", out_name, dim_id), "per_tile")
            if _TIER_RANK[tier] < _TIER_RANK["per_block"]:
                result = False
                break
    return result


def _check_placement_feasibility(ir: KernelIR) -> bool:
    """Each tensor must have at least one feasible emission depth."""
    return all(_group_feasibility_ok(ir, gi) for gi in range(len(ir.groups)))


def _group_feasibility_ok(ir: KernelIR, group_idx: int) -> bool:
    """Check placement feasibility for every tensor touched by one group."""
    ir = ir
    group = ir.groups[group_idx]
    dim_order = group.dim_order
    pos = {d: i for i, d in enumerate(dim_order)}
    tensors: set[str] = set()
    for op in group.ops:
        tensors.update(name for name in _op_tensors(ir, op) if name in ir.logical_tensors)
    return all(_tensor_feasibility_ok(ir, group_idx, name, pos, len(dim_order)) for name in tensors)


def _tensor_feasibility_ok(ir: KernelIR, group_idx: int, tensor_name: str, pos: dict[str, int], n: int) -> bool:
    """Intersection of per-dim depth ranges must be non-empty.

    For dims whose ltile loop is absorbed by a gadget (e.g.
    matmul_block iterates K/M/N internally), ``per_block``
    semantics widens: the surrounding loop never opens an
    ltile loop on that dim, so a ``per_block`` load can emit
    anywhere from ``block_depth(pos)+1`` down through
    ``body_depth(n)`` — not just the narrow (2p+1, 2p+1) slot.
    """
    lo = 0
    hi = body_depth(n)
    placements = ir.groups[group_idx].tensor_placements
    absorbed = _gadget_absorbed_blocking_dims(ir, group_idx)
    for d in ir.logical_tensors[tensor_name].dim_ids:
        if d not in pos:
            continue
        key = ("sbuf", tensor_name, d)
        if key not in placements:
            continue
        tier = placements[key]
        d_lo, d_hi = tier_depth_range(tier, pos[d], n)
        if tier == "per_block" and d in absorbed:
            d_hi = body_depth(n)
        lo = max(lo, d_lo)
        hi = min(hi, d_hi)
    return lo <= hi


def tier_depth_range(tier: str, pos: int, n: int) -> tuple[int, int]:
    """Allowed emission-depth range for a dim at ``pos`` under a given tier."""
    if tier == "per_tile":
        rng = (ltile_depth(pos) + 1, body_depth(n))
    elif tier == "per_block":
        rng = (block_depth(pos) + 1, ltile_depth(pos))
    elif tier == "full":
        rng = (0, block_depth(pos))
    else:
        raise ValueError(f"Unknown tier {tier!r}")
    return rng


def _check_cross_group_placements(ir: KernelIR, tensor_to_groups: dict[str, set[int]]) -> bool:
    """Cross-group tensors must be ``full`` in every touching group on shared-scope dims."""
    ir = ir
    return all(
        ir.groups[gi].tensor_placements.get(("sbuf", tname, d)) == "full"
        for tname, groups in tensor_to_groups.items()
        if len(groups) >= 2
        for gi in groups
        for d in set(ir.groups[gi].dim_order) & set(ir.logical_tensors[tname].dim_ids)
    )
