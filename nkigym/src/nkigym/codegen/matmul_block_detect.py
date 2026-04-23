"""Detect when an ``NKIMatmul`` should render via the ``matmul_block`` gadget.

Kept separate from ``matmul_block_render`` so ``codegen/buffers.py`` and
``codegen/dma.py`` can import the predicate without creating a
buffers→render→buffers cycle.
"""

from nkigym.kernel_ir import KernelIR
from nkigym.ops.base import NKIOp
from nkigym.ops.matmul import NKIMatmul

_TIER_RANK = {"per_tile": 0, "per_block": 1, "full": 2}


def is_matmul_block_candidate(ir: KernelIR, op: NKIOp, group_idx: int) -> bool:
    """True iff ``op`` should render via ``matmul_block`` instead of the classic PSUM path.

    Conditions:
    * ``op`` is an ``NKIMatmul``.
    * Its reduction axis ``K`` resolves to a blocked dim with
      ``num_blocks > 1``.
    * Both the ``stationary`` and ``moving`` SBUF inputs have
      ``per_block`` tier on ``K`` — so the inputs reload per
      outer-K-block iteration, the shape the gadget needs.
    * Both inputs' free-axis tiers (M for stationary, N for
      moving) are ``per_block`` or ``full`` — ``per_tile`` would
      require element-level loads that matmul_block can't
      express.
    * The matmul's output has a block-slab tier on its M and N
      dims (``per_block`` or ``full``) so the running
      accumulator persists across the outer-K loop.
    """
    k_dim = _resolve_k_dim(ir, op)
    ok = k_dim is not None and _k_has_multiple_blocks(ir, k_dim)
    if ok:
        assert k_dim is not None
        ok = (
            _k_inputs_not_full(ir, op, group_idx, k_dim)
            and _inputs_have_block_slab(ir, op, group_idx)
            and _output_has_block_slab(ir, op, group_idx)
        )
    return ok


def _inputs_have_block_slab(ir: KernelIR, op: NKIOp, group_idx: int) -> bool:
    """True iff the matmul inputs' free-axis tiers are ``per_block`` or ``full``.

    Free-axis for ``stationary`` is M (the second OPERAND_AXES
    entry) and for ``moving`` is N. ``per_tile`` would demand
    element-level accesses, which matmul_block doesn't emit.
    """
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


def _resolve_k_dim(ir: KernelIR, op: NKIOp) -> str | None:
    """Return the concrete K dim for ``op``, or None if not a matmul-block candidate shape."""
    result: str | None = None
    if type(op) is NKIMatmul:
        axis_map = ir.op_axis_map.get(op, {})
        result = axis_map.get("K")
    return result


def _k_has_multiple_blocks(ir: KernelIR, k_dim: str) -> bool:
    """True iff ``k_dim`` is split into at least two outer blocks."""
    ir = ir
    di = ir.dimensions[k_dim]
    tpb = ir.ltiles_per_block.get(k_dim, 1)
    num_blocks = di.dim_size // (tpb * di.logical_tile_size)
    return num_blocks > 1


def _k_inputs_not_full(ir: KernelIR, op: NKIOp, group_idx: int, k_dim: str) -> bool:
    """True iff both matmul SBUF inputs have ``per_block`` tier on ``k_dim``.

    matmul_block iterates the inner-K slab of a ``per_block``-tier
    input per outer-K-block step. ``full`` tier would skip the
    reload entirely and ``per_tile`` would force element-level
    loads the gadget can't express.
    """
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
    """True iff the matmul's output SBUF has ``per_block``-or-``full`` tier on its M and N dims.

    matmul_block writes the running accumulator across the
    outer-K loop, so the output buffer must hold at least a
    whole M/N block at a time — ``per_tile`` would force an
    element-level drain that the gadget can't express.
    """
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


def gadget_absorbed_dims(ir: KernelIR, group_idx: int) -> set[str]:
    """Return the dims whose ltile loops are absorbed by a gadget in this group.

    ``matmul_block`` absorbs K / M / N ltile iteration inside the
    gadget call, so the surrounding loop nest must not open those
    ltile loops and DMA gadgets must treat those ltiles as
    unbound (load the full slab per block iteration).
    """
    result: set[str] = set()
    for op in ir.groups[group_idx].ops:
        if is_matmul_block_candidate(ir, op, group_idx):
            for abstract in ("K", "M", "N"):
                dim_id = ir.op_axis_map.get(op, {}).get(abstract)
                if dim_id is not None:
                    result.add(dim_id)
    return result
