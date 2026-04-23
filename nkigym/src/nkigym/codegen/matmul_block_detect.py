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
      tier on ``K`` strictly below ``full`` (i.e. ``per_block``
      or ``per_tile``) in this group.
    """
    k_dim = _resolve_k_dim(ir, op)
    ok = k_dim is not None and _k_has_multiple_blocks(ir, k_dim)
    if ok:
        assert k_dim is not None
        ok = _k_inputs_not_full(ir, op, group_idx, k_dim)
    return ok


def _resolve_k_dim(ir: KernelIR, op: NKIOp) -> str | None:
    """Return the concrete K dim for ``op``, or None if not a matmul-block candidate shape."""
    result: str | None = None
    if type(op) is NKIMatmul:
        axis_map = ir.context.op_axis_map.get(op, {})
        result = axis_map.get("K")
    return result


def _k_has_multiple_blocks(ir: KernelIR, k_dim: str) -> bool:
    """True iff ``k_dim`` is split into at least two outer blocks."""
    context = ir.context
    di = context.dimensions[k_dim]
    tpb = context.ltiles_per_block.get(k_dim, 1)
    num_blocks = di.dim_size // (tpb * di.logical_tile_size)
    return num_blocks > 1


def _k_inputs_not_full(ir: KernelIR, op: NKIOp, group_idx: int, k_dim: str) -> bool:
    """True iff both matmul SBUF inputs have non-full tier on ``k_dim``."""
    inputs = ir.context.op_inputs.get(op, {})
    placements = ir.graph.groups[group_idx].tensor_placements
    result = True
    for role in ("stationary", "moving"):
        tensor = inputs.get(role)
        tier = placements.get(("sbuf", tensor, k_dim), "per_tile") if tensor else "full"
        if tensor is None or _TIER_RANK[tier] >= _TIER_RANK["full"]:
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
    for op in ir.graph.groups[group_idx].ops:
        if is_matmul_block_candidate(ir, op, group_idx):
            for abstract in ("K", "M", "N"):
                dim_id = ir.context.op_axis_map.get(op, {}).get(abstract)
                if dim_id is not None:
                    result.add(dim_id)
    return result
