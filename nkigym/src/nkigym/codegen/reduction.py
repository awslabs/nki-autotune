"""Multi-chunk reduction codegen.

When an op has a blocking dim (trip > 1) that is NOT in one of
its output tensors' dim_ids, that dim is a reduction loop the
renderer must close. The op's class declares the combinator via
``REDUCE_COMBINATOR[output_role]`` — resolved per-instance through
``resolve_reduce_combinator``.

Generated code shape (per reduced output):

1. ``nisa.memset(sbuf_<out>, <init>)`` outside every material
   reduction-dim block loop.
2. Per chunk: call the op writing its per-chunk result to a
   scratch buffer ``sbuf_<out>_chunk``.
3. Combine scratch into the running output via the declared
   combinator (``tensor_scalar`` add / ``tensor_tensor`` max).
"""

from dataclasses import dataclass

from nkigym.codegen.buffers import sbuf_buffer
from nkigym.codegen.group_loops import DepthPlan
from nkigym.codegen.sbuf_buffer import AxisAccess, buffer_ident
from nkigym.kernel_ir import KernelIR
from nkigym.kernel_ir.types import TensorInfo
from nkigym.kernel_ir.validate.emission import Placement, block_depth, ltile_depth
from nkigym.ops.base import NKIOp

_INIT: dict[str, str] = {"add": "0.0", "maximum": "float('-inf')", "minimum": "float('inf')"}


@dataclass
class ReducedOutput:
    """One of the op's outputs that needs multi-chunk reduction codegen."""

    role: str
    tensor_name: str
    combinator: str


def reduced_outputs_with_multichunk(ir: KernelIR, op: NKIOp) -> list[ReducedOutput]:
    """Return every output of ``op`` whose reduction dim(s) are multi-chunk."""
    ir = ir
    op_cls = type(op)
    axis_map = ir.op_axis_map.get(op, {})
    blocking = ir.op_blocking_dims.get(op, set())
    kwargs = ir.op_kwargs.get(op, {})
    outputs = ir.op_outputs.get(op, [])
    result: list[ReducedOutput] = []
    for role_idx, role in enumerate(op_cls.OUTPUT_AXES):
        if role_idx >= len(outputs):
            continue
        combinator = op_cls.resolve_reduce_combinator(role, kwargs)
        if combinator is None:
            continue
        out_axes = op_cls.OUTPUT_AXES[role]
        out_dims = {axis_map.get(ax) for ax in out_axes if axis_map.get(ax) is not None}
        reduction_dims = blocking - out_dims
        if not reduction_dims:
            continue
        if not _any_multichunk(ir, reduction_dims):
            continue
        result.append(ReducedOutput(role=role, tensor_name=outputs[role_idx], combinator=combinator))
    return result


def _any_multichunk(ir: KernelIR, dim_ids: set[str]) -> bool:
    """True iff any dim's block-loop or ltile-loop trip is > 1."""
    ir = ir
    result = False
    for d in dim_ids:
        di = ir.dimensions[d]
        tpb = ir.ltiles_per_block.get(d, 1)
        num_blocks = di.dim_size // (tpb * di.logical_tile_size)
        if num_blocks > 1 or tpb > 1:
            result = True
            break
    return result


def init_memset_lines(ir: KernelIR, gi: int, reduced: ReducedOutput, depth: int) -> list[str]:
    """Emit one ``nisa.memset`` targeting the running slot in scope at ``depth``.

    The memset fires at ``before[depth]`` (outside every material
    reduction-dim loop, but inside every outer output-dim loop).
    The running output's SBUF access uses the loop vars that are
    open at ``depth`` — so a single memset zeroes the slot that
    the next reduction sequence is about to fill.
    """
    init = _INIT.get(reduced.combinator)
    if init is None:
        raise ValueError(f"no init value for combinator {reduced.combinator!r}")
    running = _running_access_at_depth(ir, gi, reduced, depth)
    return [f"nisa.memset({running}, {init})"]


def _running_access_at_depth(ir: KernelIR, group_idx: int, reduced: ReducedOutput, depth: int) -> str:
    """SBUF access for the running output based on loops open at an arbitrary depth."""
    placement = Placement("before", depth)
    return _running_access(ir, group_idx, reduced, placement)


def init_depth(ir: KernelIR, op: NKIOp, gi: int) -> int:
    """Outermost material reduction-dim depth for this op.

    The memset fires at ``before[depth]`` where ``depth`` is the
    position of the outermost material reduction dim in the
    group's dim_order — i.e. OUTSIDE that dim's block loop.
    """
    ir = ir
    op_cls = type(op)
    axis_map = ir.op_axis_map.get(op, {})
    blocking = ir.op_blocking_dims.get(op, set())
    kwargs = ir.op_kwargs.get(op, {})
    reduction_dims: set[str] = set()
    outputs = ir.op_outputs.get(op, [])
    for role_idx, role in enumerate(op_cls.OUTPUT_AXES):
        if role_idx >= len(outputs):
            continue
        combinator = op_cls.resolve_reduce_combinator(role, kwargs)
        if combinator is None:
            continue
        out_axes = op_cls.OUTPUT_AXES[role]
        out_dims = {axis_map.get(ax) for ax in out_axes if axis_map.get(ax) is not None}
        reduction_dims |= blocking - out_dims
    dim_order = ir.groups[gi].dim_order
    positions = [dim_order.index(d) for d in reduction_dims if d in dim_order]
    return block_depth(min(positions)) if positions else 0


def scratch_shape(ir: KernelIR, tinfo: TensorInfo) -> tuple[int, int]:
    """Return ``(P, F)`` physical tile shape for a reduced output's scratch buffer."""
    dim_ids = tinfo.dim_ids
    p = ir.dimensions[dim_ids[0]].physical_tile_size
    f = ir.dimensions[dim_ids[1]].physical_tile_size if len(dim_ids) == 2 else 1
    return p, f


def apply_reduction_plan(
    ir: KernelIR,
    op: NKIOp,
    gi: int,
    reduced: list[ReducedOutput],
    placement: Placement,
    block_lines: list[str],
    before_plan: DepthPlan,
) -> list[str]:
    """Wrap a multi-chunk reduction op: init memset + per-chunk scratch + combine.

    The op call (in ``block_lines``) already has its
    reduced-output destinations pointing at scratch buffers via
    ``scratch_override``. This adds:

    1. A direct scratch-buffer ``nl.ndarray`` allocation at the
       group top (depth 0).
    2. An init memset for the running output at ``init_depth``
       (outside the outermost material reduction block loop).
    3. Combine ISA calls after the op call to accumulate
       chunk → running via the declared combinator.
    """
    ir = ir
    depth = init_depth(ir, op, gi)
    top_lines = before_plan.setdefault(gi, {}).setdefault(0, [])
    init_lines = before_plan.setdefault(gi, {}).setdefault(depth, [])
    for r in reduced:
        tinfo = ir.logical_tensors[r.tensor_name]
        p, f = scratch_shape(ir, tinfo)
        chunk_dtype = _chunk_scratch_dtype(r.combinator, tinfo.dtype)
        chunk_name = f"{buffer_ident(r.tensor_name)}_chunk"
        top_lines.append(f"sbuf_{chunk_name} = nl.ndarray(({p}, {f}), dtype=nl.{chunk_dtype}, buffer=nl.sbuf)")
        init_lines.extend(init_memset_lines(ir, gi, r, depth))
        running = _running_access(ir, gi, r, placement)
        chunk = f"sbuf_{chunk_name}[0:{p}, 0:{f}]"
        block_lines.append(combine_line(ir, r, chunk, running))
    return block_lines


def _chunk_scratch_dtype(combinator: str, tensor_dtype: str) -> str:
    """Return the scratch-buffer dtype for a per-chunk reduction output.

    Hardware mandates ``float32`` for ``nisa.tensor_scalar.operand0``,
    which is where the chunk gets plugged in when the combinator
    is ``add``. Max / min use ``nisa.tensor_tensor`` which accepts
    any dtype — those can inherit the running buffer's dtype.
    """
    return "float32" if combinator == "add" else tensor_dtype


def _running_access(ir: KernelIR, group_idx: int, reduced: ReducedOutput, placement: Placement) -> str:
    """SBUF access string for the running (accumulated) output at the op's emission slot."""
    tinfo = ir.logical_tensors[reduced.tensor_name]
    buf = sbuf_buffer(ir, reduced.tensor_name)
    dim_ids = tinfo.dim_ids
    placements = ir.groups[group_idx].tensor_placements
    dim_order = ir.groups[group_idx].dim_order
    p_access = _running_axis_access(reduced.tensor_name, dim_ids[0], placements, dim_order, placement)
    if len(dim_ids) == 2:
        f_access = _running_axis_access(reduced.tensor_name, dim_ids[1], placements, dim_order, placement)
    else:
        f_access = AxisAccess(block="0", ltile="0")
    return buf.get_tile(p_access, f_access)


def _running_axis_access(
    tensor_name: str,
    dim_id: str,
    placements: dict[tuple[str, str, str], str],
    dim_order: list[str],
    placement: Placement,
) -> AxisAccess:
    """Bind block/ltile loop vars for the running output based on tier + open loops."""
    tier = placements.get(("sbuf", tensor_name, dim_id), "per_tile")
    block = "0"
    ltile = "0"
    if dim_id in dim_order:
        pos = dim_order.index(dim_id)
        if tier == "full" and placement.loop_open(block_depth(pos)):
            block = f"i_block_{dim_id}"
        if tier in ("per_block", "full") and placement.loop_open(ltile_depth(pos)):
            ltile = f"i_ltile_{dim_id}"
    return AxisAccess(block=block, ltile=ltile)


def combine_line(ir: KernelIR, reduced: ReducedOutput, chunk_access: str, running_access: str) -> str:
    """Emit the combine ISA call: running = combinator(running, chunk)."""
    _ = ir
    if reduced.combinator == "add":
        line = f"nisa.tensor_scalar({running_access}, {running_access}, op0=nl.add, operand0={chunk_access})"
    elif reduced.combinator in ("maximum", "minimum"):
        line = f"nisa.tensor_tensor({running_access}, {running_access}, {chunk_access}, nl.{reduced.combinator})"
    else:
        raise ValueError(f"unsupported combinator {reduced.combinator!r}")
    return line
