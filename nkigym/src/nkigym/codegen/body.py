"""BlockNode-driven body emitter.

Walks the canonical / transformed schedule tree and renders each
:class:`BlockNode` as a Python source fragment. Each block emits, in
order:

1. ``nl.ndarray(...)`` declarations — one per :attr:`BlockNode.alloc_buffers`.
2. The block's body — its ``ForNode`` chain ending in one :class:`ISANode`,
   or child sub-blocks if nested.

Operand slices are rendered via :func:`render_buffer_region` from the
ISA leaf's :attr:`ISANode.operand_bindings`.
"""

from __future__ import annotations

from typing import Any

from nkigym.codegen.compact import rebased_region
from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, Mod, Mul, Var, _format_raw, format_expr
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode

_INDENT = "    "


def emit_body(ir: KernelIR) -> str:
    """Emit the kernel body for the entire tree.

    The root is a BlockNode (empty iter_vars, holds kernel-lifetime buffers).
    Emit it directly at depth=1 (one indent level inside the kernel function).
    A ``{loop_nid: annotation}`` map of every ``software_pipeline`` annotation
    is built once and threaded down so a pipelined loop rotates its
    multi-version buffer accesses (see :func:`_emit_subtree`).
    """
    code: list[str] = []
    pipeline_map = _pipeline_loops(ir)
    _emit_block(ir, ir.tree.root, depth=1, code=code, pipeline_map=pipeline_map, rotations={})
    return "\n".join(code) + "\n"


def _pipeline_loops(ir: KernelIR) -> dict[int, dict[str, Any]]:
    """Map ``loop_nid -> software_pipeline annotation`` for every annotated block.

    Scans every BlockNode; an absent ``software_pipeline`` annotation
    contributes nothing, so an un-annotated kernel yields an empty map and the
    rotation threading is a no-op.
    """
    out: dict[int, dict[str, Any]] = {}
    for block_nid in ir.tree.blocks():
        block = ir.tree.data(block_nid)
        assert isinstance(block, BlockNode)
        annotation = block.annotations.get("software_pipeline")
        if annotation is not None:
            out[annotation["loop_nid"]] = annotation
    return out


def _emit_block(
    ir: KernelIR,
    block_nid: int,
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
) -> None:
    """Emit one BlockNode: alloc_buffers then body."""
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    indent = _INDENT * depth
    for buf in block.alloc_buffers:
        code.append(indent + _emit_alloc(buf))
    for child_nid in ir.tree.children(block_nid):
        child_data = ir.tree.data(child_nid)
        if isinstance(child_data, BlockNode):
            _emit_block(ir, child_nid, depth, code, pipeline_map, rotations)
        else:
            _emit_subtree(ir, child_nid, depth, code, pipeline_map, rotations)


def _emit_subtree(
    ir: KernelIR,
    nid: int,
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
) -> None:
    """Emit a ForNode, ISANode, or nested BlockNode subtree.

    A BlockNode may appear as a ForNode child once ``compute_at`` lifts /
    sinks a block into a loop body; delegate it to :func:`_emit_block` so
    its ``alloc_buffers`` and body render in place.

    When ``nid`` is a pipelined loop (a key of ``pipeline_map``), the loop is
    emitted monolithically (Increment 1: no prologue/epilogue) and every
    ``versions>1`` buffer touched in its subtree is added to ``rotations`` with
    a ``loop_var % versions`` tile-axis rotation before recursing.
    """
    indent = _INDENT * depth
    node = ir.tree.data(nid)
    if isinstance(node, ForNode):
        child_rotations = rotations
        if nid in pipeline_map:
            child_rotations = {**rotations, **_pipeline_rotations(ir, nid, node.loop_var)}
        code.append(indent + f"for {node.loop_var} in range({node.extent}):")
        for child_nid in ir.tree.children(nid):
            _emit_subtree(ir, child_nid, depth + 1, code, pipeline_map, child_rotations)
    elif isinstance(node, ISANode):
        code.append(indent + _emit_isa_call(node, ir, rotations))
    elif isinstance(node, BlockNode):
        _emit_block(ir, nid, depth, code, pipeline_map, rotations)
    else:
        raise TypeError(f"unexpected subtree node type {type(node).__name__}")


def _pipeline_rotations(ir: KernelIR, loop_nid: int, loop_var: str) -> dict[str, Expr]:
    """Tile-axis rotations for every ``versions>1`` buffer touched in a loop's subtree.

    Walks the descendant ISA leaves of the pipelined loop, collecting the
    tensor of every operand binding, and maps each multi-version buffer to its
    ``loop_var % versions`` rotation (see :func:`_version_rotation`).
    Single-version buffers contribute nothing.
    """
    out: dict[str, Expr] = {}
    for desc_nid in ir.tree.descendants(loop_nid):
        data = ir.tree.data(desc_nid)
        if not isinstance(data, ISANode):
            continue
        for region in data.operand_bindings.values():
            rotation = _version_rotation(ir.buffer(region.tensor), loop_var)
            if rotation is not None:
                out[region.tensor] = rotation
    return out


def _version_rotation(buf: Buffer, loop_var: str) -> Expr | None:
    """Return the tile-axis version rotation for a multi-version buffer, or None.

    ``num_p_tiles`` (the per-version tile span) is the middle physical dim
    divided by versions. When ``num_p_tiles == 1`` the rotation is the bare
    ``loop_var % versions`` (NO ``* 1`` — the validated kernel renders
    ``i_d1_0 % 2``, not ``i_d1_0 % 2 * 1``); only a >1 span wraps in
    ``Mul(..., Const(num_p_tiles))``.
    """
    if buf.versions <= 1:
        result = None
    else:
        mod = Mod(left=Var(name=loop_var), right=Const(value=buf.versions))
        num_p_tiles = buf.physical_shape()[1] // buf.versions
        result = mod if num_p_tiles == 1 else Mul(left=mod, right=Const(value=num_p_tiles))
    return result


def _emit_alloc(buf: Buffer) -> str:
    """Emit a single ``nl.ndarray(...)`` line for ``buf``.

    Shape comes from :meth:`Buffer.physical_shape` — the shared source of
    truth that also feeds the tree visualization.
    """
    shape = "(" + ", ".join(str(s) for s in buf.physical_shape()) + ")"
    return f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, buffer=nl.{buf.location})"


def _emit_isa_call(node: ISANode, ir: KernelIR, rotations: dict[str, Expr]) -> str:
    """Emit ``nisa.<NAME>(slot=<region>, ..., kwarg=value, ...)`` for one ISA leaf.

    ``rotations`` maps a tensor name to its ``loop_var % versions`` tile-axis
    rotation when an enclosing loop is pipelined; ``rotations.get(...)`` is
    ``None`` for every single-version tensor, so the slice renders unchanged.
    """
    op_cls = node.op_cls
    parts: list[str] = []
    for slot in op_cls.OPERAND_AXES:
        if slot in node.operand_bindings:
            region = node.operand_bindings[slot]
            buf = ir.buffer(region.tensor)
            rendered = render_buffer_region(rebased_region(region, buf, ir.tree), buf, rotations.get(region.tensor))
            parts.append(f"{slot}={rendered}")
    for k, v in node.kwargs.items():
        parts.append(f"{k}={v!r}")
    return f"nisa.{op_cls.NAME}({', '.join(parts)})"


def _format_tile_index(lo: Expr, rotation: Expr | None) -> str:
    """Render the SBUF/PSUM tile-axis index, optionally + a version rotation.

    ``format_expr`` normalises through ``to_affine``, which RAISES
    ``NonAffineError`` on ``Mod(Var, Const)`` (a version rotation like
    ``i_d1_0 % 2``) — the modulo of a variable is not affine. So the rotation
    is rendered with the non-normalising ``_format_raw`` and combined with the
    (affine) ``lo`` here, dropping ``lo`` when it is the rebased ``Const(0)``.
    """
    if rotation is None:
        result = format_expr(lo)
    else:
        rot_str = _format_raw(rotation)
        if isinstance(lo, Const) and lo.value == 0:
            result = rot_str
        else:
            result = f"{format_expr(lo)} + {rot_str}"
    return result


def render_buffer_region(region: BufferRegion, buf: Buffer, rotation: Expr | None = None) -> str:
    """Render a :class:`BufferRegion` as a Python slice expression on its tensor.

    ``rotation`` (when not None) is the ``loop_var % versions`` term combined
    with the SBUF/PSUM tile-axis index to select a pipeline buffer version
    (see :func:`_format_tile_index`). Ignored for ``shared_hbm`` (no tile
    axis) and when None (single-version buffers, byte-identical to before).
    """
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            if not isinstance(hi, Const) or hi.value != PARTITION_DIM:
                raise AssertionError(f"{buf.name}: SBUF/PSUM partition axis must use a partition-sized tile; got {hi}")
            parts.append(f"0:{PARTITION_DIM}")
            parts.append(_format_tile_index(lo, rotation))
        else:
            lo_str = format_expr(lo)
            hi_str = format_expr(hi)
            parts.append(f"{lo_str}:{lo_str} + {hi_str}")
    return f"{region.tensor}[{', '.join(parts)}]"


__all__ = ["emit_body", "render_buffer_region"]
