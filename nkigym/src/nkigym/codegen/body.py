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

from nkigym.ir import KernelIR
from nkigym.ir.expr import Const, format_expr
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode

_INDENT = "    "
_PARTITION_DIM = 128


def emit_body(ir: KernelIR) -> str:
    """Emit the kernel body for the entire tree.

    The root is a BlockNode (empty iter_vars, holds kernel-lifetime buffers).
    Emit it directly at depth=1 (one indent level inside the kernel function).
    """
    code: list[str] = []
    _emit_block(ir, ir.tree.root, depth=1, code=code)
    return "\n".join(code) + "\n"


def _emit_block(ir: KernelIR, block_nid: int, depth: int, code: list[str]) -> None:
    """Emit one BlockNode: alloc_buffers then body."""
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    indent = _INDENT * depth
    for buf in block.alloc_buffers:
        code.append(indent + _emit_alloc(buf))
    for child_nid in ir.tree.children(block_nid):
        child_data = ir.tree.data(child_nid)
        if isinstance(child_data, BlockNode):
            _emit_block(ir, child_nid, depth, code)
        else:
            _emit_subtree(ir, child_nid, depth, code)


def _emit_subtree(ir: KernelIR, nid: int, depth: int, code: list[str]) -> None:
    """Emit a ForNode or ISANode subtree."""
    indent = _INDENT * depth
    node = ir.tree.data(nid)
    if isinstance(node, ForNode):
        code.append(indent + f"for {node.loop_var} in range({node.extent}):")
        for child_nid in ir.tree.children(nid):
            _emit_subtree(ir, child_nid, depth + 1, code)
    elif isinstance(node, ISANode):
        code.append(indent + _emit_isa_call(node, ir))
    else:
        raise TypeError(f"unexpected subtree node type {type(node).__name__}")


def _emit_alloc(buf: Buffer) -> str:
    """Emit a single ``nl.ndarray(...)`` line for ``buf``."""
    if buf.location == "shared_hbm":
        shape = "(" + ", ".join(str(s) for s in buf.shape) + ")"
    else:
        if len(buf.shape) != 2:
            raise AssertionError(f"{buf.name}: SBUF/PSUM allocation expects a 2D shape; got {buf.shape}")
        P, F = buf.shape
        if P % _PARTITION_DIM != 0:
            raise AssertionError(f"{buf.name}: P={P} must be a multiple of {_PARTITION_DIM}")
        shape = f"({_PARTITION_DIM}, {P // _PARTITION_DIM}, {F})"
    return f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.dtype}, buffer=nl.{buf.location})"


def _emit_isa_call(node: ISANode, ir: KernelIR) -> str:
    """Emit ``nisa.<NAME>(slot=<region>, ..., kwarg=value, ...)`` for one ISA leaf."""
    op_cls = node.op_cls
    parts: list[str] = []
    for slot in op_cls.OPERAND_AXES:
        if slot in node.operand_bindings:
            region = node.operand_bindings[slot]
            buf = ir.buffer(region.tensor)
            parts.append(f"{slot}={render_buffer_region(region, buf)}")
    for k, v in node.kwargs.items():
        parts.append(f"{k}={v!r}")
    return f"nisa.{op_cls.NAME}({', '.join(parts)})"


def render_buffer_region(region: BufferRegion, buf: Buffer) -> str:
    """Render a :class:`BufferRegion` as a Python slice expression on its tensor."""
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            if not isinstance(hi, Const) or hi.value != _PARTITION_DIM:
                raise AssertionError(f"{buf.name}: SBUF/PSUM partition axis must use a partition-sized tile; got {hi}")
            parts.append(f"0:{_PARTITION_DIM}")
            parts.append(format_expr(lo))
        else:
            lo_str = format_expr(lo)
            hi_str = format_expr(hi)
            parts.append(f"{lo_str}:{lo_str} + {hi_str}")
    return f"{region.tensor}[{', '.join(parts)}]"


__all__ = ["emit_body", "render_buffer_region"]
