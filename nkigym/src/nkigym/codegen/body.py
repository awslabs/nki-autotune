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

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, Mod, Mul, Var, _format_raw, format_expr, to_affine
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree

_INDENT = "    "


def emit_body(ir: KernelIR) -> str:
    """Emit the kernel body for the entire tree.

    The root is a BlockNode (empty iter_vars, holds kernel-lifetime buffers).
    Emit it directly at depth=1 (one indent level inside the kernel function).
    A ``{loop_nid: annotation}`` map of every ``software_pipeline`` annotation
    is built once and threaded down so a pipelined loop rotates its
    multi-version buffer accesses (see :func:`_emit_subtree`).

    Buffer declarations are placed at their tightest materialized scope
    (:func:`_alloc_emit_anchors`): each ``nl.ndarray`` is emitted immediately before
    the first child that uses it, without crossing the buffer's owning-block
    placement boundary. Within that block, offset-carrying loops hoist the
    declaration to cover its live range. The
    ``{node_nid: [Buffer, ...]}`` map is threaded down so each node emits, before
    each child, the declarations anchored to that child.
    """
    code: list[str] = []
    pipeline_map = _pipeline_loops(ir)
    emit_before = _alloc_emit_anchors(ir)
    _emit_block(ir, ir.tree.root, depth=1, code=code, pipeline_map=pipeline_map, rotations={}, emit_before=emit_before)
    return "\n".join(code) + "\n"


def _alloc_emit_anchors(ir: KernelIR) -> dict[int, list[Buffer]]:
    """Map each tree node to the buffers emitted immediately before it.

    Scratch-buffer scope starts from the touchers' LCA, constrained by the owning
    block and hoisted over loops carried in its offsets; ``shared_hbm`` buffers use
    the root. The declaration anchors to the first child in dataflow order whose
    subtree contains a touching leaf, immediately before first use. A lone toucher
    anchors to that ISA leaf. Kernel parameters are never declared. Buffers are
    walked in ``all_buffers`` order for deterministic anchor lists.
    """
    params = set(ir.param_buffers)
    leaves_by_tensor: dict[str, list[int]] = {}
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                leaves_by_tensor.setdefault(region.tensor, []).append(nid)
    out: dict[int, list[Buffer]] = {}
    for name, buf in ir.all_buffers().items():
        if name in params:
            continue
        leaves = leaves_by_tensor.get(name)
        assert leaves, f"buffer {name!r} is declared but touched by no ISA leaf"
        scope = ir.tree.root if buf.location == "shared_hbm" else _hoisted_scope(ir.tree, name, leaves)
        anchor = _anchor_child(ir.tree, scope, leaves)
        out.setdefault(anchor, []).append(buf)
    return out


def _anchor_child(tree: KernelTree, scope: int, leaves: list[int]) -> int:
    """Return the node to emit a buffer's declaration before.

    When ``scope`` is an ISA leaf (lone toucher), the buffer anchors to that leaf.
    Otherwise the anchor is the first child of ``scope`` (in child order) whose
    subtree contains one of ``leaves`` — the first dataflow use of the buffer.
    """
    if isinstance(tree.data(scope), ISANode):
        return scope
    touch = set(leaves)
    for child in tree.children(scope):
        subtree = {child, *tree.descendants(child)}
        if subtree & touch:
            return child
    raise AssertionError(f"scope {scope} has no child whose subtree touches the buffer")


def _lca_nodes(tree: KernelTree, nids: list[int]) -> int:
    """Lowest common ancestor of ``nids`` — the deepest node on every root->nid path.

    Each node's path is its ancestors (root-first) plus itself; the LCA is the
    last node shared by all paths. A single distinct nid is its own LCA.
    """
    unique = set(nids)
    if len(unique) == 1:
        return next(iter(unique))
    paths = [[*tree.ancestors(nid), nid] for nid in unique]
    lca = tree.root
    for level in zip(*paths):
        if len(set(level)) == 1:
            lca = level[0]
        else:
            break
    return lca


def _carried_loop_vars(tree: KernelTree, name: str, leaves: list[int]) -> set[str]:
    """Loop vars appearing in any region offset of buffer ``name`` across its touchers.

    A var in a region's ``lo`` means the buffer's live slice varies with that loop, so
    the buffer is carried across it and its declaration must hoist above the loop.
    """
    carried: set[str] = set()
    for leaf in leaves:
        data = tree.data(leaf)
        if not isinstance(data, ISANode):
            continue
        for region in data.operand_bindings.values():
            if region.tensor != name:
                continue
            for lo, _width in region.ranges:
                carried |= {var for var in to_affine(lo) if var is not None}
    return carried


def _owning_block(tree: KernelTree, name: str) -> int:
    """Return the block whose ``alloc_buffers`` entry materializes ``name``."""
    for nid in tree.blocks():
        block = tree.data(nid)
        assert isinstance(block, BlockNode)
        if any(buf.name == name for buf in block.alloc_buffers):
            return nid
    raise AssertionError(f"buffer {name!r} is declared by no block")


def _hoisted_scope(tree: KernelTree, name: str, leaves: list[int]) -> int:
    """Find the tightest declaration scope consistent with placement and offsets.

    The owning block is a material placement boundary. The renderer may tighten
    within that block's direct loop nest, but it must not cross into a nested block;
    doing so would silently place a root-owned structural-only buffer after
    CodeMotion. Within the allowed nest, an offset that references an enclosing
    loop carries the allocation across that loop, so the scope rises above the
    outermost such loop.
    """
    lca = _lca_nodes(tree, leaves)
    owner = _owning_block(tree, name)
    chain = [*tree.ancestors(lca), lca]
    if owner in chain:
        owner_index = chain.index(owner)
        local_chain = chain[owner_index + 1 :]
        if any(isinstance(tree.data(nid), BlockNode) for nid in local_chain):
            return owner
    else:
        local_chain = chain
    carried = _carried_loop_vars(tree, name, leaves)
    scope = lca
    for nid in local_chain:
        data = tree.data(nid)
        if isinstance(data, ForNode) and data.loop_var in carried:
            parent = tree.parent(nid)
            assert parent is not None, f"carried loop {nid} has no parent"
            scope = parent
            break
    return scope


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
    emit_before: dict[int, list[Buffer]],
) -> None:
    """Emit one BlockNode: each child's anchored buffer declarations, then the child."""
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    indent = _INDENT * depth
    for child_nid in ir.tree.children(block_nid):
        for buf in emit_before.get(child_nid, ()):
            code.append(indent + _emit_alloc(buf))
        child_data = ir.tree.data(child_nid)
        if isinstance(child_data, BlockNode):
            _emit_block(ir, child_nid, depth, code, pipeline_map, rotations, emit_before)
        else:
            _emit_subtree(ir, child_nid, depth, code, pipeline_map, rotations, emit_before)


def _emit_subtree(
    ir: KernelIR,
    nid: int,
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
    emit_before: dict[int, list[Buffer]],
) -> None:
    """Emit a ForNode, ISANode, or nested BlockNode subtree.

    A BlockNode may appear as a ForNode child once ``compute_at`` lifts / sinks a
    block into a loop body; delegate it to :func:`_emit_block`.

    A ForNode emits, before each of its children, any buffers anchored to that
    child (``emit_before[child]``) — so a buffer used only within the loop is
    declared inside it, immediately before its first use.

    When ``nid`` is a pipelined loop (a key of ``pipeline_map``), the loop is
    emitted monolithically and the buffers versioned by that pipeline are added
    to ``rotations`` before recursing.
    """
    indent = _INDENT * depth
    node = ir.tree.data(nid)
    if isinstance(node, ForNode):
        child_rotations = rotations
        if nid in pipeline_map:
            annotation = pipeline_map[nid]
            child_rotations = {**rotations, **_pipeline_rotations(ir, node.loop_var, annotation["versioned_buffers"])}
        code.append(indent + f"for {node.loop_var} in range({node.extent}):")
        child_indent = _INDENT * (depth + 1)
        for child_nid in ir.tree.children(nid):
            for buf in emit_before.get(child_nid, ()):
                code.append(child_indent + _emit_alloc(buf))
            _emit_subtree(ir, child_nid, depth + 1, code, pipeline_map, child_rotations, emit_before)
    elif isinstance(node, ISANode):
        code.append(indent + _emit_isa_call(node, ir, rotations))
    elif isinstance(node, BlockNode):
        _emit_block(ir, nid, depth, code, pipeline_map, rotations, emit_before)
    else:
        raise TypeError(f"unexpected subtree node type {type(node).__name__}")


def _pipeline_rotations(ir: KernelIR, loop_var: str, versioned_buffers: tuple[str, ...]) -> dict[str, Expr]:
    """Return rotations for the buffers versioned by one pipeline."""
    out: dict[str, Expr] = {}
    for name in versioned_buffers:
        rotation = _version_rotation(ir.buffer(name), loop_var)
        if rotation is None:
            raise AssertionError(f"pipeline marks single-version buffer {name!r} as versioned")
        out[name] = rotation
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
    """Emit the buffer declaration for ``buf``.

    ``shared_hbm`` buffers emit a single bare ``nl.ndarray`` of
    :meth:`Buffer.physical_shape` (no tile axis). Every sbuf/psum buffer emits a
    Python list of :attr:`Buffer.list_len` per-tile ndarrays
    (:meth:`Buffer.per_tile_physical_shape`) — uniformly, including ``list_len == 1``
    (a list-of-one), so the call site always indexes with a leading ``[list_idx]``.
    """
    if buf.location == "shared_hbm":
        shape = "(" + ", ".join(str(s) for s in buf.physical_shape()) + ")"
        result = f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, buffer=nl.{buf.location})"
    else:
        shape = "(" + ", ".join(str(s) for s in buf.per_tile_physical_shape()) + ")"
        result = (
            f"{buf.name} = [nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, "
            f"buffer=nl.{buf.location}) for _ in range({buf.list_len})]"
        )
    return result


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
            rendered = render_buffer_region(region, buf, rotations.get(region.tensor))
            parts.append(f"{slot}={rendered}")
    for k, v in node.kwargs.items():
        parts.append(f"{k}={_render_kwarg(k, v)}")
    return f"nisa.{op_cls.NAME}({', '.join(parts)})"


_NL_OP_KWARGS = frozenset({"op", "op0", "op1", "reduce_op"})
"""ISA kwargs whose string value names an ``nl`` math operator. ``nisa`` ALU
ops (``tensor_tensor``, ``tensor_scalar``, ``activation``, ``tensor_reduce``)
take the operator as an ``nl`` reference (e.g. ``op=nl.add``), not a bare
string — so these render as ``nl.<value>`` while every other kwarg renders
via ``repr`` (e.g. memset's ``value=0.0``)."""


def _render_kwarg(key: str, value: Any) -> str:
    """Render one ISA kwarg value, mapping ALU-operator names to ``nl.<name>``."""
    if key in _NL_OP_KWARGS and isinstance(value, str):
        rendered = f"nl.{value}"
    else:
        rendered = repr(value)
    return rendered


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

    ``shared_hbm`` renders flat ``name[lo:hi, ...]``. Every sbuf/psum buffer renders
    as a list access ``name[list_idx][0:P, mid_idx, F]`` (uniform — there is no bare
    form). The partition axis (axis 0) carries the tile index ``t``; with
    ``a = per_tile middle = T // list_len``, branch on ``list_len``:

    * ``list_len == 1`` — a list-of-one: ``list_idx = 0``, ``mid_idx = t`` (the whole
      tile index). Preserves the pre-uniform packed middle, so a canonical multi-tile
      buffer renders ``buf[0][0:P, t, F]``.
    * ``a == 1`` (``list_len == T``, the full split) — ``list_idx = t``, ``mid_idx = 0``.
    * ``a > 1`` (``1 < list_len < T``) — ``list_idx = t // a``, ``mid_idx = t % a``,
      both via the non-normalising ``_format_raw`` (the aligned index is non-affine
      under ``FloorDiv``, so ``format_expr``/``to_affine`` would raise).

    ``rotation`` (the pipeline version term) applies only on the ``list_len == 1`` and
    ``a == 1`` paths; ``a > 1`` requires ``list_len > 1``, and ``versions > 1`` with
    ``list_len > 1`` is rejected at allocation, so no rotation reaches the ``a > 1`` path.
    """
    list_subscript = ""
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            if not isinstance(hi, Const) or hi.value != PARTITION_DIM:
                raise AssertionError(f"{buf.name}: SBUF/PSUM partition axis must use a partition-sized tile; got {hi}")
            a = buf.per_tile_physical_shape()[1]
            if buf.list_len == 1:
                list_subscript = "[0]"
                parts.append(f"0:{PARTITION_DIM}")
                parts.append(_format_tile_index(lo, rotation))
            elif a == 1:
                list_subscript = f"[{_format_tile_index(lo, rotation)}]"
                parts.append(f"0:{PARTITION_DIM}")
                parts.append("0")
            else:
                tile = f"({_format_raw(lo)})"
                list_subscript = f"[{tile} // {a}]"
                parts.append(f"0:{PARTITION_DIM}")
                parts.append(f"{tile} % {a}")
        else:
            lo_str = format_expr(lo)
            hi_str = format_expr(hi)
            parts.append(f"{lo_str}:{lo_str} + {hi_str}")
    return f"{region.tensor}{list_subscript}[{', '.join(parts)}]"


__all__ = ["emit_body", "render_buffer_region"]
