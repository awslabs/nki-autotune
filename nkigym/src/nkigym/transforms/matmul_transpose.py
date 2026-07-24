"""Transpose a matmul by swapping its operands.

For ``NKIMatmul(stationary=A, moving=B)``:

``A.T @ B = (B.T @ A).T``.

The transform rewrites one canonical matmul, its synthesized memset, and its
direct PSUM drain into a swapped matmul, an intermediate drain,
``NKITranspose``, and the original drain.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Mul, Var
from nkigym.ir.buffer_placement import place_buffers
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


@dataclass(frozen=True)
class MatmulTransposeOption(TransformOption):
    """Transpose the canonical matmul block at ``target_nid``."""

    target_nid: int


@dataclass(frozen=True)
class _CanonicalSpec:
    """Payloads for one canonical single-ISA block."""

    block: BlockNode
    loops: tuple[ForNode, ...]
    leaf: ISANode


@dataclass(frozen=True)
class _Match:
    """Canonical matmul segment selected for rewriting."""

    memset_block: int
    matmul_block: int
    stationary: str
    moving: str
    output: str


class MatmulTranspose(Transform[MatmulTransposeOption]):
    """Rewrite ``A.T @ B`` as ``(B.T @ A).T``."""

    def analyze(self, ir: KernelIR) -> list[MatmulTransposeOption]:
        """Return every canonical matmul segment eligible for operand swapping."""
        options: list[MatmulTransposeOption] = []
        for block_nid in ir.tree.children(ir.tree.root):
            if _match(ir, block_nid) is not None:
                options.append(MatmulTransposeOption(target_nid=block_nid))
        return options

    def apply(self, ir: KernelIR, option: MatmulTransposeOption) -> KernelIR:
        """Recheck ``option``, rewrite a deep copy, place buffers, and rebuild dependencies."""
        match = _match(ir, option.target_nid)
        if match is None:
            raise TransformLegalityError(
                f"MatmulTranspose target {option.target_nid} is not an eligible canonical matmul segment"
            )

        new_ir = copy.deepcopy(ir)
        copied_match = _match(new_ir, option.target_nid)
        if copied_match is None:
            raise AssertionError("MatmulTranspose match disappeared after deepcopy")
        _apply_match(new_ir, copied_match)
        place_buffers(new_ir.tree)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir


def _match(ir: KernelIR, target_nid: int) -> _Match | None:
    """Return the canonical segment rooted at ``target_nid``, or ``None``."""
    result: _Match | None = None
    tree = ir.tree
    if target_nid in tree.graph and target_nid != tree.root and isinstance(tree.data(target_nid), BlockNode):
        root_children = tree.children(tree.root)
        if target_nid in root_children:
            target_index = root_children.index(target_nid)
            if 0 < target_index < len(root_children) - 1:
                memset_block = root_children[target_index - 1]
                drain_block = root_children[target_index + 1]
                matmul_leaf = _single_leaf(tree, target_nid)
                memset_leaf = _single_leaf(tree, memset_block)
                drain_leaf = _single_leaf(tree, drain_block)
                result = _validate_segment(
                    ir,
                    memset_block=memset_block,
                    memset_leaf=memset_leaf,
                    matmul_block=target_nid,
                    matmul_leaf=matmul_leaf,
                    drain_block=drain_block,
                    drain_leaf=drain_leaf,
                )
    return result


def _validate_segment(
    ir: KernelIR,
    *,
    memset_block: int,
    memset_leaf: int | None,
    matmul_block: int,
    matmul_leaf: int | None,
    drain_block: int,
    drain_leaf: int | None,
) -> _Match | None:
    """Validate one top-level ``memset -> matmul -> drain`` segment."""
    result: _Match | None = None
    if memset_leaf is not None and matmul_leaf is not None and drain_leaf is not None:
        memset = ir.tree.isa(memset_leaf)
        matmul = ir.tree.isa(matmul_leaf)
        drain = ir.tree.isa(drain_leaf)
        if memset.op_cls is NKIMemset and matmul.op_cls is NKIMatmul and drain.op_cls is NKITensorCopy:
            output = matmul.operand_bindings["dst"].tensor
            same_output = (
                memset.operand_bindings["dst"].tensor == output and drain.operand_bindings["src"].tensor == output
            )
            canonical = (
                _is_canonical_block(ir, memset_block)
                and _is_canonical_block(ir, matmul_block)
                and _is_canonical_block(ir, drain_block)
            )
            if same_output and canonical and _eligible_buffers(ir, matmul_block, matmul, drain, output):
                touches = set(ir.dependency.touches_by_tensor.get(output, ()))
                exact_touches = touches == {memset_leaf, matmul_leaf, drain_leaf}
                direct_drain = ir.dependency.direct_consumers(matmul_leaf) == [drain_leaf]
                not_already_transposed = not _drain_feeds_transpose(ir, drain_leaf)
                if exact_touches and direct_drain and not_already_transposed:
                    result = _Match(
                        memset_block=memset_block,
                        matmul_block=matmul_block,
                        stationary=matmul.operand_bindings["stationary"].tensor,
                        moving=matmul.operand_bindings["moving"].tensor,
                        output=output,
                    )
    return result


def _single_leaf(tree: KernelTree, block_nid: int) -> int | None:
    """Return the sole ISA leaf under ``block_nid``, or ``None``."""
    result: int | None = None
    if block_nid in tree.graph and isinstance(tree.data(block_nid), BlockNode):
        leaves = [nid for nid in tree.descendants(block_nid) if isinstance(tree.data(nid), ISANode)]
        if len(leaves) == 1:
            result = leaves[0]
    return result


def _eligible_buffers(ir: KernelIR, matmul_block: int, matmul: ISANode, drain: ISANode, output: str) -> bool:
    """Whether operand residency, dtype, rank, and tile divisibility permit the rewrite."""
    buffers = ir.all_buffers()
    names = (
        matmul.operand_bindings["stationary"].tensor,
        matmul.operand_bindings["moving"].tensor,
        output,
        drain.operand_bindings["dst"].tensor,
    )
    eligible = all(name in buffers for name in names)
    if eligible:
        stationary, moving, out, drained = (buffers[name] for name in names)
        dtypes = {stationary.dtype, moving.dtype, out.dtype, drained.dtype}
        eligible = (
            stationary.location == "sbuf"
            and moving.location == "sbuf"
            and out.location == "psum"
            and out.storage_dtype == NKIMatmul.OUTPUT_STORAGE_DTYPE
            and drained.location == "sbuf"
            and len(dtypes) == 1
            and all(len(buf.shape) == 2 for buf in (stationary, moving, out, drained))
        )
    if eligible:
        block = ir.tree.block(matmul_block)
        axis_extents = _axis_extents(ir)
        k_axis = block.axis_map.get("K")
        m_axis = block.axis_map.get("M")
        n_axis = block.axis_map.get("N")
        eligible = (
            isinstance(k_axis, str)
            and isinstance(m_axis, str)
            and isinstance(n_axis, str)
            and k_axis in axis_extents
            and m_axis in axis_extents
            and n_axis in axis_extents
            and axis_extents[k_axis] % 128 == 0
            and axis_extents[m_axis] % 512 == 0
            and axis_extents[n_axis] % 512 == 0
        )
    return eligible


def _drain_feeds_transpose(ir: KernelIR, drain_leaf: int) -> bool:
    """Whether the direct drain destination is consumed by ``NKITranspose``."""
    result = False
    for consumer in ir.dependency.direct_consumers(drain_leaf):
        node = ir.tree.data(consumer)
        if isinstance(node, ISANode) and node.op_cls is NKITranspose:
            result = True
            break
    return result


def _is_canonical_block(ir: KernelIR, block_nid: int) -> bool:
    """Whether ``block_nid`` exactly matches canonical construction."""
    leaf_nid = _single_leaf(ir.tree, block_nid)
    result = False
    if leaf_nid is not None:
        leaf = ir.tree.isa(leaf_nid)
        operand_names = {slot: region.tensor for slot, region in leaf.operand_bindings.items()}
        spec = _canonical_spec(ir, leaf.op_cls, operand_names, ir.tree.block(block_nid).axis_map, leaf.kwargs)
        chain = _block_chain(ir, block_nid)
        if spec is not None and chain is not None:
            result = chain == (spec.block, *spec.loops, spec.leaf)
    return result


def _block_chain(ir: KernelIR, block_nid: int) -> tuple[BlockNode | ForNode | ISANode, ...] | None:
    """Return one unbranched block-to-ISA payload chain, or ``None``."""
    payloads: list[BlockNode | ForNode | ISANode] = [ir.tree.block(block_nid)]
    current = block_nid
    complete = False
    while not complete:
        children = ir.tree.children(current)
        if len(children) != 1:
            return None
        current = children[0]
        payload = ir.tree.data(current)
        if isinstance(payload, BlockNode):
            return None
        payloads.append(payload)
        complete = isinstance(payload, ISANode)
    return tuple(payloads)


def _axis_extents(ir: KernelIR) -> dict[str, int]:
    """Collect concrete axis extents, requiring all declarations to agree."""
    extents: dict[str, int] = {}
    for block_nid in ir.tree.blocks():
        for iter_var in ir.tree.block(block_nid).iter_vars:
            extent = iter_var.dom[1] - iter_var.dom[0]
            prior = extents.get(iter_var.axis)
            if prior is not None and prior != extent:
                raise ValueError(f"axis {iter_var.axis} has conflicting extents {prior} and {extent}")
            extents[iter_var.axis] = extent
    return extents


def _canonical_spec(
    ir: KernelIR, op_cls: type[NKIOp], operand_names: dict[str, str], axis_map: dict[str, str], kwargs: dict[str, Any]
) -> _CanonicalSpec | None:
    """Build canonical payloads for one op, or ``None`` if its tile shape is invalid."""
    extents = _axis_extents(ir)
    buffers = ir.all_buffers()
    valid = all(
        axis in axis_map and axis_map[axis] in extents for axes in op_cls.OPERAND_AXES.values() for axis in axes
    )
    valid = valid and all(name in buffers for name in operand_names.values())
    tiles: dict[str, int] = {}
    if valid:
        for abstract, concrete in axis_map.items():
            extent = extents[concrete]
            maximum = op_cls.MAX_TILE_SIZE.get(abstract)
            tile = extent if maximum is None else maximum
            if tile <= 0 or extent < tile or extent % tile != 0:
                valid = False
                break
            tiles[abstract] = tile
    result: _CanonicalSpec | None = None
    if valid:
        iter_vars: list[IterVar] = []
        iter_values: list[Const | Var] = []
        loops: list[ForNode] = []
        loop_vars: dict[str, str] = {}
        for abstract, concrete in axis_map.items():
            extent = extents[concrete]
            trip = extent // tiles[abstract]
            loop_var = f"i_{concrete}_0"
            loop_vars[abstract] = loop_var
            iter_vars.append(
                IterVar(axis=concrete, dom=(0, extent), role=op_cls.AXIS_ROLES.get(abstract, AxisRole.PARALLEL))
            )
            iter_values.append(Var(name=loop_var) if trip > 1 else Const(value=0))
            if trip > 1:
                loops.append(ForNode(loop_var=loop_var, extent=trip))
        bindings = {
            slot: _canonical_region(
                tensor=operand_names[slot],
                axes=axes,
                axis_map=axis_map,
                loop_vars=loop_vars,
                tiles=tiles,
                extents=extents,
                buffers=buffers,
            )
            for slot, axes in op_cls.OPERAND_AXES.items()
            if slot in operand_names
        }
        reads: list[BufferRegion] = []
        writes: list[BufferRegion] = []
        for slot, region in bindings.items():
            if slot in op_cls.INPUT_OPERANDS:
                reads.append(region)
            elif slot in op_cls.RMW_OPERANDS:
                reads.append(region)
                writes.append(region)
            else:
                writes.append(region)
        result = _CanonicalSpec(
            block=BlockNode(
                iter_vars=tuple(iter_vars),
                iter_values=tuple(iter_values),
                reads=tuple(reads),
                writes=tuple(writes),
                alloc_buffers=(),
                axis_map=dict(axis_map),
            ),
            loops=tuple(loops),
            leaf=ISANode(op_cls=op_cls, operand_bindings=bindings, kwargs=dict(kwargs)),
        )
    return result


def _canonical_region(
    *,
    tensor: str,
    axes: tuple[str, ...],
    axis_map: dict[str, str],
    loop_vars: dict[str, str],
    tiles: dict[str, int],
    extents: dict[str, int],
    buffers: dict[str, Buffer],
) -> BufferRegion:
    """Build one canonical operand region."""
    ranges: list[tuple[Const | Var | Mul, Const]] = []
    buffer = buffers[tensor]
    for axis_index, abstract in enumerate(axes):
        concrete = axis_map[abstract]
        tile = tiles[abstract]
        trip = extents[concrete] // tile
        if trip == 1:
            lo: Const | Var | Mul = Const(value=0)
        elif axis_index == 0 and buffer.location in {"sbuf", "psum"} and tile == PARTITION_DIM:
            lo = Var(name=loop_vars[abstract])
        else:
            lo = Mul(left=Var(name=loop_vars[abstract]), right=Const(value=tile))
        ranges.append((lo, Const(value=tile)))
    return BufferRegion(tensor=tensor, ranges=tuple(ranges))


def _apply_match(ir: KernelIR, match: _Match) -> None:
    """Apply one validated matmul transpose rewrite in place."""
    matmul_block = ir.tree.block(match.matmul_block)
    k_axis = matmul_block.axis_map["K"]
    old_m_axis = matmul_block.axis_map["M"]
    old_n_axis = matmul_block.axis_map["N"]
    old_output = ir.buffer(match.output)

    psum_name = _fresh_name(ir, f"{match.output}_swapped")
    sbuf_name = _fresh_name(ir, f"sbuf_{match.output}_swapped")
    swapped_shape = (old_output.shape[1], old_output.shape[0])
    swapped_psum = Buffer(
        name=psum_name,
        shape=swapped_shape,
        dtype=old_output.dtype,
        location="psum",
        storage_dtype=NKIMatmul.OUTPUT_STORAGE_DTYPE,
    )
    swapped_sbuf = Buffer(name=sbuf_name, shape=swapped_shape, dtype=old_output.dtype, location="sbuf")
    _replace_buffer(ir, replace(old_output, storage_dtype=NKITranspose.OUTPUT_STORAGE_DTYPE))
    _append_root_buffers(ir, (swapped_psum, swapped_sbuf))

    memset_spec = _required_spec(ir, NKIMemset, {"dst": psum_name}, {"P": old_n_axis, "F": old_m_axis}, {"value": 0.0})
    matmul_spec = _required_spec(
        ir,
        NKIMatmul,
        {"stationary": match.moving, "moving": match.stationary, "dst": psum_name},
        {"K": k_axis, "M": old_n_axis, "N": old_m_axis},
        {},
    )
    drain_spec = _required_spec(
        ir, NKITensorCopy, {"src": psum_name, "dst": sbuf_name}, {"P": old_n_axis, "F": old_m_axis}, {}
    )
    transpose_spec = _required_spec(
        ir, NKITranspose, {"data": sbuf_name, "dst": match.output}, {"P": old_n_axis, "F": old_m_axis}, {}
    )

    _rewrite_block(ir, match.memset_block, memset_spec)
    _rewrite_block(ir, match.matmul_block, matmul_spec)
    drain_block = _append_block(ir, drain_spec)
    transpose_block = _append_block(ir, transpose_spec)
    _replace_in_parent_children(
        ir.tree, ir.tree.root, [match.matmul_block], [match.matmul_block, drain_block, transpose_block]
    )


def _required_spec(
    ir: KernelIR, op_cls: type[NKIOp], operand_names: dict[str, str], axis_map: dict[str, str], kwargs: dict[str, Any]
) -> _CanonicalSpec:
    """Return a canonical spec that validated rewrite preconditions guarantee."""
    spec = _canonical_spec(ir, op_cls, operand_names, axis_map, kwargs)
    if spec is None:
        raise AssertionError(f"could not construct canonical {op_cls.__name__} block")
    return spec


def _fresh_name(ir: KernelIR, stem: str) -> str:
    """Return a deterministic buffer name not present in ``ir``."""
    names = set(ir.all_buffers())
    candidate = stem
    suffix = 1
    while candidate in names:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _replace_buffer(ir: KernelIR, replacement: Buffer) -> None:
    """Replace one declared buffer by name."""
    found = 0
    for block_nid in ir.tree.blocks():
        block = ir.tree.block(block_nid)
        updated = tuple(replacement if buffer.name == replacement.name else buffer for buffer in block.alloc_buffers)
        if updated != block.alloc_buffers:
            ir.tree.graph.nodes[block_nid]["data"] = replace(block, alloc_buffers=updated)
            found += 1
    if found != 1:
        raise AssertionError(f"expected one declaration of {replacement.name!r}, found {found}")


def _append_root_buffers(ir: KernelIR, buffers: tuple[Buffer, ...]) -> None:
    """Append new buffers to the root before placement recomputation."""
    root = ir.tree.block(ir.tree.root)
    ir.tree.graph.nodes[ir.tree.root]["data"] = replace(root, alloc_buffers=(*root.alloc_buffers, *buffers))


def _rewrite_block(ir: KernelIR, block_nid: int, spec: _CanonicalSpec) -> None:
    """Replace a block's local subtree with ``spec`` while retaining its nid."""
    descendants = list(ir.tree.descendants(block_nid))
    ir.tree.graph.remove_nodes_from(descendants)
    ir.tree.graph.nodes[block_nid]["data"] = spec.block
    parent = block_nid
    for loop in spec.loops:
        parent = ir.tree.add_node(loop, parent=parent)
    ir.tree.add_node(spec.leaf, parent=parent)


def _append_block(ir: KernelIR, spec: _CanonicalSpec) -> int:
    """Append one detached canonical block and return its nid."""
    block_nid = ir.tree.add_node(spec.block)
    parent = block_nid
    for loop in spec.loops:
        parent = ir.tree.add_node(loop, parent=parent)
    ir.tree.add_node(spec.leaf, parent=parent)
    return block_nid


__all__ = ["MatmulTranspose", "MatmulTransposeOption"]
