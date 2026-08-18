"""Commute one transpose upward through a matrix multiplication."""

from __future__ import annotations

from dataclasses import dataclass, replace

from nkigym.ir import BlockNode, Buffer, BufferRegion, Const, FloorDiv, ISANode, KernelIR, Mul
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose
from nkigym.search.buffer_placement import layout_satisfies_output_alignment
from nkigym.search.state_facts import operation_facts
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.canonical_rewrite import (
    block_chain,
    finalize_rewrite,
    is_canonical_block,
    remove_buffers,
    replace_buffer,
    single_leaf,
)
from nkigym.transforms.helper.transpose_pattern import TransposeChain, match_transpose_chain
from nkigym.transforms.helper.tree_ops import _replace_in_parent_children


@dataclass(frozen=True)
class TransposeThroughMatmulOption(TransformOption):
    """Commute the transpose block at ``transpose_nid`` through its producer."""

    transpose_nid: int


@dataclass(frozen=True)
class _Match:
    """A canonical matmul, drain, and following transpose chain."""

    memset_block: int
    matmul_block: int
    matmul_drain_block: int
    transpose: _OutputTranspose
    stationary: str
    moving: str
    old_psum: str
    old_output: str


@dataclass(frozen=True)
class _OutputTranspose:
    """One logical or DMA transpose that materializes an SBUF result."""

    blocks: tuple[int, ...]
    leaf: int
    source: str
    psum: str | None
    output: str


@dataclass(frozen=True)
class _ScheduledRewrite:
    """One block/leaf replacement with its existing loop tree retained."""

    block_nid: int
    leaf_nid: int
    block: BlockNode
    leaf: ISANode


class TransposeThroughMatmul(Transform[TransposeThroughMatmulOption]):
    """Apply ``T(A.T @ B) = B.T @ A`` to one adjacent transpose."""

    SPLIT_PREPARATION_DEPTH = 3

    def split_preparation_applicable(self, ir: KernelIR) -> bool:
        """Return whether only retained tile legality blocks one commute."""
        children = tuple(ir.tree.children(ir.tree.root))
        return any(
            _match(ir, TransposeThroughMatmulOption(block), children, index, require_legal_tiles=False) is not None
            for index, block in enumerate(children)
        )

    def analyze(self, ir: KernelIR) -> list[TransposeThroughMatmulOption]:
        """Return every transpose that can commute through a canonical matmul."""
        facts = operation_facts(ir)
        if not facts.has_ops(NKIMatmul, NKIMemset, NKITensorCopy) or not {NKITranspose, NKIDMATranspose}.intersection(
            facts.op_classes
        ):
            return []
        options: list[TransposeThroughMatmulOption] = []
        root_children = tuple(ir.tree.children(ir.tree.root))
        for index, block_nid in enumerate(root_children):
            leaf_nid = single_leaf(ir.tree, block_nid)
            if leaf_nid is None or ir.tree.isa(leaf_nid).op_cls not in {NKITranspose, NKIDMATranspose}:
                continue
            option = TransposeThroughMatmulOption(transpose_nid=block_nid)
            if _match(ir, option, root_children, index) is not None:
                options.append(option)
        return options

    def apply(self, ir: KernelIR, option: TransposeThroughMatmulOption) -> KernelIR:
        """Recheck ``option``, swap the matmul operands, and consume the transpose."""
        match = _match(ir, option)
        if match is None:
            raise TransformLegalityError(
                f"TransposeThroughMatmul target {option.transpose_nid} is not adjacent to an eligible canonical matmul"
            )
        new_ir = copy_for_rewrite(ir)
        copied_match = _match(new_ir, option)
        if copied_match is None:
            raise AssertionError("TransposeThroughMatmul match disappeared after deepcopy")
        _apply_match(new_ir, copied_match)
        finalize_rewrite(new_ir)
        return new_ir


def _match(
    ir: KernelIR,
    option: TransposeThroughMatmulOption,
    root_children: tuple[int, ...] | None = None,
    index: int | None = None,
    require_legal_tiles: bool = True,
) -> _Match | None:
    """Return one legal matmul commute."""
    result: _Match | None = None
    children = tuple(ir.tree.children(ir.tree.root)) if root_children is None else root_children
    if index is None and option.transpose_nid in children:
        index = children.index(option.transpose_nid)
    if index is not None:
        if 3 <= index:
            memset_block = children[index - 3]
            matmul_block = children[index - 2]
            matmul_drain_block = children[index - 1]
            next_block = children[index + 1] if index + 1 < len(children) else None
            transpose = _match_output_transpose(ir, option.transpose_nid, next_block)
            simple = all(_is_simple_block(ir, block) for block in (memset_block, matmul_block, matmul_drain_block))
            if transpose is not None and simple:
                result = _validate_segment(
                    ir,
                    memset_block=memset_block,
                    matmul_block=matmul_block,
                    matmul_drain_block=matmul_drain_block,
                    transpose=transpose,
                    require_legal_tiles=require_legal_tiles,
                )
    return result


def _match_output_transpose(ir: KernelIR, transpose_block: int, drain_block: int | None) -> _OutputTranspose | None:
    """Return one adjacent logical chain or direct DMA transpose."""
    logical = (
        match_transpose_chain(ir, transpose_block, drain_block, adjacent=True) if drain_block is not None else None
    )
    if logical is not None:
        return _OutputTranspose(
            blocks=(logical.transpose_block, logical.drain_block),
            leaf=logical.transpose_leaf,
            source=logical.source,
            psum=logical.psum,
            output=logical.output,
        )
    leaf = single_leaf(ir.tree, transpose_block)
    if leaf is None or not is_canonical_block(ir, transpose_block):
        return None
    operation = ir.tree.isa(leaf)
    if operation.op_cls is not NKIDMATranspose:
        return None
    source = operation.operand_bindings["src"].tensor
    output = operation.operand_bindings["dst"].tensor
    buffers = ir.all_buffers()
    if source not in buffers or output not in buffers:
        return None
    source_buffer, output_buffer = buffers[source], buffers[output]
    valid = (
        len(source_buffer.shape) == 2
        and output_buffer.shape == source_buffer.shape[::-1]
        and source_buffer.location == output_buffer.location == "sbuf"
        and source_buffer.dtype == output_buffer.dtype
        and source_buffer.physical_dtype() == output_buffer.physical_dtype() == source_buffer.dtype
    )
    return _OutputTranspose((transpose_block,), leaf, source, None, output) if valid else None


def _is_simple_block(ir: KernelIR, block_nid: int) -> bool:
    """Return whether one block has a plain schedule that can be retained."""
    chain = block_chain(ir.tree, block_nid)
    return bool(
        chain is not None
        and isinstance(chain[0], BlockNode)
        and isinstance(chain[-1], ISANode)
        and not chain[0].annotations
        and not chain[-1].access_patterns
    )


def _transpose_region(region: BufferRegion, tensor: str) -> BufferRegion:
    """Return one rank-two region with its physical dimensions exchanged."""
    if len(region.ranges) != 2:
        raise ValueError(f"cannot transpose rank-{len(region.ranges)} region {region.tensor!r}")
    (partition_lower, partition_width), (free_lower, free_width) = region.ranges
    if not isinstance(partition_width, Const) or not isinstance(free_width, Const):
        raise ValueError(f"cannot transpose symbolic-width region {region.tensor!r}")
    analyzer = Analyzer()
    transposed_partition = analyzer.simplify(FloorDiv(left=free_lower, right=free_width))
    transposed_free = analyzer.simplify(Mul(left=partition_lower, right=partition_width))
    return BufferRegion(tensor=tensor, ranges=((transposed_partition, free_width), (transposed_free, partition_width)))


def _scheduled_rewrite(
    ir: KernelIR,
    block_nid: int,
    op_cls: type[NKIMemset] | type[NKIMatmul] | type[NKITensorCopy],
    bindings: dict[str, BufferRegion],
    axis_map: dict[str, str],
    kwargs: dict[str, object],
) -> _ScheduledRewrite | None:
    """Build one replacement while retaining its current loops and tile widths."""
    leaf_nid = single_leaf(ir.tree, block_nid)
    if leaf_nid is None:
        return None
    old_block = ir.tree.block(block_nid)
    rmw = op_cls.rmw_operands(kwargs)
    reads = tuple(region for slot, region in bindings.items() if slot in op_cls.INPUT_OPERANDS or slot in rmw)
    writes = tuple(region for slot, region in bindings.items() if slot not in op_cls.INPUT_OPERANDS)
    block = replace(old_block, reads=reads, writes=writes, axis_map=axis_map)
    leaf = ISANode(op_cls=op_cls, operand_bindings=bindings, kwargs=kwargs)
    return _ScheduledRewrite(block_nid, leaf_nid, block, leaf) if _legal_tiles(block, leaf) else None


def _legal_tiles(block: BlockNode, leaf: ISANode) -> bool:
    """Return whether retained region widths satisfy the rewritten ISA contract."""
    extents = {item.axis: item.dom[1] - item.dom[0] for item in block.iter_vars}
    widths: dict[str, set[int]] = {}
    valid = all(concrete in extents for concrete in block.axis_map.values())
    for slot, region in leaf.operand_bindings.items():
        axes = leaf.op_cls.OPERAND_AXES[slot]
        valid = valid and len(region.ranges) == len(axes)
        for abstract, (_lower, width) in zip(axes, region.ranges):
            valid = valid and isinstance(width, Const)
            if isinstance(width, Const):
                widths.setdefault(abstract, set()).add(width.value)
    for abstract, concrete in block.axis_map.items():
        values = widths.get(abstract, set())
        if len(values) != 1:
            valid = False
            continue
        tile = next(iter(values))
        extent = extents[concrete]
        minimum = min(leaf.op_cls.MIN_TILE_SIZE.get(abstract, 1), extent)
        maximum = leaf.op_cls.MAX_TILE_SIZE.get(abstract)
        valid = valid and tile >= minimum and extent % tile == 0 and (maximum is None or tile <= maximum)
    return valid


def _preserved_rewrites(
    ir: KernelIR, memset_block: int, matmul_block: int, drain_block: int, target_psum: str, target_output: str
) -> tuple[_ScheduledRewrite, ...] | None:
    """Return schedule-preserving replacements for one transpose commute."""
    memset_leaf = single_leaf(ir.tree, memset_block)
    matmul_leaf = single_leaf(ir.tree, matmul_block)
    drain_leaf = single_leaf(ir.tree, drain_block)
    if memset_leaf is None or matmul_leaf is None or drain_leaf is None:
        return None
    memset = ir.tree.isa(memset_leaf)
    matmul = ir.tree.isa(matmul_leaf)
    drain = ir.tree.isa(drain_leaf)
    axes = ir.tree.block(matmul_block).axis_map
    old_m, old_n = axes["M"], axes["N"]
    rewrites = (
        _scheduled_rewrite(
            ir,
            memset_block,
            NKIMemset,
            {"dst": _transpose_region(memset.operand_bindings["dst"], target_psum)},
            {"P": old_n, "F": old_m},
            dict(memset.kwargs),
        ),
        _scheduled_rewrite(
            ir,
            matmul_block,
            NKIMatmul,
            {
                "stationary": matmul.operand_bindings["moving"],
                "moving": matmul.operand_bindings["stationary"],
                "dst": _transpose_region(matmul.operand_bindings["dst"], target_psum),
            },
            {"K": axes["K"], "M": old_n, "N": old_m},
            dict(matmul.kwargs),
        ),
        _scheduled_rewrite(
            ir,
            drain_block,
            NKITensorCopy,
            {
                "src": _transpose_region(drain.operand_bindings["src"], target_psum),
                "dst": _transpose_region(drain.operand_bindings["dst"], target_output),
            },
            {"P": old_n, "F": old_m},
            dict(drain.kwargs),
        ),
    )
    return None if any(item is None for item in rewrites) else tuple(item for item in rewrites if item is not None)


def _rewritten_psum_buffer(
    ir: KernelIR, target_psum: str, target_output: str, rewrites: tuple[_ScheduledRewrite, ...]
) -> Buffer | None:
    """Return the transposed accumulator allocation when its layout is valid."""
    partition_width = rewrites[0].leaf.operand_bindings["dst"].ranges[0][1]
    shape = ir.buffer(target_output).shape
    result: Buffer | None = None
    if isinstance(partition_width, Const):
        candidate = replace(
            ir.buffer(target_psum),
            shape=shape,
            storage_dtype=NKIMatmul.OUTPUT_STORAGE_DTYPE,
            partition_size=partition_width.value,
        )
        tiles = shape[0] // partition_width.value
        if (
            1 <= partition_width.value <= 128
            and shape[0] % partition_width.value == 0
            and tiles % candidate.list_len == 0
            and layout_satisfies_output_alignment(ir.tree, candidate)
        ):
            result = candidate
    return result


def _validate_segment(
    ir: KernelIR,
    *,
    memset_block: int,
    matmul_block: int,
    matmul_drain_block: int,
    transpose: _OutputTranspose,
    require_legal_tiles: bool,
) -> _Match | None:
    """Validate one ``matmul -> drain -> transpose`` segment."""
    result: _Match | None = None
    memset_leaf = single_leaf(ir.tree, memset_block)
    matmul_leaf = single_leaf(ir.tree, matmul_block)
    drain_leaf = single_leaf(ir.tree, matmul_drain_block)
    if memset_leaf is not None and matmul_leaf is not None and drain_leaf is not None:
        memset = ir.tree.isa(memset_leaf)
        matmul = ir.tree.isa(matmul_leaf)
        drain = ir.tree.isa(drain_leaf)
        operations = memset.op_cls is NKIMemset and matmul.op_cls is NKIMatmul and drain.op_cls is NKITensorCopy
        if operations:
            old_psum = matmul.operand_bindings["dst"].tensor
            old_output = drain.operand_bindings["dst"].tensor
            connected = (
                memset.operand_bindings["dst"].tensor == old_psum
                and drain.operand_bindings["src"].tensor == old_psum
                and transpose.source == old_output
            )
            buffers = ir.all_buffers()
            names = (
                matmul.operand_bindings["stationary"].tensor,
                matmul.operand_bindings["moving"].tensor,
                old_psum,
                old_output,
                transpose.output,
            )
            if transpose.psum is not None:
                names = (*names, transpose.psum)
            if connected and all(name in buffers for name in names):
                stationary, moving, old_psum_buffer, old_output_buffer, transpose_output = (
                    buffers[name] for name in names[:5]
                )
                transpose_psum = buffers[transpose.psum] if transpose.psum is not None else None
                matmul_block_data = ir.tree.block(matmul_block)
                axes = all(axis in matmul_block_data.axis_map for axis in ("K", "M", "N"))
                rank_two = all(
                    len(buffer.shape) == 2
                    for buffer in (stationary, moving, old_psum_buffer, old_output_buffer, transpose_output)
                )
                rank_two = rank_two and (transpose_psum is None or len(transpose_psum.shape) == 2)
                shapes = rank_two and (
                    stationary.shape[0] == moving.shape[0]
                    and old_psum_buffer.shape == (stationary.shape[1], moving.shape[1])
                    and old_output_buffer.shape == old_psum_buffer.shape
                    and transpose_output.shape == old_output_buffer.shape[::-1]
                    and (transpose_psum is None or transpose_psum.shape == transpose_output.shape)
                )
                storage = (
                    stationary.location == "sbuf"
                    and moving.location == "sbuf"
                    and old_psum_buffer.location == "psum"
                    and old_output_buffer.location == "sbuf"
                    and transpose_output.location == "sbuf"
                    and (transpose_psum is None or transpose_psum.location == "psum")
                )
                dtype_buffers = (stationary, moving, old_psum_buffer, old_output_buffer, transpose_output)
                dtype = len({buffer.dtype for buffer in dtype_buffers}) == 1 and (
                    transpose_psum is None or transpose_psum.dtype == stationary.dtype
                )
                physical_dtype = (
                    old_psum_buffer.storage_dtype == NKIMatmul.OUTPUT_STORAGE_DTYPE
                    and stationary.physical_dtype() == stationary.dtype
                    and moving.physical_dtype() == moving.dtype
                    and (transpose_psum is None or transpose_psum.storage_dtype == NKITranspose.OUTPUT_STORAGE_DTYPE)
                )
                target_psum = old_psum if transpose_psum is None else transpose_psum.name
                rewrites = (
                    _preserved_rewrites(
                        ir, memset_block, matmul_block, matmul_drain_block, target_psum, transpose.output
                    )
                    if axes and require_legal_tiles
                    else ()
                )
                allocation = (
                    None
                    if rewrites is None
                    else (
                        True
                        if not require_legal_tiles
                        else _rewritten_psum_buffer(ir, target_psum, transpose.output, rewrites)
                    )
                )
                exact_old_psum = set(ir.dependency.touches_by_tensor.get(old_psum, ())) == {
                    memset_leaf,
                    matmul_leaf,
                    drain_leaf,
                }
                exact_old_output = set(ir.dependency.touches_by_tensor.get(old_output, ())) == {
                    drain_leaf,
                    transpose.leaf,
                }
                if (
                    axes
                    and shapes
                    and storage
                    and dtype
                    and physical_dtype
                    and rewrites is not None
                    and allocation is not None
                    and exact_old_psum
                    and exact_old_output
                ):
                    result = _Match(
                        memset_block=memset_block,
                        matmul_block=matmul_block,
                        matmul_drain_block=matmul_drain_block,
                        transpose=transpose,
                        stationary=stationary.name,
                        moving=moving.name,
                        old_psum=old_psum,
                        old_output=old_output,
                    )
    return result


def _apply_match(ir: KernelIR, match: _Match) -> None:
    """Swap the matmul and consume the following transpose."""
    target_psum = match.old_psum if match.transpose.psum is None else match.transpose.psum
    rewrites = _preserved_rewrites(
        ir, match.memset_block, match.matmul_block, match.matmul_drain_block, target_psum, match.transpose.output
    )
    if rewrites is None:
        raise AssertionError("matched transpose commute lost its retained schedule")
    target_buffer = _rewritten_psum_buffer(ir, target_psum, match.transpose.output, rewrites)
    if target_buffer is None:
        raise AssertionError("matched transpose commute lost its accumulator allocation")
    replace_buffer(ir, target_buffer)
    for rewrite in rewrites:
        ir.tree.graph.nodes[rewrite.block_nid]["data"] = rewrite.block
        ir.tree.graph.nodes[rewrite.leaf_nid]["data"] = rewrite.leaf

    removed_blocks = list(match.transpose.blocks)
    _replace_in_parent_children(ir.tree, ir.tree.root, removed_blocks, [])
    for block_nid in removed_blocks:
        ir.tree.graph.remove_nodes_from({block_nid, *ir.tree.descendants(block_nid)})
    remove_buffers(ir, {match.old_output} | ({match.old_psum} if target_psum != match.old_psum else set()))


__all__ = ["TransposeThroughMatmul", "TransposeThroughMatmulOption"]
