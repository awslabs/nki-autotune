"""Propagate a copy source into a co-located consumer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from weakref import WeakKeyDictionary

from nkigym.ir import Dependency, KernelIR
from nkigym.ir.arith import LE, Add, Analyzer, Const, FloorDiv, Mul, Sub
from nkigym.ir.arith.expr import Expr, NonAffineError, Var, affine_terms, substitute
from nkigym.ir.canonical_build import _build_subblock
from nkigym.ir.dimension_analysis import TensorDims, _AnalysisResult, _OpRecord
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.program_sharding import configured_program_shards, owning_block
from nkigym.ir.tree import AccessPattern, BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.base import CopyContract, PointwiseContract, SliceContract
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.interleaved_left_load import NKIInterleavedLeftLoad
from nkigym.ops.interleaved_stationary_load import NKIInterleavedStationaryLoad
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.ops.transpose_store import NKITransposeStore
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    intersects_software_pipeline,
    software_pipeline_overlap_nodes,
)
from nkigym.transforms.helper.canonical_rewrite import (
    CanonicalSpec,
    axis_extents,
    block_chain,
    finalize_rewrite,
    is_canonical_block,
    rewrite_block,
)

_LOOP_ACCESS_INDEPENDENCE: WeakKeyDictionary[Dependency, dict[tuple[int, int], bool]] = WeakKeyDictionary()


@dataclass(frozen=True)
class CopyPropagationOption(TransformOption):
    """Identify one ordered copy and consumer input."""

    copy_block_nid: int
    consumer_block_nid: int
    consumer_operand: str


@dataclass(frozen=True)
class _CopyPropagationMatch:
    """Resolved copy propagation with exact source and destination regions."""

    option: CopyPropagationOption
    copy_leaf_nid: int
    consumer_leaf_nid: int
    source: BufferRegion
    copied: BufferRegion
    access_pattern: AccessPattern | None


@dataclass(frozen=True)
class _TransposeForwardingMatch:
    """One consumer replacement with the source transpose retained."""

    block_nid: int
    spec: CanonicalSpec


class CopyPropagation(Transform[CopyPropagationOption]):
    """Substitute one value-preserving copy source into its consumer."""

    def analyze(self, ir: KernelIR) -> list[CopyPropagationOption]:
        """Return ordered copy-consumer pairs accepted by storage contracts."""
        options: list[CopyPropagationOption] = []
        buffers = ir.all_buffers()
        overlap_nodes = software_pipeline_overlap_nodes(ir)
        positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
        analyzer = _projection_analyzer(ir)
        for copy_leaf_nid in ir.dependency.graph.nodes:
            copy_leaf = ir.tree.isa(copy_leaf_nid)
            if not isinstance(_copy_contract(copy_leaf), (CopyContract, SliceContract)) and copy_leaf.op_cls not in {
                NKIDMATranspose,
                NKITransposeStore,
            }:
                continue
            copy_nid = owning_block(ir, copy_leaf_nid)
            for consumer_leaf_nid in ir.dependency.direct_consumers(copy_leaf_nid):
                consumer_nid = owning_block(ir, consumer_leaf_nid)
                if consumer_nid == copy_nid:
                    continue
                consumer = ir.tree.isa(consumer_leaf_nid)
                for operand in consumer.op_cls.INPUT_OPERANDS:
                    option = CopyPropagationOption(copy_nid, consumer_nid, operand)
                    if self._resolve(ir, option, buffers, overlap_nodes, positions=positions, analyzer=analyzer) or (
                        _transpose_forwarding(ir, option, positions, overlap_nodes) is not None
                    ):
                        options.append(option)
        return options

    def apply(self, ir: KernelIR, option: CopyPropagationOption) -> KernelIR:
        """Recheck, copy, and propagate the source region into one consumer."""
        match = self._resolve(ir, option, ir.all_buffers())
        transpose = _transpose_forwarding(ir, option)
        if match is None and transpose is None:
            raise TransformLegalityError(f"illegal CopyPropagation option: {option}")
        new_ir = copy_for_rewrite(ir)
        if transpose is None:
            copied_match = self._resolve(new_ir, option, new_ir.all_buffers())
            if copied_match is None:
                raise AssertionError(f"CopyPropagation option disappeared after deepcopy: {option}")
            self._rewrite(new_ir, copied_match)
        else:
            copied_transpose = _transpose_forwarding(new_ir, option)
            if copied_transpose is None:
                raise AssertionError(f"transpose forwarding disappeared after deepcopy: {option}")
            rewrite_block(new_ir.tree, copied_transpose.block_nid, copied_transpose.spec)
            finalize_rewrite(new_ir)
        return new_ir

    def _resolve(
        self,
        ir: KernelIR,
        option: CopyPropagationOption,
        buffers: dict[str, Buffer],
        overlap_nodes: frozenset[int] | None = None,
        ordered: bool | None = None,
        positions: dict[int, int] | None = None,
        analyzer: Analyzer | None = None,
    ) -> _CopyPropagationMatch | None:
        """Resolve an option when copy semantics, storage, and use-def all agree."""
        result: _CopyPropagationMatch | None = None
        copy_nid = option.copy_block_nid
        consumer_nid = option.consumer_block_nid
        if copy_nid not in ir.tree.graph or consumer_nid not in ir.tree.graph:
            return result
        if not isinstance(ir.tree.data(copy_nid), BlockNode) or not isinstance(ir.tree.data(consumer_nid), BlockNode):
            return result
        if intersects_software_pipeline(ir, (copy_nid, consumer_nid), overlap_nodes):
            return result
        copy_scope = tuple(nid for nid in ir.tree.ancestors(copy_nid) if isinstance(ir.tree.data(nid), ForNode))
        consumer_scope = tuple(nid for nid in ir.tree.ancestors(consumer_nid) if isinstance(ir.tree.data(nid), ForNode))
        if copy_scope != consumer_scope:
            return result
        if positions is None:
            positions = {nid: index for index, nid in enumerate(ir.tree.preorder())}
        copy_leaf_nid = _owned_leaf(ir, copy_nid)
        consumer_leaf_nid = _owned_leaf(ir, consumer_nid)
        if copy_leaf_nid is None or consumer_leaf_nid is None:
            return result
        if ordered is None:
            parent = ir.tree.parent(copy_nid)
            siblings = ir.tree.children(parent) if parent is not None else []
            sibling_ordered = (
                parent is not None
                and ir.tree.parent(consumer_nid) == parent
                and copy_nid in siblings
                and consumer_nid in siblings
                and siblings.index(copy_nid) < siblings.index(consumer_nid)
            )
            ordered = sibling_ordered or (
                copy_leaf_nid in ir.dependency.direct_producers(consumer_leaf_nid)
                and positions[copy_leaf_nid] < positions[consumer_leaf_nid]
            )
        if not ordered:
            return result
        copy_leaf = ir.tree.isa(copy_leaf_nid)
        consumer_leaf = ir.tree.isa(consumer_leaf_nid)
        contract = _copy_contract(copy_leaf)
        if not isinstance(contract, (CopyContract, SliceContract)):
            return result
        source = copy_leaf.operand_bindings.get(contract.input_operand)
        copied = copy_leaf.operand_bindings.get(contract.output_operand)
        consumed = consumer_leaf.operand_bindings.get(option.consumer_operand)
        if source is None or copied is None or consumed is None or consumed.tensor != copied.tensor:
            return result
        copied_buffer = buffers[copied.tensor]
        source_buffer = buffers[source.tensor]
        dma_promotion = _lossless_dma_promotion(source_buffer, copied_buffer, consumer_leaf, buffers)
        hbm_identity = (
            isinstance(contract, CopyContract)
            and source_buffer.location == copied_buffer.location == "shared_hbm"
            and source_buffer.shape == copied_buffer.shape
            and (
                dma_promotion
                or source_buffer.dtype == copied_buffer.dtype
                and source_buffer.physical_dtype() == copied_buffer.physical_dtype()
            )
        )
        if copy_leaf.access_patterns or (
            not hbm_identity
            and (
                option.consumer_operand in consumer_leaf.access_patterns
                or consumer_leaf.access_patterns.keys() - consumer_leaf.op_cls.INPUT_OPERANDS
            )
        ):
            return result
        access_pattern = None
        if isinstance(contract, SliceContract) and contract.pattern:
            access_pattern = _strided_source_pattern(
                contract, source, copied, consumed, consumer_leaf, option.consumer_operand, buffers[source.tensor]
            )
            if access_pattern is None:
                return result
        elif isinstance(contract, SliceContract):
            if contract.start < 0 or contract.width < 1 or not 0 <= contract.axis < len(source.ranges):
                return result
            extent = source.ranges[contract.axis][1]
            if not isinstance(extent, Const) or contract.start + contract.width > extent.value:
                return result
            source = source.with_partition_aligned_slice(contract.axis, contract.start, contract.width)
        if access_pattern is None:
            projection_analyzer = _projection_analyzer(ir) if analyzer is None else analyzer
            projected = None if hbm_identity else _project_source_region(projection_analyzer, source, copied, consumed)
            if projected is None and isinstance(contract, CopyContract):
                projected = _project_complete_copy(
                    ir, copy_nid, copy_leaf_nid, source, copied, consumed, projection_analyzer, dma_promotion
                )
            source = projected
        if source is None:
            return result
        if copied.tensor in ir.param_buffers or copied.tensor in ir.return_names:
            return result
        if copied_buffer.location == "shared_hbm" and not hbm_identity:
            if (
                copy_leaf.op_cls is not NKIStore
                or consumer_leaf.op_cls is not NKILoad
                or source_buffer.location != "sbuf"
                or copy_leaf.kwargs
                or consumer_leaf.kwargs
            ):
                return result
        accepted_locations = consumer_leaf.op_cls.INPUT_LOCATIONS.get(option.consumer_operand)
        locations = {
            operand: buffers[region.tensor].location
            for operand, region in consumer_leaf.operand_bindings.items()
            if operand in consumer_leaf.op_cls.INPUT_OPERANDS
        }
        locations[option.consumer_operand] = source_buffer.location
        dtypes = {
            operand: buffers[region.tensor].physical_dtype()
            for operand, region in consumer_leaf.operand_bindings.items()
            if operand in consumer_leaf.op_cls.INPUT_OPERANDS
        }
        dtypes[option.consumer_operand] = source_buffer.physical_dtype()
        required_dtype = consumer_leaf.op_cls.REQUIRED_INPUT_STORAGE_DTYPES.get(option.consumer_operand)
        accepted_dtypes = consumer_leaf.op_cls.INPUT_STORAGE_DTYPES.get(option.consumer_operand, frozenset())
        storage_compatible = (
            source_buffer.physical_dtype() == copied_buffer.physical_dtype()
            or source_buffer.physical_dtype() in {"bfloat16", "float16"}
            and copied_buffer.physical_dtype() == "float32"
            and (source_buffer.physical_dtype() in accepted_dtypes or dma_promotion)
        )
        if (
            source_buffer.location not in {"shared_hbm", "sbuf", "psum"}
            or accepted_locations is None
            or source_buffer.location not in accepted_locations
            or (
                source_buffer.dtype != copied_buffer.dtype
                and not (
                    source_buffer.physical_dtype() in {"bfloat16", "float16"}
                    and copied_buffer.physical_dtype() == "float32"
                    and (source_buffer.physical_dtype() in accepted_dtypes or dma_promotion)
                )
            )
            or not storage_compatible
            or not consumer_leaf.op_cls.accepts_input_locations(locations)
            or not consumer_leaf.op_cls.accepts_input_storage_dtypes(dtypes)
            or (required_dtype is not None and source_buffer.physical_dtype() != required_dtype)
            or source.tensor == copied.tensor
            or len(source.ranges) != len(copied.ranges)
            or not _location_tile_compatible(consumer_leaf, option.consumer_operand, source, source_buffer)
        ):
            return result
        copy_block = ir.tree.block(copy_nid)
        if any(buffer.name != copied.tensor for buffer in copy_block.alloc_buffers):
            return result
        if not self._has_unique_definition(ir, copied.tensor, copy_leaf_nid, consumer_leaf_nid):
            if access_pattern is None or not _copied_definition_reaches(
                ir, copied.tensor, copy_leaf_nid, consumer_leaf_nid, positions
            ):
                return result
        if not source_remains_stable(
            ir, copy_leaf.operand_bindings[contract.input_operand], copy_leaf_nid, consumer_leaf_nid, positions
        ):
            return result
        result = _CopyPropagationMatch(
            option=option,
            copy_leaf_nid=copy_leaf_nid,
            consumer_leaf_nid=consumer_leaf_nid,
            source=source,
            copied=consumed,
            access_pattern=access_pattern,
        )
        return result

    def _has_unique_definition(self, ir: KernelIR, tensor: str, copy_leaf_nid: int, consumer_leaf_nid: int) -> bool:
        """Require one defining copy and a selected reader; other readers keep the copy live."""
        readers: list[int] = []
        writers: list[int] = []
        for nid in ir.dependency.touches_by_tensor.get(tensor, ()):
            node = ir.tree.data(nid)
            assert isinstance(node, ISANode)
            rmw_operands = node.op_cls.rmw_operands(node.kwargs)
            for slot, region in node.operand_bindings.items():
                if region.tensor != tensor:
                    continue
                if slot in node.op_cls.INPUT_OPERANDS or slot in rmw_operands:
                    readers.append(nid)
                if slot not in node.op_cls.INPUT_OPERANDS:
                    writers.append(nid)
        return consumer_leaf_nid in readers and writers == [copy_leaf_nid]

    def _rewrite(self, ir: KernelIR, match: _CopyPropagationMatch) -> None:
        """Retarget the consumer and rebuild derived metadata."""
        consumer = ir.tree.isa(match.consumer_leaf_nid)
        bindings = dict(consumer.operand_bindings)
        bindings[match.option.consumer_operand] = match.source
        patterns = dict(consumer.access_patterns)
        if match.access_pattern is not None:
            patterns[match.option.consumer_operand] = match.access_pattern
        ir.tree.graph.nodes[match.consumer_leaf_nid]["data"] = replace(
            consumer, operand_bindings=bindings, access_patterns=patterns
        )

        consumer_block = ir.tree.block(match.option.consumer_block_nid)
        replaced = False
        reads: list[BufferRegion] = []
        for region in consumer_block.reads:
            if region == match.copied and not replaced:
                reads.append(match.source)
                replaced = True
            else:
                reads.append(region)
        if not replaced:
            raise AssertionError(
                f"consumer block {match.option.consumer_block_nid} does not read {match.copied.tensor!r}"
            )
        ir.tree.graph.nodes[match.option.consumer_block_nid]["data"] = replace(consumer_block, reads=tuple(reads))
        finalize_rewrite(ir)


def _transpose_forwarding(
    ir: KernelIR,
    option: CopyPropagationOption,
    positions: dict[int, int] | None = None,
    overlap_nodes: frozenset[int] | None = None,
) -> _TransposeForwardingMatch | None:
    """Forward one complete transpose into a native HBM copy or packing load."""
    if option.consumer_operand != "src":
        return None
    producer, consumer = option.copy_block_nid, option.consumer_block_nid
    if any(
        nid not in ir.tree.graph or not isinstance(ir.tree.data(nid), BlockNode) or ir.tree.parent(nid) != ir.tree.root
        for nid in (producer, consumer)
    ):
        return None
    copy_leaf, use_leaf = _owned_leaf(ir, producer), _owned_leaf(ir, consumer)
    if copy_leaf is None or use_leaf is None:
        return None
    copied_op, used_op = ir.tree.isa(copy_leaf), ir.tree.isa(use_leaf)
    pair = (copied_op.op_cls, used_op.op_cls)
    if pair not in {(NKIDMATranspose, NKIStore), (NKITransposeStore, NKIInterleavedStationaryLoad)}:
        return None
    if intersects_software_pipeline(ir, (producer, consumer), overlap_nodes) or any(
        ir.tree.block(nid).annotations
        or ir.tree.block(nid).alloc_buffers
        or configured_program_shards(ir).keys() & set(ir.tree.descendants(nid))
        for nid in (producer, consumer)
    ):
        return None
    if set(copied_op.kwargs) - {"name"} or set(used_op.operand_bindings) != {"src", "dst"}:
        return None
    source, copied, output = (
        ir.buffer(copied_op.operand_bindings["src"].tensor),
        ir.buffer(copied_op.operand_bindings["dst"].tensor),
        ir.buffer(used_op.operand_bindings["dst"].tensor),
    )
    if (
        used_op.operand_bindings["src"].tensor != copied.name
        or len(source.shape) != 2
        or copied.shape != source.shape[::-1]
        or source.name == copied.name
        or source.versions != 1
        or source.location not in {"sbuf", "shared_hbm"}
        or any(
            value.dtype != source.dtype or value.physical_dtype() != source.physical_dtype()
            for value in (copied, output)
        )
        or copied.name in ir.param_buffers
        or copied.name in ir.return_names
        or not CopyPropagation()._has_unique_definition(ir, copied.name, copy_leaf, use_leaf)
    ):
        return None
    ordered = positions if positions is not None else {nid: i for i, nid in enumerate(ir.tree.preorder())}
    full = BufferRegion(tensor=source.name, ranges=tuple((Const(value=0), Const(value=n)) for n in source.shape))
    if not source_remains_stable(ir, full, copy_leaf, use_leaf, ordered):
        return None
    spec: CanonicalSpec | None = None
    if copied_op.op_cls is NKIDMATranspose:
        if not is_canonical_block(ir, producer) or copied_op.access_patterns:
            return None
        spec = _store_transpose_spec(ir, consumer, source, copied)
    elif (
        source.location == "shared_hbm"
        and copied.location == "shared_hbm"
        and _complete_hbm_rectangle(ir, producer, copy_leaf, copied_op.operand_bindings["dst"], copied)
        and copied_op == _transpose_store_leaf(source, copied_op.operand_bindings["dst"], copied_op.kwargs)
    ):
        original = _packing_spec(ir, consumer, copied.name, NKIInterleavedStationaryLoad)
        chain = block_chain(ir.tree, consumer)
        if original is not None and chain == (original.block, *original.loops, original.leaf):
            spec = _packing_spec(ir, consumer, source.name, NKIInterleavedLeftLoad)
            if spec is not None and (
                spec.loops != original.loops
                or spec.block.writes != original.block.writes
                or spec.leaf.access_patterns.get("dst") != original.leaf.access_patterns.get("dst")
            ):
                return None
    return _TransposeForwardingMatch(consumer, spec) if spec is not None else None


def _store_transpose_spec(ir: KernelIR, block_nid: int, source: Buffer, transposed: Buffer) -> CanonicalSpec | None:
    """Preserve a store's loops while substituting its transposed input view."""
    chain = block_chain(ir.tree, block_nid)
    if chain is None or not isinstance(chain[-1], ISANode):
        return None
    node = chain[-1]
    output = ir.buffer(node.operand_bindings["dst"].tensor)
    if (
        output.location != "shared_hbm"
        or output.shape != transposed.shape
        or node.access_patterns
        or set(node.kwargs) - {"name"}
    ):
        return None
    read, write = node.operand_bindings["src"], node.operand_bindings["dst"]
    analyzer = Analyzer()
    normalized = (
        (Mul(left=read.ranges[0][0], right=Const(value=transposed.partition_extent())), read.ranges[0][1]),
        *read.ranges[1:],
    )
    if any(
        not analyzer.can_prove_equal(left, right)
        for old, new in zip(normalized, write.ranges, strict=True)
        for left, right in zip(old, new, strict=True)
    ):
        return None
    replacement = _transpose_store_leaf(source, write, node.kwargs)
    if replacement is None:
        return None
    block = chain[0]
    if not isinstance(block, BlockNode) or set(block.axis_map) != {"P", "F"}:
        return None
    mapping = {("F" if axis == "P" else "P"): concrete for axis, concrete in block.axis_map.items()}
    loops = tuple(node for node in chain[1:-1] if isinstance(node, ForNode))
    return CanonicalSpec(
        replace(block, reads=(replacement.operand_bindings["src"],), axis_map=mapping), loops, replacement
    )


def _transpose_store_leaf(source: Buffer, output: BufferRegion, kwargs: dict[str, object]) -> ISANode | None:
    """Encode one unchanged transpose tile using only DMA source/destination views."""
    if len(output.ranges) != 2 or source.versions != 1 or source.list_len != 1:
        return None
    (column, width), (row, height) = output.ranges
    if not isinstance(width, Const) or not isinstance(height, Const) or not 1 <= height.value <= 128:
        return None
    analyzer = Analyzer()
    stride = source.shape[1]
    leading = row
    offset = Add(left=Mul(left=row, right=Const(value=stride)), right=column)
    if source.location == "sbuf":
        extent = source.partition_extent()
        leading = analyzer.simplify(FloorDiv(left=row, right=Const(value=extent)))
        if height.value != extent or not analyzer.can_prove_equal(Mul(left=leading, right=Const(value=extent)), row):
            return None
        physical = source.per_tile_physical_shape()
        stride = physical[1] * physical[2]
        offset = Add(left=Mul(left=leading, right=Const(value=physical[2])), right=column)
    elif source.location != "shared_hbm":
        return None
    region = BufferRegion(tensor=source.name, ranges=((leading, height), (column, width)))
    patterns = {
        "src": AccessPattern(
            pattern=((Const(value=stride), height), (Const(value=1), width)), offset=analyzer.simplify(offset)
        ),
        "dst": AccessPattern(
            pattern=((Const(value=1), height), (Const(value=source.shape[0]), width)),
            offset=analyzer.simplify(Add(left=Mul(left=column, right=Const(value=source.shape[0])), right=row)),
        ),
    }
    return ISANode(
        op_cls=NKITransposeStore,
        operand_bindings={"src": region, "dst": output},
        kwargs=kwargs,
        access_patterns=patterns,
    )


def _complete_hbm_rectangle(ir: KernelIR, block_nid: int, leaf_nid: int, region: BufferRegion, buffer: Buffer) -> bool:
    """Prove that one unsharded loop chain visits every HBM element exactly once."""
    chain = block_chain(ir.tree, block_nid)
    if chain is None or _owned_leaf(ir, block_nid) != leaf_nid:
        return False
    loops = {Var(name=node.loop_var): node.extent for node in chain[1:-1] if isinstance(node, ForNode)}
    used: set[Var] = set()
    for (lower, width), extent in zip(region.ranges, buffer.shape, strict=True):
        if not isinstance(width, Const) or width.value < 1:
            return False
        try:
            terms = affine_terms(lower)
        except NonAffineError:
            return False
        if terms.pop(None, 0) != 0:
            return False
        covered = width.value
        for variable, stride in sorted(terms.items(), key=lambda item: item[1]):
            if not isinstance(variable, Var) or variable not in loops or variable in used or stride != covered:
                return False
            covered *= loops[variable]
            used.add(variable)
        if covered != extent:
            return False
    return used == set(loops)


def _packing_spec(
    ir: KernelIR,
    block_nid: int,
    source: str,
    operation: type[NKIInterleavedLeftLoad] | type[NKIInterleavedStationaryLoad],
) -> CanonicalSpec | None:
    """Construct one complete packing-load instruction with its existing axis identities."""
    leaf_nid = _owned_leaf(ir, block_nid)
    if leaf_nid is None:
        return None
    leaf, block = ir.tree.isa(leaf_nid), ir.tree.block(block_nid)
    output = ir.buffer(leaf.operand_bindings["dst"].tensor)
    if (
        output.versions != 1
        or output.list_len != 1
        or output.free_alignment != 1
        or set(leaf.kwargs) - {"tiles", "name"}
    ):
        return None
    tensors = {
        name: TensorDims(name, value.shape, (), value.location, value.dtype, value.storage_dtype)
        for name, value in ir.all_buffers().items()
    }
    analysis = _AnalysisResult("", [], (), axis_extents(ir), tensors, [], [])
    record = _OpRecord(operation, {"src": source, "dst": output.name}, block.axis_map, leaf.kwargs)
    tree = KernelTree()
    target = _build_subblock(tree, tree.root, record, analysis)
    chain = block_chain(tree, target)
    if chain is None or not isinstance(chain[0], BlockNode) or not isinstance(chain[-1], ISANode):
        return None
    return CanonicalSpec(chain[0], tuple(node for node in chain[1:-1] if isinstance(node, ForNode)), chain[-1])


def _copied_definition_reaches(
    ir: KernelIR, tensor: str, copy_leaf: int, consumer_leaf: int, positions: dict[int, int]
) -> bool:
    """Prove the selected strided copy is the last write in one straight-line scope.

    Other definitions before the copy or after the selected read may reuse its
    storage. A conditional operation scope or any intervening write prevents
    this proof. The caller separately checks matching loop scopes, complete
    copied-view coverage, and stability of the original source through the read.
    """
    copy_block, consumer_block = (owning_block(ir, nid) for nid in (copy_leaf, consumer_leaf))
    if (
        ir.tree.parent(copy_block) != ir.tree.parent(consumer_block)
        or any("predicate" in ir.tree.block(nid).annotations for nid in (copy_block, consumer_block))
        or ir.tree.isa(copy_leaf).op_cls.rmw_operands(ir.tree.isa(copy_leaf).kwargs)
    ):
        return False
    first, last = positions[copy_leaf], positions[consumer_leaf]
    if first >= last:
        return False
    return not any(
        first < positions[nid] < last
        and any(region.tensor == tensor for region in ir.dependency.info(nid).write_regions)
        for nid in ir.dependency.touches_by_tensor.get(tensor, ())
    )


def _strided_source_pattern(
    contract: SliceContract,
    source: BufferRegion,
    copied: BufferRegion,
    consumed: BufferRegion,
    consumer: ISANode,
    operand: str,
    source_buffer: Buffer,
) -> AccessPattern | None:
    """Preserve one complete strided view in an elementwise consumer.

    Keep the original source footprint for dependency and lifetime checks.
    The physical view is the same one the removed materialization would read.
    Partial copied tiles and non-elementwise consumers require other view
    transformations and are outside this match.
    """
    pointwise = consumer.op_cls.algebraic_contract(consumer.kwargs)
    if (
        not isinstance(pointwise, PointwiseContract)
        or consumer.op_cls.OPERAND_AXES[operand] != consumer.op_cls.OPERAND_AXES[pointwise.output_operand]
        or source_buffer.location != "sbuf"
        or contract.axis != 1
        or copied != consumed
        or len(source.ranges) != 2
        or len(copied.ranges) != 2
        or source.ranges[0][1] != copied.ranges[0][1]
        or not 1 <= len(contract.pattern) <= 4
        or contract.start < 0
        or any(stride < 0 or extent < 1 for stride, extent in contract.pattern)
    ):
        return None
    width = source.ranges[1][1]
    selected_width = copied.ranges[1][1]
    last = contract.start + sum(stride * (extent - 1) for stride, extent in contract.pattern)
    if (
        not isinstance(width, Const)
        or not isinstance(selected_width, Const)
        or contract.width != selected_width.value
        or prod(extent for _, extent in contract.pattern) != contract.width
        or last >= width.value
    ):
        return None
    free = source_buffer.per_tile_physical_shape()[2]
    return AccessPattern(
        pattern=(
            (Const(value=source_buffer.logical_tile_count() * free), source.ranges[0][1]),
            *((Const(value=stride), Const(value=extent)) for stride, extent in contract.pattern),
        ),
        offset=Add(
            left=Mul(left=source.ranges[0][0], right=Const(value=free)),
            right=Add(left=source.ranges[1][0], right=Const(value=contract.start)),
        ),
    )


def _copy_contract(leaf: ISANode) -> CopyContract | SliceContract | None:
    """Recognize copies and native lossless floating-point promotions."""
    contract = leaf.op_cls.algebraic_contract(leaf.kwargs)
    if (
        contract == PointwiseContract(operator="copy", input_operands=("data",), output_operand="dst")
        and leaf.op_cls.OUTPUT_DTYPE == "float32"
        and leaf.op_cls.OUTPUT_STORAGE_DTYPE == "float32"
    ):
        contract = CopyContract(input_operand="data", output_operand="dst")
    return contract if isinstance(contract, (CopyContract, SliceContract)) else None


def source_remains_stable(
    ir: KernelIR, source: BufferRegion, copy_leaf_nid: int, consumer_leaf_nid: int, positions: dict[int, int]
) -> bool:
    """Require a live source allocation and no writes before the delayed read."""
    copy_position = positions[copy_leaf_nid]
    consumer_position = positions[consumer_leaf_nid]
    if copy_position >= consumer_position:
        return False
    consumer_ancestors = set(ir.tree.ancestors(consumer_leaf_nid))
    copy_ancestors = ir.tree.ancestors(copy_leaf_nid)
    for nid in copy_ancestors:
        node = ir.tree.data(nid)
        if nid not in consumer_ancestors and isinstance(node, BlockNode):
            if any(buffer.name == source.tensor and buffer.location != "shared_hbm" for buffer in node.alloc_buffers):
                return False
    crossed = [nid for nid in set(copy_ancestors) ^ consumer_ancestors if isinstance(ir.tree.data(nid), ForNode)]
    shards = configured_program_shards(ir)
    for nid in ir.dependency.touches_by_tensor.get(source.tensor, ()):
        position = positions[nid]
        writes = tuple(region for region in ir.dependency.info(nid).write_regions if region.tensor == source.tensor)
        if writes and copy_position < position <= consumer_position:
            return False
        ancestors = set(ir.tree.ancestors(nid))
        for loop in crossed:
            iterations = ir.tree.loop(loop).extent // shards.get(loop, 1)
            if (
                loop in ancestors
                and iterations > 1
                and any(
                    _future_write_overlaps(ir, loop, (copy_leaf_nid, source), (nid, write), iterations)
                    for write in writes
                )
            ):
                return False
    return True


def independent_loop_accesses(ir: KernelIR, loop_nid: int, leaf_nid: int) -> bool:
    """Prove that separating one leaf preserves cross-iteration memory dependencies."""
    cache = _LOOP_ACCESS_INDEPENDENCE.setdefault(ir.dependency, {})
    key = (loop_nid, leaf_nid)
    if key in cache:
        return cache[key]
    info = ir.dependency.info(leaf_nid)
    iterations = ir.tree.loop(loop_nid).extent
    for regions, writes in ((info.read_regions, False), (info.write_regions, True)):
        for region in regions:
            for other in ir.dependency.touches_by_tensor.get(region.tensor, ()):
                if other == leaf_nid or loop_nid not in ir.tree.ancestors(other):
                    continue
                other_info = ir.dependency.info(other)
                conflicts = (
                    (*other_info.read_regions, *other_info.write_regions) if writes else other_info.write_regions
                )
                for conflict in conflicts:
                    if conflict.tensor != region.tensor:
                        continue
                    endpoints = ((leaf_nid, region), (other, conflict))
                    if _future_write_overlaps(ir, loop_nid, *endpoints, iterations) or _future_write_overlaps(
                        ir, loop_nid, endpoints[1], endpoints[0], iterations
                    ):
                        cache[key] = False
                        return False
    cache[key] = True
    return True


def _owned_leaf(ir: KernelIR, block_nid: int) -> int | None:
    """Return the sole ISA leaf directly owned by one block."""
    leaves = [
        nid
        for nid in ir.tree.preorder(block_nid)
        if isinstance(ir.tree.data(nid), ISANode) and owning_block(ir, nid) == block_nid
    ]
    return leaves[0] if len(leaves) == 1 else None


def _iteration_region(
    ir: KernelIR, endpoint: tuple[int, BufferRegion], shared_loop: int, future: bool
) -> tuple[BufferRegion, dict[str, int]]:
    """Give endpoint-local loop variables distinct names below a shared iteration."""
    leaf_nid, region = endpoint
    substitutions: dict[str, Expr] = {}
    extents: dict[str, int] = {}
    nested = False
    for nid in ir.tree.ancestors(leaf_nid):
        node = ir.tree.data(nid)
        if not isinstance(node, ForNode):
            continue
        prefix = ("write" if future else "read") if nested else "shared"
        name = f"_copy_{prefix}_{nid}"
        value: Expr = Var(name=name)
        extents[name] = node.extent
        if nid == shared_loop:
            nested = True
            if future:
                value = Add(left=value, right=Add(left=Const(value=1), right=Var(name="_copy_distance")))
        substitutions[node.loop_var] = value
    ranges = tuple(
        (substitute(lower, substitutions), substitute(width, substitutions)) for lower, width in region.ranges
    )
    return replace(region, ranges=ranges), extents


def _future_write_overlaps(
    ir: KernelIR, loop_nid: int, read: tuple[int, BufferRegion], write: tuple[int, BufferRegion], iterations: int
) -> bool:
    """Prove whether any later local iteration may overwrite a saved copy source."""
    read_region, read_extents = _iteration_region(ir, read, loop_nid, False)
    write_region, write_extents = _iteration_region(ir, write, loop_nid, True)
    extents = {**read_extents, **write_extents, "_copy_distance": iterations - 1}
    buffer = ir.param_buffers.get(read_region.tensor) or ir.dependency.info(read[0]).buffers[read_region.tensor]
    return not regions_disjoint(read_region, write_region, buffer, buffer, extents)


def _projection_analyzer(ir: KernelIR) -> Analyzer:
    """Bind the projection proof's unchanged loop bounds once per analysis."""
    analyzer = Analyzer()
    extents: dict[str, int] = {}
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, ForNode):
            extents[node.loop_var] = max(extents.get(node.loop_var, 0), node.extent)
    for loop_var, extent in extents.items():
        analyzer.bind(loop_var, 0, extent)
    return analyzer


def _lossless_dma_promotion(source: Buffer, copied: Buffer, consumer: ISANode, buffers: dict[str, Buffer]) -> bool:
    """Allow an exact floating-point widening only into FP32 DMA destinations."""
    outputs = [
        region.tensor
        for slot, region in consumer.operand_bindings.items()
        if slot not in consumer.op_cls.INPUT_OPERANDS
    ]
    return (
        consumer.op_cls.NAME == "dma_copy"
        and source.dtype in {"bfloat16", "float16", "float32"}
        and source.physical_dtype() in {"bfloat16", "float16"}
        and copied.dtype == copied.physical_dtype() == "float32"
        and bool(outputs)
        and all(buffers[name].dtype == buffers[name].physical_dtype() == "float32" for name in outputs)
    )


def _project_complete_copy(
    ir: KernelIR,
    block_nid: int,
    leaf_nid: int,
    source: BufferRegion,
    copied: BufferRegion,
    consumed: BufferRegion,
    analyzer: Analyzer,
    dma_promotion: bool,
) -> BufferRegion | None:
    """Forward an exact full-tensor copy across different consumer tile boundaries."""
    source_buffer, copied_buffer = ir.buffer(source.tensor), ir.buffer(copied.tensor)
    if (
        ir.tree.parent(block_nid) != ir.tree.root
        or ir.tree.block(block_nid).annotations
        or "program_ownership" in ir.tree.isa(leaf_nid).kwargs
        or source_buffer.location != "shared_hbm"
        or copied_buffer.location not in {"sbuf", "shared_hbm"}
        or source_buffer.shape != copied_buffer.shape
        or source_buffer.physical_dtype() != copied_buffer.physical_dtype()
        and not dma_promotion
    ):
        return None
    scale = Const(value=copied_buffer.partition_extent() if copied_buffer.location == "sbuf" else 1)
    copied = replace(
        copied, ranges=((Mul(left=copied.ranges[0][0], right=scale), copied.ranges[0][1]), *copied.ranges[1:])
    )
    consumed = replace(
        consumed, ranges=((Mul(left=consumed.ranges[0][0], right=scale), consumed.ranges[0][1]), *consumed.ranges[1:])
    )
    if len(source.ranges) != len(copied.ranges) or any(
        not analyzer.can_prove_equal(left, right)
        for original, target in zip(source.ranges, copied.ranges, strict=True)
        for left, right in zip(original, target, strict=True)
    ):
        return None
    loops: dict[Var, int] = {}
    shards = configured_program_shards(ir)
    children = ir.tree.children(block_nid)
    while len(children) == 1 and isinstance(ir.tree.data(children[0]), ForNode):
        nid = children[0]
        loop = ir.tree.loop(nid)
        if nid in shards or loop.extent < 1:
            return None
        loops[Var(name=loop.loop_var)] = loop.extent
        children = ir.tree.children(nid)
    if children != [leaf_nid]:
        return None
    used: set[Var] = set()
    for (lower, width), extent in zip(copied.ranges, copied_buffer.shape, strict=True):
        if not isinstance(width, Const) or width.value < 1:
            return None
        terms = affine_terms(lower)
        if terms.pop(None, 0) != 0:
            return None
        covered = width.value
        for variable, stride in sorted(terms.items(), key=lambda item: item[1]):
            if not isinstance(variable, Var) or variable not in loops or variable in used or stride != covered:
                return None
            covered *= loops[variable]
            used.add(variable)
        if covered != extent:
            return None
    if used != set(loops):
        return None
    ranges = tuple((Const(value=0), Const(value=extent)) for extent in copied_buffer.shape)
    return _project_source_region(
        analyzer,
        BufferRegion(tensor=source.tensor, ranges=ranges),
        BufferRegion(tensor=copied.tensor, ranges=ranges),
        consumed,
    )


def _project_source_region(
    analyzer: Analyzer, source: BufferRegion, copied: BufferRegion, consumed: BufferRegion
) -> BufferRegion | None:
    """Map one contained copied-buffer subregion back to its value source."""
    if len(source.ranges) != len(copied.ranges) or len(copied.ranges) != len(consumed.ranges):
        return None
    ranges = []
    for (source_lower, source_width), (copied_lower, copied_width), (used_lower, used_width) in zip(
        source.ranges, copied.ranges, consumed.ranges
    ):
        if not analyzer.can_prove_equal(source_width, copied_width):
            return None
        offset = (
            Const(value=0)
            if analyzer.can_prove_equal(used_lower, copied_lower)
            else analyzer.simplify(Sub(left=used_lower, right=copied_lower))
        )
        end = analyzer.simplify(Add(left=offset, right=used_width))
        if not analyzer.can_prove(LE(left=Const(value=0), right=offset)) or not analyzer.can_prove(
            LE(left=end, right=copied_width)
        ):
            return None
        lower = analyzer.simplify(Add(left=source_lower, right=offset))
        ranges.append((lower, used_width))
    return BufferRegion(tensor=source.tensor, ranges=tuple(ranges))


def _location_tile_compatible(consumer: ISANode, operand: str, source: BufferRegion, source_buffer: Buffer) -> bool:
    """Enforce any operation-specific direct-HBM tile limits."""
    if source_buffer.location != "shared_hbm":
        return True
    limits = getattr(consumer.op_cls, "HBM_SOURCE_MAX_TILE_SIZE", {})
    for abstract_axis, maximum in limits.items():
        dimension = consumer.op_cls.operand_dimension(operand, abstract_axis)
        width = source.ranges[dimension][1]
        if not isinstance(width, Const) or width.value > maximum:
            return False
    return True


__all__ = ["CopyPropagation", "CopyPropagationOption"]
