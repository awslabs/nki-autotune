"""Prepare one rotating buffer or assign one explicit pipeline stage boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast
from weakref import WeakKeyDictionary

from nkigym.ir import KernelIR, to_affine
from nkigym.ir.arith.expr import Add, Const, Expr, Var, expr_variables, substitute
from nkigym.ir.buffer_placement import layout_satisfies_output_alignment
from nkigym.ir.dependency import _tensor_carried_across
from nkigym.ir.dependency_rebind import rebind_unchanged_dependency
from nkigym.ir.interval import regions_disjoint
from nkigym.ir.program_sharding import configured_program_shards
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.sendrecv import NKISendRecv
from nkigym.transforms.base import (
    Transform,
    TransformLegalityError,
    TransformOption,
    copy_for_rewrite,
    invalidate_software_pipeline_overlap,
)
from nkigym.transforms.buffer_region_normalization import access_patterns_fit_buffer

_UNIT_LEAVES: WeakKeyDictionary[KernelTree, dict[int, tuple[int, ...]]] = WeakKeyDictionary()
_BUFFER_OWNERS: WeakKeyDictionary[KernelTree, dict[str, int]] = WeakKeyDictionary()


@dataclass(frozen=True)
class SoftwarePipelineOption(TransformOption):
    """Change one loop's buffer preparation or pipeline schedule.

    Attributes:
        loop_nid: the ForNode whose child blocks are staged.
        stages: stage index per child block, in child order. Full assignment
            (one entry per child) with one stage boundary. An all-zero
            assignment removes scheduling while retaining prepared buffers.
        tensor: one buffer to prepare or restore, with an empty stage assignment.
        versions: physical version count for the selected buffer.
    """

    loop_nid: int
    stages: tuple[int, ...] = ()
    tensor: str | None = None
    versions: int | None = None


class SoftwarePipeline(Transform[SoftwarePipelineOption]):
    """Separate buffer preparation from stage scheduling in intrinsic child order."""

    def analyze(self, ir: KernelIR) -> list[SoftwarePipelineOption]:
        """Enumerate individual buffer changes and ready stage assignments."""
        options: list[SoftwarePipelineOption] = []
        buffers = ir.all_buffers()
        pipelines = self._pipeline_annotations(ir)
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            children = ir.tree.children(nid)
            if len(children) >= 2:
                options.extend(self._loop_options(ir, nid, children, buffers, pipelines))
        return options

    def _loop_options(
        self,
        ir: KernelIR,
        loop_nid: int,
        children: list[int],
        buffers: dict[str, Buffer],
        pipelines: dict[int, tuple[int, dict[str, object]]],
    ) -> list[SoftwarePipelineOption]:
        """Return atomic choices for one structurally unchanged loop."""
        current = pipelines.get(loop_nid)
        parent = self._parent_block(ir, loop_nid)
        if current is None and "software_pipeline" in ir.tree.block(parent).annotations:
            return []
        prepared = () if current is None else cast(tuple[str, ...], current[1]["versioned_buffers"])
        if any(buffers[name].versions > 1 and name not in prepared for name in self._touched_tensors(ir, children)):
            return []
        if current is not None and any(cast(tuple[int, ...], current[1]["stages"])):
            return [SoftwarePipelineOption(loop_nid, (0,) * len(children))]
        options = [SoftwarePipelineOption(loop_nid, tensor=name, versions=1) for name in prepared]
        if ir.tree.loop(loop_nid).extent // configured_program_shards(ir).get(loop_nid, 1) <= 1:
            return options
        dependencies = self._dependency_pairs(ir, children)
        for stages in self._stage_labelings(len(children)):
            schedule = SoftwarePipelineOption(loop_nid, stages)
            if not self._is_legal(ir, schedule, children, buffers, dependencies):
                continue
            required = self._version_counts(ir, schedule, children)
            missing = {name: count for name, count in required.items() if buffers[name].versions < count}
            if missing:
                options.extend(
                    SoftwarePipelineOption(loop_nid, tensor=name, versions=count)
                    for name, count in sorted(missing.items())
                )
            else:
                options.append(schedule)
        return list(dict.fromkeys(options))

    def apply(self, ir: KernelIR, option: SoftwarePipelineOption) -> KernelIR:
        """Change one buffer's versions or one schedule, preserving the other."""
        children = self._selected_children(ir, option)
        self._check_legality(ir, option, children)
        new_ir = copy_for_rewrite(ir)
        current = self._pipeline_annotation(new_ir, option.loop_nid)
        prepared = set() if current is None else set(cast(tuple[str, ...], current[1]["versioned_buffers"]))
        if option.tensor is not None:
            assert option.versions is not None
            self._set_versions(new_ir, option.tensor, option.versions)
            if option.versions > 1:
                prepared.add(option.tensor)
            else:
                prepared.remove(option.tensor)
            stages = (0,) * len(children)
        else:
            stages = option.stages
        parent = self._parent_block(new_ir, option.loop_nid)
        block = new_ir.tree.block(parent)
        annotations = dict(block.annotations)
        if any(stages) or prepared:
            annotations["software_pipeline"] = {
                "loop_nid": option.loop_nid,
                "loop": new_ir.tree.loop(option.loop_nid),
                "children": tuple(children),
                "stages": stages,
                "order": tuple(range(len(children))),
                "versioned_buffers": tuple(sorted(prepared)),
            }
        else:
            del annotations["software_pipeline"]
        new_ir.tree.graph.nodes[parent]["data"] = replace(block, annotations=annotations)
        invalidate_software_pipeline_overlap(new_ir.tree)
        new_ir.dependency = rebind_unchanged_dependency(ir.dependency, new_ir.tree)
        return new_ir

    def _selected_children(self, ir: KernelIR, option: SoftwarePipelineOption) -> list[int]:
        """Validate the selected loop and return its stageable children."""
        if option.loop_nid not in ir.tree.graph:
            raise TransformLegalityError(f"SoftwarePipeline target loop {option.loop_nid} does not exist")
        if not isinstance(ir.tree.data(option.loop_nid), ForNode):
            raise TransformLegalityError(f"SoftwarePipeline target {option.loop_nid} is not a ForNode")
        children = list(ir.tree.children(option.loop_nid))
        if len(children) < 2:
            raise TransformLegalityError(
                f"SoftwarePipeline target loop {option.loop_nid} must have at least two children"
            )
        return children

    def _stage_labelings(self, n: int) -> list[tuple[int, ...]]:
        """Return every assignment containing one contiguous stage boundary."""
        return [(0,) * boundary + (1,) * (n - boundary) for boundary in range(n - 1, 0, -1)]

    def _unit_leaves(self, ir: KernelIR, unit_nid: int) -> tuple[int, ...]:
        """ISA-leaf nids inside a stageable unit (a direct loop child) — works
        for a BlockNode child or a ForNode-nest child (the matmul loop nest)."""
        leaves_by_unit = _UNIT_LEAVES.setdefault(ir.tree, {})
        leaves = leaves_by_unit.get(unit_nid)
        if leaves is None:
            candidates = (unit_nid, *ir.tree.descendants(unit_nid))
            leaves = tuple(candidate for candidate in candidates if isinstance(ir.tree.data(candidate), ISANode))
            leaves_by_unit[unit_nid] = leaves
        return leaves

    def _touched_tensors(self, ir: KernelIR, children: list[int]) -> set[str]:
        """Return every tensor touched by the staged units."""
        touched: set[str] = set()
        for child in children:
            for leaf in self._unit_leaves(ir, child):
                node = ir.tree.data(leaf)
                assert isinstance(node, ISANode)
                touched.update(region.tensor for region in node.operand_bindings.values())
        return touched

    def _dependency_pairs(self, ir: KernelIR, children: list[int]) -> tuple[tuple[int, int], ...]:
        """Return ordered child-index pairs connected by a dependency path."""
        leaves = tuple(self._unit_leaves(ir, child) for child in children)
        return tuple(
            (source, target)
            for source, source_leaves in enumerate(leaves)
            for target, target_leaves in enumerate(leaves)
            if source != target
            and any(ir.dependency.must_precede(left, right) for left in source_leaves for right in target_leaves)
        )

    def _is_legal(
        self,
        ir: KernelIR,
        option: SoftwarePipelineOption,
        children: list[int],
        buffers: dict[str, Buffer],
        dependencies: tuple[tuple[int, int], ...],
    ) -> bool:
        """Check TVM's two graph rules in intrinsic source order."""
        result = True
        if len(option.stages) != len(children):
            result = False
        elif (
            not option.stages
            or min(option.stages) != 0
            or max(option.stages) < 1
            or set(option.stages) != set(range(max(option.stages) + 1))
        ):
            result = False
        else:
            loop = ir.tree.loop(option.loop_nid)
            programs = configured_program_shards(ir).get(option.loop_nid, 1)
            if any(
                ir.tree.isa(leaf).op_cls is NKISendRecv for child in children for leaf in self._unit_leaves(ir, child)
            ):
                result = False
            elif loop.extent // programs <= max(option.stages):
                result = False
            if any(
                option.stages[source] > option.stages[target]
                or option.stages[source] == option.stages[target]
                and source >= target
                for source, target in dependencies
            ):
                result = False
            if self._has_cross_iteration_read_before_write_hazard(ir, option, children):
                result = False
            version_counts = self._version_counts(ir, option, children)
            version_counts = {name: max(count, buffers[name].versions) for name, count in version_counts.items()}
            owners = self._buffer_owners(ir)
            enclosing = ir.tree.ancestors(option.loop_nid)
            if any(count > 1 and owners[name] not in enclosing for name, count in version_counts.items()):
                result = False
            if any(
                versions > 1 and _tensor_carried_across(ir.dependency, option.loop_nid, name)
                for name, versions in version_counts.items()
            ):
                result = False
            if self._versions_split_rmw(ir, children, version_counts):
                result = False
            if not self._versioned_buffer_touches_are_local(ir, children, version_counts):
                result = False
            if any(
                versions > 1
                and not access_patterns_fit_buffer(
                    ir.tree, name, replace(buffers[name], versions=versions), prior=buffers[name]
                )
                for name, versions in version_counts.items()
            ):
                result = False
            if any(
                versions > 1
                and not layout_satisfies_output_alignment(ir.tree, replace(buffers[name], versions=versions))
                for name, versions in version_counts.items()
            ):
                result = False
            if not self._version_accesses_are_aligned(ir, option, children, version_counts):
                result = False
        return result

    def _buffer_owners(self, ir: KernelIR) -> dict[str, int]:
        """Cache allocation scopes shared by every stage-boundary candidate."""
        owners = _BUFFER_OWNERS.get(ir.tree)
        if owners is None:
            owners = {buffer.name: nid for nid in ir.tree.blocks() for buffer in ir.tree.block(nid).alloc_buffers}
            _BUFFER_OWNERS[ir.tree] = owners
        return owners

    def _versions_split_rmw(self, ir: KernelIR, children: list[int], version_counts: dict[str, int]) -> bool:
        """Return whether version rotation would split one hardware accumulator."""
        result = False
        for child in children:
            for leaf in self._unit_leaves(ir, child):
                node = ir.tree.isa(leaf)
                for slot in node.op_cls.rmw_operands(node.kwargs):
                    region = node.operand_bindings.get(slot)
                    if region is not None and version_counts.get(region.tensor, 1) > 1:
                        result = True
        return result

    def _has_cross_iteration_read_before_write_hazard(
        self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]
    ) -> bool:
        """Return whether staging violates a loop-carried read-before-write."""
        loop = ir.tree.loop(option.loop_nid)
        stage_of = {children[index]: option.stages[index] for index in range(len(children))}
        reads: dict[str, list[tuple[int, int, int, BufferRegion]]] = {}
        writes: dict[str, list[tuple[int, int, int, BufferRegion]]] = {}
        for child_index, child in enumerate(children):
            stage = stage_of[child]
            for leaf in self._unit_leaves(ir, child):
                info = ir.dependency.info(leaf)
                for region in info.read_regions:
                    reads.setdefault(region.tensor, []).append((child_index, stage, leaf, region))
                for region in info.write_regions:
                    writes.setdefault(region.tensor, []).append((child_index, stage, leaf, region))

        return any(
            read_index < write_index
            and read_stage < write_stage
            and any(
                self._regions_overlap_future_iteration(
                    ir, loop, distance, read_leaf, read_region, write_leaf, write_region
                )
                for distance in range(1, min(write_stage - read_stage, loop.extent - 1) + 1)
            )
            for name, tensor_reads in reads.items()
            for read_index, read_stage, read_leaf, read_region in tensor_reads
            for write_index, write_stage, write_leaf, write_region in writes.get(name, ())
        )

    def _regions_overlap_future_iteration(
        self,
        ir: KernelIR,
        loop: ForNode,
        distance: int,
        read_leaf: int,
        read_region: BufferRegion,
        write_leaf: int,
        write_region: BufferRegion,
    ) -> bool:
        """Return whether iteration ``i`` writes what iteration ``i + distance`` reads."""
        future_iteration = Add(left=Var(name=loop.loop_var), right=Const(value=distance))
        substitutions: dict[str, Expr] = {loop.loop_var: future_iteration}
        shifted_read = replace(
            read_region,
            ranges=tuple(
                (substitute(lower, substitutions), substitute(width, substitutions))
                for lower, width in read_region.ranges
            ),
        )
        extents = {
            **ir.dependency.info(write_leaf).extents,
            **ir.dependency.info(read_leaf).extents,
            loop.loop_var: loop.extent - distance,
        }
        buffer = ir.buffer(read_region.tensor)
        return not regions_disjoint(write_region, shifted_read, buffer, buffer, extents)

    def _versioned_buffer_touches_are_local(
        self, ir: KernelIR, children: list[int], version_counts: dict[str, int]
    ) -> bool:
        """Return whether every versioned intermediate is private to the pipeline."""
        pipeline_leaves = {leaf for child in children for leaf in self._unit_leaves(ir, child)}
        return all(
            versions <= 1 or set(ir.dependency.touches_by_tensor.get(name, ())).issubset(pipeline_leaves)
            for name, versions in version_counts.items()
        )

    def _version_accesses_are_aligned(
        self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int], version_counts: dict[str, int]
    ) -> bool:
        """Return whether each later read stays within its pipeline-selected write."""
        loop = ir.tree.data(option.loop_nid)
        assert isinstance(loop, ForNode)
        stage_of = {children[index]: option.stages[index] for index in range(len(children))}
        reads: dict[str, list[tuple[int, BufferRegion]]] = {}
        writes: dict[str, list[tuple[int, BufferRegion]]] = {}
        for child in children:
            stage = stage_of[child]
            for leaf in self._unit_leaves(ir, child):
                info = ir.dependency.info(leaf)
                for region in info.read_regions:
                    reads.setdefault(region.tensor, []).append((stage, region))
                for region in info.write_regions:
                    writes.setdefault(region.tensor, []).append((stage, region))

        result = True
        for name, versions in version_counts.items():
            if versions <= 1:
                continue
            tensor_writes = writes.get(name, [])
            dependent_writes = [
                (stage, region, self._region_axes_using(region, loop.loop_var))
                for stage, region in tensor_writes
                if self._region_axes_using(region, loop.loop_var)
            ]
            for read_stage, read_region in reads.get(name, []):
                if any(stage <= read_stage and write_region == read_region for stage, write_region in tensor_writes):
                    continue
                prior_dependent = [
                    (write_region, axes)
                    for write_stage, write_region, axes in dependent_writes
                    if write_stage < read_stage
                ]
                if prior_dependent and not any(
                    self._regions_match_axes(write_region, read_region, axes) for write_region, axes in prior_dependent
                ):
                    result = False
        return result

    def _region_axes_using(self, region: BufferRegion, loop_var: str) -> tuple[int, ...]:
        """Return region axes whose lower bound references ``loop_var``."""
        return tuple(axis for axis, (lower, _extent) in enumerate(region.ranges) if loop_var in expr_variables(lower))

    def _regions_match_axes(self, write_region: BufferRegion, read_region: BufferRegion, axes: tuple[int, ...]) -> bool:
        """Return whether ``read_region`` matches the selected axes of ``write_region``."""
        return len(write_region.ranges) == len(read_region.ranges) and all(
            write_region.ranges[axis] == read_region.ranges[axis] for axis in axes
        )

    def _check_legality(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> None:
        """Recheck one explicit buffer or schedule choice without combining them."""
        options = self._loop_options(ir, option.loop_nid, children, ir.all_buffers(), self._pipeline_annotations(ir))
        if option not in options or option.tensor is not None and type(option.versions) is not int:
            raise TransformLegalityError(f"illegal software-pipeline option {option}")

    def _version_counts(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> dict[str, int]:
        """Return the pipeline version count required for each defined-and-used buffer."""
        stage_of = {children[i]: option.stages[i] for i in range(len(children))}
        defs: dict[str, int] = {}
        uses: dict[str, int] = {}
        for unit_nid in children:
            st = stage_of[unit_nid]
            reads: set[str] = set()
            writes: set[str] = set()
            for leaf in self._unit_leaves(ir, unit_nid):
                info = ir.dependency.info(leaf)
                reads |= set(info.reads)
                writes |= set(info.writes)
            for name in writes:
                defs[name] = min(defs.get(name, st), st)
            for name in reads:
                uses[name] = max(uses.get(name, st), st)
        buffers = ir.all_buffers()
        return {
            name: uses[name] - defs[name] + 1
            for name in set(defs) & set(uses)
            if buffers[name].location in {"sbuf", "psum"}
        }

    def _set_versions(self, ir: KernelIR, name: str, versions: int) -> None:
        """Replace the owning block's alloc entry for ``name`` with a versions-updated copy."""
        for nid in ir.tree.blocks():
            block = ir.tree.data(nid)
            assert isinstance(block, BlockNode)
            new_allocs = tuple(replace(b, versions=versions) if b.name == name else b for b in block.alloc_buffers)
            if new_allocs != block.alloc_buffers:
                ir.tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=new_allocs)

    def _parent_block(self, ir: KernelIR, loop_nid: int) -> int:
        """Nearest enclosing BlockNode of the loop. ``ancestors`` is root-first,
        so the nearest block is the LAST BlockNode match."""
        result = ir.tree.root
        for anc in ir.tree.ancestors(loop_nid):
            if isinstance(ir.tree.data(anc), BlockNode):
                result = anc
        return result

    def _pipeline_annotation(self, ir: KernelIR, loop_nid: int) -> tuple[int, dict[str, object]] | None:
        """Return the block and active pipeline annotation for one loop."""
        return self._pipeline_annotations(ir).get(loop_nid)

    def _pipeline_annotations(self, ir: KernelIR) -> dict[int, tuple[int, dict[str, object]]]:
        """Return active pipeline annotations indexed by loop node."""
        result: dict[int, tuple[int, dict[str, object]]] = {}
        for nid in ir.tree.blocks():
            annotation = ir.tree.block(nid).annotations.get("software_pipeline")
            if annotation is not None:
                result[cast(int, annotation["loop_nid"])] = (nid, annotation)
        return result


__all__ = ["SoftwarePipeline", "SoftwarePipelineOption"]
