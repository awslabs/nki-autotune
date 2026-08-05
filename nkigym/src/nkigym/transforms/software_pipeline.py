"""``SoftwarePipeline`` transform — assign a loop's child blocks to pipeline
stages, deriving per-buffer version counts (Tier B: stage only, identity order).

Faithful port of TVM ``InjectSoftwarePipeline``. ``apply`` derives versions and
writes an annotation; the prologue/skewed-body/epilogue + ``% versions`` rotation
are manifested by the renderer."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode
from nkigym.transforms._access_pattern import tensor_has_access_pattern
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_EXHAUSTIVE_STAGE_CHILD_LIMIT = 8


@dataclass(frozen=True)
class SoftwarePipelineOption(TransformOption):
    """Pipeline ``loop_nid``'s child blocks across stages.

    Attributes:
        loop_nid: the ForNode whose child blocks are staged.
        stages: stage index per child block, in child order. Full assignment
            (one entry per child); non-decreasing along the dependency chain.
        order: emission order per child block. Tier B: identity.
    """

    loop_nid: int
    stages: tuple[int, ...]
    order: tuple[int, ...]


class SoftwarePipeline(Transform[SoftwarePipelineOption]):
    """Stage-driven accumulator multi-buffer (Tier B)."""

    def analyze(self, ir: KernelIR) -> list[SoftwarePipelineOption]:
        """Enumerate bounded non-decreasing stage labelings for pipelineable loops."""
        options: list[SoftwarePipelineOption] = []
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            children = list(ir.tree.children(nid))
            if len(children) < 2 or self._already_pipelined(ir, nid):
                continue
            order = tuple(range(len(children)))
            for stages in self._nondecreasing_labelings(len(children)):
                opt = SoftwarePipelineOption(loop_nid=nid, stages=stages, order=order)
                if self._is_legal(ir, opt, children):
                    options.append(opt)
        return options

    def apply(self, ir: KernelIR, option: SoftwarePipelineOption) -> KernelIR:
        """Re-check legality, deep-copy, derive versions, write annotation."""
        children = list(ir.tree.children(option.loop_nid))
        self._check_legality(ir, option, children)
        new_ir = copy.deepcopy(ir)
        new_children = list(new_ir.tree.children(option.loop_nid))
        versioned_buffers = self._apply_versions(new_ir, option, new_children)
        parent = self._parent_block(new_ir, option.loop_nid)
        new_ir.tree.block(parent).annotations["software_pipeline"] = {
            "loop_nid": option.loop_nid,
            "loop": new_ir.tree.loop(option.loop_nid),
            "children": tuple(new_children),
            "stages": option.stages,
            "order": option.order,
            "versioned_buffers": versioned_buffers,
        }
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _nondecreasing_labelings(self, n: int) -> list[tuple[int, ...]]:
        """Return useful contiguous stage labelings without large-body explosion.

        Small bodies retain every labeling. Larger bodies offer every contiguous
        two-stage and three-stage partition, keeping option count quadratic while
        avoiding pathological version counts.
        """
        out: list[tuple[int, ...]] = []
        if 1 < n <= _EXHAUSTIVE_STAGE_CHILD_LIMIT:
            for advances in itertools.product((0, 1), repeat=n - 1):
                stage = 0
                labeling = [stage]
                for advance in advances:
                    stage += advance
                    labeling.append(stage)
                if stage > 0:
                    out.append(tuple(labeling))
        elif n > _EXHAUSTIVE_STAGE_CHILD_LIMIT:
            for boundary in range(n - 1, 0, -1):
                out.append((0,) * boundary + (1,) * (n - boundary))
            for first, second in itertools.combinations(range(1, n), 2):
                out.append((0,) * first + (1,) * (second - first) + (2,) * (n - second))
        return out

    def _unit_leaves(self, ir: KernelIR, unit_nid: int) -> list[int]:
        """ISA-leaf nids inside a stageable unit (a direct loop child) — works
        for a BlockNode child or a ForNode-nest child (the matmul loop nest)."""
        candidates = [unit_nid, *ir.tree.descendants(unit_nid)]
        return [d for d in candidates if isinstance(ir.tree.data(d), ISANode)]

    def _touched_tensors(self, ir: KernelIR, children: list[int]) -> set[str]:
        """Return every tensor touched by the staged units."""
        touched: set[str] = set()
        for child in children:
            for leaf in self._unit_leaves(ir, child):
                node = ir.tree.data(leaf)
                assert isinstance(node, ISANode)
                touched.update(region.tensor for region in node.operand_bindings.values())
        return touched

    def _is_legal(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> bool:
        """TVM's two graph rules over the dependency DAG, plus order-permutation."""
        result = True
        if len(option.stages) != len(children) or len(option.order) != len(children):
            result = False
        elif (
            not option.stages
            or min(option.stages) != 0
            or max(option.stages) == 0
            or set(option.stages) != set(range(max(option.stages) + 1))
        ):
            result = False
        elif sorted(option.order) != list(range(len(children))):
            result = False
        else:
            stage_of = {children[i]: option.stages[i] for i in range(len(children))}
            order_of = {children[i]: option.order[i] for i in range(len(children))}
            for src_b in children:
                for dst_b in children:
                    if src_b is dst_b:
                        continue
                    dep = any(
                        ir.dependency.must_precede(ls, ld)
                        for ls in self._unit_leaves(ir, src_b)
                        for ld in self._unit_leaves(ir, dst_b)
                    )
                    if not dep:
                        continue
                    if stage_of[src_b] > stage_of[dst_b]:
                        result = False
                    elif stage_of[src_b] == stage_of[dst_b] and order_of[src_b] >= order_of[dst_b]:
                        result = False
            if any(ir.buffer(name).versions > 1 for name in self._touched_tensors(ir, children)):
                result = False
            version_counts = self._version_counts(ir, option, children)
            if not self._versioned_buffer_touches_are_local(ir, children, version_counts):
                result = False
            if any(
                versions > 1 and tensor_has_access_pattern(ir.tree, name) for name, versions in version_counts.items()
            ):
                result = False
            if not self._version_accesses_are_aligned(ir, option, children, version_counts):
                result = False
        return result

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
        return tuple(axis for axis, (lower, _extent) in enumerate(region.ranges) if loop_var in to_affine(lower))

    def _regions_match_axes(self, write_region: BufferRegion, read_region: BufferRegion, axes: tuple[int, ...]) -> bool:
        """Return whether ``read_region`` matches the selected axes of ``write_region``."""
        return len(write_region.ranges) == len(read_region.ranges) and all(
            write_region.ranges[axis] == read_region.ranges[axis] for axis in axes
        )

    def _check_legality(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> None:
        """Raise TransformLegalityError if illegal."""
        if not self._is_legal(ir, option, children):
            raise TransformLegalityError(f"illegal software-pipeline option {option}")

    def _apply_versions(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> tuple[str, ...]:
        """Set and return buffers requiring more than one pipeline version."""
        version_counts = self._version_counts(ir, option, children)
        versioned_buffers = tuple(sorted(name for name, versions in version_counts.items() if versions > 1))
        for name, versions in version_counts.items():
            if versions > 1:
                self._set_versions(ir, name, versions)
        return versioned_buffers

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
        return {name: uses[name] - defs[name] + 1 for name in set(defs) & set(uses)}

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

    def _already_pipelined(self, ir: KernelIR, loop_nid: int) -> bool:
        """True if some block already annotates this loop as pipelined."""
        return any(
            ir.tree.block(nid).annotations.get("software_pipeline", {}).get("loop_nid") == loop_nid
            for nid in ir.tree.blocks()
        )


__all__ = ["SoftwarePipeline", "SoftwarePipelineOption"]
