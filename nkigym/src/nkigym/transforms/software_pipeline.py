"""``SoftwarePipeline`` transform — assign a loop's child blocks to pipeline
stages, deriving per-buffer version counts (Tier B: stage only, identity order).

Faithful port of TVM ``InjectSoftwarePipeline``. ``apply`` derives versions and
writes an annotation; the prologue/skewed-body/epilogue + ``% versions`` rotation
are manifested by the renderer."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption


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


class SoftwarePipeline(Transform):
    """Stage-driven accumulator multi-buffer (Tier B)."""

    def analyze(self, ir: KernelIR) -> list[SoftwarePipelineOption]:
        """Enumerate non-decreasing stage labelings for every pipelineable loop."""
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
        self._apply_versions(new_ir, option, new_children)
        parent = self._parent_block(new_ir, option.loop_nid)
        new_ir.tree.data(parent).annotations["software_pipeline"] = {
            "loop_nid": option.loop_nid,
            "stages": option.stages,
            "order": option.order,
        }
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _nondecreasing_labelings(self, n: int) -> list[tuple[int, ...]]:
        """Contiguous non-decreasing stage labelings of n blocks: stage 0
        present, no stage gap, excluding the all-zero single-stage no-op."""
        out: list[tuple[int, ...]] = []
        for combo in itertools.product(range(n), repeat=n):
            if combo[0] != 0:
                continue
            if any(combo[i + 1] < combo[i] for i in range(n - 1)):
                continue
            if any(combo[i + 1] - combo[i] > 1 for i in range(n - 1)):
                continue
            if max(combo) == 0:
                continue
            out.append(combo)
        return out

    def _unit_leaves(self, ir: KernelIR, unit_nid: int) -> list[int]:
        """ISA-leaf nids inside a stageable unit (a direct loop child) — works
        for a BlockNode child or a ForNode-nest child (the matmul loop nest)."""
        candidates = [unit_nid, *ir.tree.descendants(unit_nid)]
        return [d for d in candidates if isinstance(ir.tree.data(d), ISANode)]

    def _is_legal(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> bool:
        """TVM's two graph rules over the dependency DAG, plus order-permutation."""
        result = True
        if len(option.stages) != len(children) or len(option.order) != len(children):
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
        return result

    def _check_legality(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> None:
        """Raise TransformLegalityError if illegal."""
        if not self._is_legal(ir, option, children):
            raise TransformLegalityError(f"illegal software-pipeline option {option}")

    def _apply_versions(self, ir: KernelIR, option: SoftwarePipelineOption, children: list[int]) -> None:
        """Set Buffer.versions = use_stage - def_stage + 1 for each buffer the
        staged units touch. Per-leaf reads/writes from ir.dependency.info(leaf)
        (the matmul nest has no BlockNode carrying regions)."""
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
        for name in set(defs) & set(uses):
            versions = uses[name] - defs[name] + 1
            if versions > 1:
                self._set_versions(ir, name, versions)

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
            ir.tree.data(nid).annotations.get("software_pipeline", {}).get("loop_nid") == loop_nid
            for nid in ir.tree.blocks()
        )


__all__ = ["SoftwarePipeline", "SoftwarePipelineOption"]
