"""``RFactor`` transform — one-stage → two-stage accumulation (spec 2026-06-12 §2).

Factors a reduction loop into ``ko``/``ki`` and restructures the one-stage
accumulation (init once, reduce, drain once) into the FUSED two-stage form:
``ki`` produces a per-``ko`` partial in PSUM, and ``ko`` sums the partials into an
SBUF accumulator via a ``tensor_tensor`` fold. Loops are NOT reordered.

This is the fused single-accumulator form, NOT TVM's multi-slot terminal: the
PSUM accumulator is per-output-tile (it is re-zeroed every ``ko``, never grown by
``factor`` and never carries a ``ko`` slot), so no ``ko``-stride term ever rides
its M (partition-tile) axis — a later ``Split(M)`` cannot corrupt it. See
``docs/superpowers/specs/2026-06-12-same-prefix-computeat-and-two-stage-rfactor-design.md``.

This transform covers the two-dimensional ``"rmw"`` recipe used by
``NKIMatmul``. Fused pointwise reductions use the ``"slot"`` recipe as part of
the atomic tensorize ``Split`` action, so no overwriting split intermediate is
ever exposed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from nkigym.ir import KernelIR
from nkigym.ir.arith.expr import Const, Expr, NonAffineError, Var, substitute, to_affine
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import PARTITION_DIM, BlockNode, Buffer, BufferRegion, ForNode, ISANode, IterVar, KernelTree
from nkigym.ops.base import AxisRole, NKIOp
from nkigym.ops.memset import NKIMemset
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms._canonical_rewrite import single_leaf
from nkigym.transforms._tree_ops import _replace_in_parent_children, invalidate_stale_software_pipelines
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_RMW_STAGING_BUFFER = "sbuf_rfactor"
"""The ``"rmw"`` recipe's transient SBUF staging buffer (spec §2.3). The per-``ko``
PSUM partial is copied here before the ``tensor_tensor`` fold, because
``tensor_tensor`` cannot read a PSUM operand."""

_RMW_COMBINERS = frozenset({"add", "multiply"})
"""Associative, commutative reducers supported by the generated tensor_tensor fold."""


@dataclass(frozen=True)
class RFactorOption(TransformOption):
    """Factor the reduction loop ``target_loop_nid``.

    Attributes:
        target_loop_nid: the ForNode (a reduction/ACCUMULATION loop) to factor.
        factor_axis: retained for API parity with TVM ``rfactor(loop, factor_axis)``;
            the fused form keeps a per-output-tile accumulator (no factor slot), so
            only ``0`` is supported.
    """

    target_loop_nid: int
    factor_axis: int = 0


class RFactor(Transform[RFactorOption]):
    """One-stage → two-stage accumulation: per-``ko`` PSUM partial + SBUF fold."""

    def analyze(self, ir: KernelIR) -> list[RFactorOption]:
        """Enumerate every fully legal ACCUMULATION loop of an rfactorable op."""
        options: list[RFactorOption] = []
        for nid in ir.tree.preorder():
            if not isinstance(ir.tree.data(nid), ForNode):
                continue
            if self._rfactorable(ir, nid):
                options.append(RFactorOption(target_loop_nid=nid, factor_axis=0))
        return options

    def apply(self, ir: KernelIR, option: RFactorOption) -> KernelIR:
        """Re-check legality, deep-copy, emit the two-stage accumulation, return."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        self._emit_rmw(new_ir, option)
        return new_ir

    def _rfactorable(self, ir: KernelIR, loop_nid: int) -> bool:
        """True iff ``loop_nid`` is the outer loop of a split ACCUMULATION axis
        on an op with RFACTOR_RECIPE='rmw', a removable drain, and sufficient
        output."""
        leaf = self._owning_matmul_leaf(ir, loop_nid)
        result = False
        if leaf is not None:
            op_cls = ir.tree.isa(leaf).op_cls
            block_nid = self._enclosing_block_nid(ir.tree, leaf)
            block = ir.tree.block(block_nid)
            axis = self._loop_axis(ir, loop_nid, block)
            axis_loops: list[int] = []
            if axis is not None:
                binding_vars = self._axis_binding_loopvars(block, axis)
                axis_loops = [
                    nid
                    for nid in ir.tree.ancestors(leaf)
                    if isinstance((node := ir.tree.data(nid)), ForNode)
                    and node.loop_var in binding_vars
                    and block_nid in ir.tree.ancestors(nid)
                ]
            if (
                self._supports_rmw_op(op_cls)
                and axis is not None
                and self._role_of(block, axis) == AxisRole.ACCUMULATION
                and len(axis_loops) == 2
                and axis_loops[0] == loop_nid
                and self._init_block_is_retargetable(ir, loop_nid, leaf)
                and self._drain_block_is_removable(ir, loop_nid, leaf)
                and self._gadget_region_fits_output(ir, loop_nid, leaf)
            ):
                result = True
        return result

    def _supports_rmw_op(self, op_cls: type[NKIOp]) -> bool:
        """Whether ``op_cls`` satisfies the currently implemented 2D PSUM recipe."""
        reduction_axes = [axis for axis, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION]
        output_axes = op_cls.OPERAND_AXES.get("dst", ())
        reducer = op_cls.REDUCE_COMBINATOR
        return (
            op_cls.RFACTOR_RECIPE == "rmw"
            and reducer is not None
            and reducer.combiner in _RMW_COMBINERS
            and op_cls.RMW_OPERANDS == frozenset({"dst"})
            and op_cls.OUTPUT_LOCATION == "psum"
            and len(reduction_axes) == 1
            and len(output_axes) == 2
            and reduction_axes[0] not in output_axes
        )

    def _init_block_is_retargetable(self, ir: KernelIR, loop_nid: int, matmul_leaf: int) -> bool:
        """Whether one canonical identity memset initializes PSUM outside ``ko``."""
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        reducer = matmul.op_cls.REDUCE_COMBINATOR
        assert reducer is not None
        inits = [
            nid
            for nid in ir.tree.preorder()
            if isinstance((node := ir.tree.data(nid)), ISANode)
            and node.op_cls.NAME == "memset"
            and node.operand_bindings["dst"].tensor == psum_name
        ]
        result = False
        if len(inits) == 1:
            init_nid = inits[0]
            init_block = self._enclosing_block_nid(ir.tree, init_nid)
            preorder = list(ir.tree.preorder())
            init = ir.tree.isa(init_nid)
            result = (
                single_leaf(ir.tree, init_block) == init_nid
                and loop_nid not in ir.tree.ancestors(init_nid)
                and preorder.index(init_nid) < preorder.index(matmul_leaf)
                and init.kwargs.get("value") == float(reducer.identity)
            )
        return result

    def _drain_block_is_removable(self, ir: KernelIR, loop_nid: int, matmul_leaf: int) -> bool:
        """Whether the sole drain is an outside-``ko`` identity copy in its own block."""
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        drains: list[int] = []
        for nid in ir.tree.preorder():
            node = ir.tree.data(nid)
            if (
                isinstance(node, ISANode)
                and node.op_cls.NAME == "tensor_copy"
                and node.operand_bindings["src"].tensor == psum_name
            ):
                drains.append(nid)
        result = False
        if len(drains) == 1:
            drain_nid = drains[0]
            drain_block = self._enclosing_block_nid(ir.tree, drain_nid)
            leaves = [nid for nid in ir.tree.preorder(drain_block) if isinstance(ir.tree.data(nid), ISANode)]
            drain = ir.tree.isa(drain_nid)
            out_name = drain.operand_bindings["dst"].tensor
            preorder = list(ir.tree.preorder())
            result = (
                leaves == drains
                and loop_nid not in ir.tree.ancestors(drain_nid)
                and preorder.index(matmul_leaf) < preorder.index(drain_nid)
                and drain.operand_bindings["src"].ranges == drain.operand_bindings["dst"].ranges
                and ir.buffer(out_name).location == "sbuf"
            )
        return result

    def _gadget_region_fits_output(self, ir: KernelIR, loop_nid: int, matmul_leaf: int) -> bool:
        """Return whether a contiguous generated region fits PSUM and SBUF.

        RFactor derives its partition offset and absorbed free-axis span from the
        matmul accumulator, then uses that region for PSUM, output, and staging.
        Check the full affine region over every enclosing loop against both
        existing buffers. List layout and pipeline versions do not increase
        ``shape`` capacity.
        """
        ki_nid = self._ki_loop_nid(ir, loop_nid)
        matmul = ir.tree.data(matmul_leaf)
        assert isinstance(matmul, ISANode)
        psum_name = matmul.operand_bindings["dst"].tensor
        out_name = self._drain_out_tensor(ir.tree, psum_name)
        dst_region = matmul.operand_bindings["dst"]
        free_footprint = self._free_footprint(ir, ki_nid, matmul_leaf)
        loop_extents = {
            node.loop_var: node.extent
            for nid in ir.tree.ancestors(matmul_leaf)
            if isinstance((node := ir.tree.data(nid)), ForNode)
        }
        result = False
        if free_footprint is not None:
            free_lo, free_extent = free_footprint
            region = self._partition_region(out_name, dst_region.ranges[0][0], free_lo, free_extent)
            result = all(
                self._region_axis_fits(lo, width, axis, buf, loop_extents)
                for buf in (ir.buffer(psum_name), ir.buffer(out_name))
                for axis, (lo, width) in enumerate(region.ranges)
            )
        return result

    def _region_axis_fits(self, lo: Expr, width: Expr, axis: int, buf: Buffer, loop_extents: dict[str, int]) -> bool:
        """Return whether one affine gadget axis stays within ``buf.shape``."""
        result = False
        coeffs: dict[str | None, int] | None = None
        width_value = width.value if isinstance(width, Const) else None
        if width_value is not None:
            try:
                coeffs = to_affine(lo)
            except (NonAffineError, TypeError):
                coeffs = None
        if coeffs is not None and width_value is not None:
            lower = coeffs.get(None, 0)
            upper = lower
            bounded = axis < len(buf.shape)
            for var, coeff in coeffs.items():
                if var is None:
                    continue
                extent = loop_extents.get(var)
                if extent is None:
                    bounded = False
                    break
                span = coeff * (extent - 1)
                if span < 0:
                    lower += span
                else:
                    upper += span
            if axis == 0 and buf.location in ("sbuf", "psum") and width_value == PARTITION_DIM:
                lower *= PARTITION_DIM
                upper *= PARTITION_DIM
            result = bounded and lower >= 0 and upper + width_value <= buf.shape[axis]
        return result

    def _check_legality(self, ir: KernelIR, option: RFactorOption) -> None:
        """Raise TransformLegalityError if the option is not a valid rmw rfactor."""
        nid = option.target_loop_nid
        if nid not in ir.tree.graph or not isinstance(ir.tree.data(nid), ForNode):
            raise TransformLegalityError(f"RFactor target {nid} is not a ForNode in the tree")
        if option.factor_axis != 0:
            raise TransformLegalityError(
                f"RFactor factor_axis must be 0 for the fused rmw recipe; got {option.factor_axis}"
            )
        if not self._rfactorable(ir, nid):
            raise TransformLegalityError(
                f"RFactor target loop {nid} is not a legal reduction: an rmw recipe must be "
                f"the outermost of exactly two loops binding an ACCUMULATION axis, "
                f"have canonical outside-loop init and identity-mapped drain blocks, "
                f"use a supported combiner, and fit a contiguous gadget footprint "
                f"within PSUM/output capacity"
            )

    def _emit_rmw(self, ir: KernelIR, option: RFactorOption) -> None:
        """Restructure one-stage → fused two-stage accumulation on ``ir`` (spec §2.1).

        The factored loop ``ko`` (``option.target_loop_nid``) is the matmul's
        outermost reduction loop; ``ki`` is its INNERMOST reduction loop, whose
        subtree ends in the matmul run-op. The emission keeps the PSUM accumulator
        per-output-tile and folds the ``factor`` partials into the SBUF output
        ``out_sbuf``, anchoring the gadgets to ``ki`` (spec 2026-07-07 §B):

        - ``init_two_stage_0``: the canonical flat ``memset`` that zeroed ``psum``
          is RETARGETED to zero ``out_sbuf`` (the second-stage accumulator), staying
          a root sibling before ``ko``.
        - ``init_two_stage_1``: a NEW ``memset`` zeroing ``psum`` (bare, NO slot),
          spliced as ``ki``'s PRECEDING sibling.
        - run-op: the matmul block's K iter_var role flips ACCUMULATION → PARALLEL
          (each ``ko`` is an independent partial; ``ki`` HW-accumulates in PSUM); its
          ``dst`` keeps no factor slot and its existing full-frame region.
        - ``drain_two_stage_0``: a ``tensor_copy`` (``psum`` → SBUF ``sbuf_rfactor``)
          then a ``tensor_tensor`` fold (``out_sbuf = combiner(out_sbuf,
          sbuf_rfactor)``), spliced as ``ki``'s FOLLOWING siblings. The fold block
          carries ``ko`` as ACCUMULATION (the closing second-stage reduction on
          ``out_sbuf``).
        - ``drain_two_stage_1``: empty — the result is already in ``out_sbuf``.

        The gadgets are sized to the footprint R — the accumulator region the
        ``ki``-subtree writes over one full ``ki`` execution: partition loops between
        ``ki`` and the matmul are MATERIALIZED (early-packed: the 16-trip ``M`` loop),
        free loops are ABSORBED into the op width. This is structural-only: the
        existing PSUM and generated SBUF staging buffer retain their full-frame
        allocation geometry and region offsets. Explicit downstream
        ``BufferCompaction`` actions materialize their tighter lifetimes.
        """
        tree = ir.tree
        ko_loop = tree.data(option.target_loop_nid)
        assert isinstance(ko_loop, ForNode)
        ko_var = ko_loop.loop_var

        matmul_leaf = self._owning_matmul_leaf(ir, option.target_loop_nid)
        assert matmul_leaf is not None
        matmul_block_nid = self._enclosing_block_nid(tree, matmul_leaf)
        matmul_node = tree.data(matmul_leaf)
        assert isinstance(matmul_node, ISANode)
        op_cls = matmul_node.op_cls
        reducer = op_cls.REDUCE_COMBINATOR
        assert reducer is not None

        psum_name = matmul_node.operand_bindings["dst"].tensor
        out_name = self._drain_out_tensor(tree, psum_name)
        staging_name = self._staging_buffer_name(ir)
        identity = float(reducer.identity)
        combiner = reducer.combiner

        ki_nid = self._ki_loop_nid(ir, option.target_loop_nid)
        footprint = self._footprint(ir, ki_nid, matmul_leaf)
        part_lo = matmul_node.operand_bindings["dst"].ranges[0][0]
        free_footprint = self._free_footprint(ir, ki_nid, matmul_leaf)
        assert free_footprint is not None
        free_lo, free_extent = free_footprint

        self._add_rf_buffer(ir, psum_name, out_name, staging_name)
        self._flip_matmul_k_role(tree, matmul_block_nid)
        self._retarget_init(tree, psum_name, out_name)
        self._remove_flat_block(tree, self._reader_leaf(tree, psum_name, "tensor_copy"))
        self._nest_memset(
            ir, matmul_leaf, ki_nid, psum_name, ko_var, footprint, part_lo, free_lo, free_extent, identity
        )
        copy_block_nid = self._nest_copy(
            ir, matmul_leaf, ki_nid, psum_name, staging_name, ko_var, footprint, part_lo, free_lo, free_extent
        )
        self._nest_combine(
            ir,
            matmul_leaf,
            ki_nid,
            copy_block_nid,
            out_name,
            staging_name,
            ko_var,
            footprint,
            part_lo,
            free_lo,
            free_extent,
            combiner,
        )

        invalidate_stale_software_pipelines(ir)
        ir.dependency = Dependency(tree)

    def _enclosing_block_nid(self, tree: KernelTree, nid: int) -> int:
        """Nearest enclosing BlockNode nid of ``nid`` (deepest ancestor block)."""
        for anc in reversed(tree.ancestors(nid)):
            if isinstance(tree.data(anc), BlockNode):
                return anc
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _ki_loop_nid(self, ir: KernelIR, ko_loop_nid: int) -> int:
        """The INNERMOST ACCUMULATION (K-axis) loop enclosing the matmul, at or below ko.

        The matmul's reduction is driven by its K-axis loops; ki is the deepest of them.
        Found as the last (innermost) matmul-ancestor ForNode whose loop_var binds the
        matmul block's K axis. For early-packed ki sits directly under ko; at the
        fully scheduled endpoint ki is the innermost loop and the matmul is its sole
        body.
        """
        tree = ir.tree
        matmul_leaf = self._owning_matmul_leaf(ir, ko_loop_nid)
        assert matmul_leaf is not None
        block_nid = self._enclosing_block_nid(tree, matmul_leaf)
        block = tree.block(block_nid)
        op_cls = self._op_cls_of_block(tree, block_nid)
        reduction_abstract = next(a for a, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        k_axis = block.axis_map[reduction_abstract]
        k_binding_vars = self._axis_binding_loopvars(block, k_axis)
        k_loops = [
            a
            for a in tree.ancestors(matmul_leaf)
            if isinstance((node := tree.data(a)), ForNode)
            and node.loop_var in k_binding_vars
            and block_nid in tree.ancestors(a)
        ]
        return k_loops[-1]

    def _axis_binding_loopvars(self, block: BlockNode, axis: str) -> set[str]:
        """Loop vars appearing in the iter_value of ``axis`` (the loops that bind it)."""
        value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == axis)
        return {n for n in to_affine(value) if n is not None}

    def _footprint(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> list[tuple[str, int]]:
        """Ordered (loop_var, extent) of the ForNodes STRICTLY between ki and the matmul
        leaf whose loop_var binds the matmul's PARTITION (dst axis-0) dim.

        These are the partition-tile loops the ki-subtree sweeps over one ki execution;
        the gadgets materialize them (early-packed: the 16-trip M loop). Free-axis loops
        between ki and the matmul are absorbed into the op width, so they are NOT returned.
        At the fully scheduled endpoint there are no loops between ki and the matmul,
        so this is empty (R = one tile, loopless gadgets).
        """
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        m_abstract = tree.isa(matmul_leaf).op_cls.OPERAND_AXES["dst"][0]
        m_axis = block.axis_map[m_abstract]
        m_binding_vars = self._axis_binding_loopvars(block, m_axis)
        between = [
            a
            for a in tree.ancestors(matmul_leaf)
            if isinstance(tree.data(a), ForNode) and ki_loop_nid in tree.ancestors(a)
        ]
        return [
            (tree.loop(a).loop_var, tree.loop(a).extent) for a in between if tree.loop(a).loop_var in m_binding_vars
        ]

    def _free_footprint(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> tuple[Expr, int] | None:
        """Return one contiguous free-axis interval swept below ``ki``, if representable."""
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        matmul = tree.isa(matmul_leaf)
        dst_region = matmul.operand_bindings["dst"]
        free_abstract = matmul.op_cls.OPERAND_AXES["dst"][1]
        free_axis = block.axis_map[free_abstract]
        free_binding_vars = self._axis_binding_loopvars(block, free_axis)
        tile_width = dst_region.ranges[1][1]
        absorbed = {
            node.loop_var: node.extent
            for a in tree.ancestors(matmul_leaf)
            if isinstance((node := tree.data(a)), ForNode)
            and ki_loop_nid in tree.ancestors(a)
            and node.loop_var in free_binding_vars
        }
        result: tuple[Expr, int] | None = None
        coeffs: dict[str | None, int] | None = None
        try:
            coeffs = to_affine(dst_region.ranges[1][0])
        except (NonAffineError, TypeError):
            coeffs = None
        if coeffs is not None and isinstance(tile_width, Const):
            strides = sorted((coeffs.get(var, 0), var, extent) for var, extent in absorbed.items())
            width = tile_width.value
            contiguous = True
            for stride, _var, extent in strides:
                if stride != width:
                    contiguous = False
                    break
                width *= extent
            if contiguous:
                free_lo = substitute(dst_region.ranges[1][0], {var: Const(value=0) for var in absorbed})
                result = free_lo, width
        return result

    def _op_cls_of_block(self, tree: KernelTree, block_nid: int) -> type[NKIOp]:
        """Return the op class of the rfactorable (reduction) leaf under ``block_nid``.

        A block may own several ISA leaves once co-location nested a memset / drain
        beside the matmul. ``tree.descendants`` is an unordered set, so select the
        single leaf whose op declares an ``RFACTOR_RECIPE`` (the matmul).
        """
        leaves = [d for d in tree.descendants(block_nid) if isinstance(tree.data(d), ISANode)]
        rfactorable = [n for n in leaves if tree.isa(n).op_cls.RFACTOR_RECIPE is not None]
        if len(rfactorable) != 1:
            raise TransformLegalityError(
                f"block {block_nid} must own exactly one rfactorable leaf; got {len(rfactorable)}"
            )
        return tree.isa(rfactorable[0]).op_cls

    def _drain_out_tensor(self, tree: KernelTree, psum_name: str) -> str:
        """Tensor the drain ``tensor_copy`` writes (reads ``psum_name``, writes SBUF out)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == "tensor_copy":
                if data.operand_bindings["src"].tensor == psum_name:
                    return data.operand_bindings["dst"].tensor
        raise TransformLegalityError(f"no drain tensor_copy reading {psum_name!r}")

    def _staging_buffer_name(self, ir: KernelIR) -> str:
        """Return the first available deterministic staging-buffer name."""
        names = set(ir.all_buffers())
        candidate = _RMW_STAGING_BUFFER
        suffix = 0
        while candidate in names:
            suffix += 1
            candidate = f"{_RMW_STAGING_BUFFER}_{suffix}"
        return candidate

    def _add_rf_buffer(self, ir: KernelIR, psum_name: str, out_name: str, staging_name: str) -> None:
        """Add the SBUF staging buffer ``sbuf_rfactor`` (spec §2.3 transient).

        The structural RFactor atom gives staging the output's existing full-frame
        shape and declares it beside the PSUM. A later per-buffer
        ``BufferCompaction`` action can place, shrink, and normalize it.
        """
        tree = ir.tree
        out_buf = ir.buffer(out_name)
        rf_buf = Buffer(
            name=staging_name,
            shape=out_buf.shape,
            dtype=out_buf.dtype,
            location="sbuf",
            storage_dtype=out_buf.storage_dtype,
        )
        for nid in tree.blocks():
            block = tree.data(nid)
            assert isinstance(block, BlockNode)
            if any(buf.name == psum_name for buf in block.alloc_buffers):
                tree.graph.nodes[nid]["data"] = replace(block, alloc_buffers=(*block.alloc_buffers, rf_buf))
                return
        raise TransformLegalityError(f"no block allocates {psum_name!r}")

    def _flip_matmul_k_role(self, tree: KernelTree, block_nid: int) -> None:
        """Flip the matmul block's K iter_var ACCUMULATION → PARALLEL (regions unchanged).

        Each ``ko`` is now an independent partial, so the K axis (driving both
        ``ko`` and ``ki``) is data-parallel in the run-op block; the inner ``ki``
        accumulate is the HW ``+=`` encapsulated in the single matmul leaf. The
        closing cross-``ko`` reduction re-emerges as ACCUMULATION on the stage-2
        ``tensor_tensor`` block. Legal only because the reducer is assoc + comm.
        """
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        op_cls = self._op_cls_of_block(tree, block_nid)
        reduction_abstract = next(a for a, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        k_axis = block.axis_map[reduction_abstract]
        new_iter_vars = tuple(
            replace(iv, role=AxisRole.PARALLEL) if iv.axis == k_axis else iv for iv in block.iter_vars
        )
        tree.graph.nodes[block_nid]["data"] = replace(block, iter_vars=new_iter_vars)

    def _retarget_init(self, tree: KernelTree, psum_name: str, out_name: str) -> None:
        """Retarget the canonical flat ``memset(psum)`` to ``memset(out_sbuf)`` in place.

        ``init_one_stage`` zeroed the PSUM accumulator; the two-stage form's
        ``init_two_stage_0`` zeros the SBUF accumulator instead (PSUM is re-zeroed
        per-``ko`` by ``init_two_stage_1``). The block keeps its loop nest, iter_vars,
        and position (a root sibling before ``ko``); only the written tensor name is
        retargeted ``psum_name`` → ``out_name`` (the region ranges are preserved).
        """
        leaf_nid = self._writer_leaf(tree, psum_name, "memset")
        block_nid = self._enclosing_block_nid(tree, leaf_nid)
        block = tree.data(block_nid)
        assert isinstance(block, BlockNode)
        new_writes = tuple(self._retarget_region(w, psum_name, out_name) for w in block.writes)
        new_reads = tuple(self._retarget_region(r, psum_name, out_name) for r in block.reads)
        tree.graph.nodes[block_nid]["data"] = replace(block, reads=new_reads, writes=new_writes)
        leaf = tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        new_bindings = {
            slot: self._retarget_region(region, psum_name, out_name) for slot, region in leaf.operand_bindings.items()
        }
        tree.graph.nodes[leaf_nid]["data"] = ISANode(
            op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs)
        )

    def _retarget_region(self, region: BufferRegion, old_tensor: str, new_tensor: str) -> BufferRegion:
        """Rename ``region``'s tensor ``old_tensor`` → ``new_tensor`` (ranges unchanged)."""
        if region.tensor != old_tensor:
            return region
        return BufferRegion(tensor=new_tensor, ranges=region.ranges)

    def _nest_memset(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        psum_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
        identity: float,
    ) -> None:
        """Splice ``init_two_stage_1``: a ``memset(psum)`` as ``ki``'s preceding sibling.

        Zeros the per-``ki`` PSUM partial before the ``ki`` matmul nest. The block
        binds ``ko`` (axis ``d0``, role PARALLEL — matching the flipped matmul ``ko``,
        so this is an ordinary init, not a re-zero of ``ko``'s own reduction) and the
        output partition dim; its dst is bare ``psum[part_lo]`` (NO ``ko`` slot). The
        ``footprint`` partition loops (if any) are materialized by ``_splice_beside_ki``.
        """
        region = self._partition_region(psum_name, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.PARALLEL, reads=(), writes=(region,)
        )
        leaf = ISANode(op_cls=NKIMemset, operand_bindings={"dst": region}, kwargs={"value": identity})
        self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, insert_after=None)

    def _nest_copy(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        psum_name: str,
        staging_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
    ) -> int:
        """Splice the first half of ``drain_two_stage_0``: ``tensor_copy(psum → sbuf_rfactor)``.

        Staged as ``ki``'s following sibling. Moves the per-``ki`` PSUM partial into
        SBUF so the following ``tensor_tensor`` fold can read it (``tensor_tensor``
        cannot read a PSUM operand). The block binds ``ko`` (PARALLEL) + the output
        partition dim; both regions are bare ``[part_lo]``.
        """
        src = self._partition_region(psum_name, part_lo, free_lo, free_extent)
        dst = self._partition_region(staging_name, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.PARALLEL, reads=(src,), writes=(dst,)
        )
        leaf = ISANode(op_cls=NKITensorCopy, operand_bindings={"src": src, "dst": dst}, kwargs={})
        return self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, insert_after=ki_nid)

    def _nest_combine(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ki_nid: int,
        copy_block_nid: int,
        out_name: str,
        staging_name: str,
        ko_var: str,
        footprint: list[tuple[str, int]],
        part_lo: Expr,
        free_lo: Expr,
        free_extent: int,
        combiner: str,
    ) -> None:
        """Splice the second half of ``drain_two_stage_0``: the cross-``ko`` SBUF fold.

        ``tensor_tensor(data1=out_sbuf, data2=sbuf_rfactor, dst=out_sbuf, op=combiner)``
        as ``ki``'s following sibling (after the copy). ``data1`` and ``dst`` alias
        the accumulator (constant address across ``ko``), so the block binds ``ko`` as
        ACCUMULATION — its cross-``ko`` carry on ``out_sbuf`` is the closing
        second-stage reduction. ``combiner`` is the op's ``REDUCE_COMBINATOR.combiner``
        (``"add"`` for matmul).
        """
        out_region = self._partition_region(out_name, part_lo, free_lo, free_extent)
        rf_region = self._partition_region(staging_name, part_lo, free_lo, free_extent)
        block = self._gadget_block(
            ir,
            matmul_leaf,
            ko_var,
            footprint,
            free_extent,
            AxisRole.ACCUMULATION,
            reads=(out_region, rf_region),
            writes=(out_region,),
        )
        leaf = ISANode(
            op_cls=NKITensorTensor,
            operand_bindings={"data1": out_region, "data2": rf_region, "dst": out_region},
            kwargs={"op": combiner},
        )
        self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, insert_after=copy_block_nid)

    def _remove_flat_block(self, tree: KernelTree, leaf_nid: int) -> None:
        """Delete the canonical flat block owning ``leaf_nid`` (block + loop + leaf).

        The flat memset/drain blocks are perfect single-leaf nests under the root: a
        BlockNode, one partition ForNode, one ISA leaf. Detach the block from its
        parent (preserving sibling order) and remove all three nodes.
        """
        block_nid = self._enclosing_block_nid(tree, leaf_nid)
        parent = tree.parent(block_nid)
        assert parent is not None
        remaining = [c for c in tree.children(parent) if c != block_nid]
        _replace_in_parent_children(tree, parent, [block_nid], [])
        assert tree.children(parent) == remaining
        for nid in tree.descendants(block_nid) | {block_nid}:
            tree.graph.remove_node(nid)

    def _partition_region(self, tensor: str, part_lo: Expr, free_lo: Expr, free_extent: int) -> BufferRegion:
        """Canonical ``tensor[part_lo : +128, free_lo : +free_extent]`` region.

        ``part_lo`` is the matmul dst partition (axis-0) offset — a bare loop Var when a
        footprint loop is materialized (early-packed ``i_d1_0``) or a compound affine
        inherited from the enclosing output-tile loops
        (``i_d1_0*4 + i_d1_1`` at the fully scheduled endpoint).
        ``free_lo`` is the matmul dst free (axis-1) offset with the absorbed free-loop
        vars stripped (see :meth:`_free_footprint`) — ``i_d2_0*512`` for the fully
        scheduled ladder endpoint and ``Const(0)`` for early-packed. A subsequent
        ``BufferCompaction`` can rewrite instance-selecting offsets to zero.
        """
        return BufferRegion(
            tensor=tensor, ranges=((part_lo, Const(value=PARTITION_DIM)), (free_lo, Const(value=free_extent)))
        )

    def _gadget_block(
        self,
        ir: KernelIR,
        matmul_leaf: int,
        ko_var: str,
        footprint: list[tuple[str, int]],
        free_extent: int,
        d0_role: AxisRole,
        reads: tuple[BufferRegion, ...],
        writes: tuple[BufferRegion, ...],
    ) -> BlockNode:
        """Build a per-``ki`` gadget :class:`BlockNode` bracketing the ``ki`` loop.

        iter_vars mirror the matmul block: ``d0`` (K, bound to ``ko_var``, role ``d0_role``),
        ``d1`` (the output partition dim, bound to the matmul's own partition iter_value —
        ``i_d1_0`` early-packed, ``i_d1_0*4 + i_d1_1`` at the fully scheduled
        endpoint), and ``d2`` (free, loopless).
        The ``footprint`` partition loops are materialized by the caller as ForNodes; when
        empty (the fully scheduled endpoint) the ``d1`` value is inherited from the
        enclosing output-tile loops.

        The ``d2`` iter_value is left as ``Const(0)`` for all gadgets. This is intentionally
        inert: the free axis is loopless (the gadgets absorb free loops into their op width),
        so no loop resolves to ``d2``; the free position lives on the ``BufferRegion`` ranges
        (``free_lo``), which is what codegen and dependency analysis consume.
        """
        tree = ir.tree
        block = self._enclosing_block(ir, matmul_leaf)
        op_cls = tree.isa(matmul_leaf).op_cls
        reduction_abstract = next(axis for axis, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
        output_axes = op_cls.OPERAND_AXES["dst"]
        k_axis = block.axis_map[reduction_abstract]
        m_axis = block.axis_map[output_axes[0]]
        free_axis = block.axis_map[output_axes[1]]
        k_dom = next(iv.dom for iv in block.iter_vars if iv.axis == k_axis)
        m_value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == m_axis)
        m_dom = next(iv.dom for iv in block.iter_vars if iv.axis == m_axis)
        return BlockNode(
            iter_vars=(
                IterVar(axis=k_axis, dom=k_dom, role=d0_role),
                IterVar(axis=m_axis, dom=m_dom, role=AxisRole.PARALLEL),
                IterVar(axis=free_axis, dom=(0, free_extent), role=AxisRole.PARALLEL),
            ),
            iter_values=(Var(name=ko_var), m_value, Const(value=0)),
            reads=reads,
            writes=writes,
            alloc_buffers=(),
            axis_map={"K": k_axis, "P": m_axis, "F": free_axis},
        )

    def _splice_beside_ki(
        self,
        tree: KernelTree,
        ki_loop_nid: int,
        block: BlockNode,
        footprint: list[tuple[str, int]],
        leaf: ISANode,
        insert_after: int | None,
    ) -> int:
        """Splice ``block`` (with the ``footprint`` partition loops + ``leaf``) as a sibling
        of ``ki_loop_nid`` under ki's parent.

        ``insert_after=None`` puts the block immediately before ``ki``. Otherwise
        the block is inserted immediately after the named sibling. The caller
        chains copy then fold, preserving unrelated siblings after the gadget.
        Each footprint entry becomes a materialized ForNode (outer->inner); an
        empty footprint attaches ``leaf`` directly to ``block``.
        """
        parent = tree.parent(ki_loop_nid)
        assert parent is not None, f"ki loop {ki_loop_nid} has no parent"
        block_nid = tree.add_node(block, parent=parent)
        cursor = block_nid
        for loop_var, extent in footprint:
            cursor = tree.add_node(ForNode(loop_var=loop_var, extent=extent), parent=cursor)
        tree.add_node(leaf, parent=cursor)
        siblings = [c for c in tree.children(parent) if c != block_nid]
        anchor = ki_loop_nid if insert_after is None else insert_after
        anchor_index = siblings.index(anchor)
        insertion_index = anchor_index if insert_after is None else anchor_index + 1
        new_order = [*siblings[:insertion_index], block_nid, *siblings[insertion_index:]]
        for child in tree.children(parent):
            tree.graph.remove_edge(parent, child)
        for child in new_order:
            tree.graph.add_edge(parent, child)
        return block_nid

    def _writer_leaf(self, tree: KernelTree, tensor: str, op_name: str) -> int:
        """The single ISA leaf with NAME ``op_name`` that writes ``tensor`` (dst slot)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == op_name:
                if data.operand_bindings.get("dst") is not None and data.operand_bindings["dst"].tensor == tensor:
                    return nid
        raise TransformLegalityError(f"no {op_name} writing {tensor!r}")

    def _reader_leaf(self, tree: KernelTree, tensor: str, op_name: str) -> int:
        """The single ISA leaf with NAME ``op_name`` that reads ``tensor`` (src slot)."""
        for nid in tree.preorder():
            data = tree.data(nid)
            if isinstance(data, ISANode) and data.op_cls.NAME == op_name:
                if data.operand_bindings.get("src") is not None and data.operand_bindings["src"].tensor == tensor:
                    return nid
        raise TransformLegalityError(f"no {op_name} reading {tensor!r}")

    def _owning_matmul_leaf(self, ir: KernelIR, loop_nid: int) -> int | None:
        """The single ISA leaf under ``loop_nid`` whose op is rfactorable, or None."""
        leaves = [
            d
            for d in ir.tree.descendants(loop_nid)
            if isinstance((node := ir.tree.data(d)), ISANode) and node.op_cls.RFACTOR_RECIPE is not None
        ]
        return leaves[0] if len(leaves) == 1 else None

    def _enclosing_block(self, ir: KernelIR, nid: int) -> BlockNode:
        """Nearest enclosing BlockNode payload of ``nid``."""
        for anc in reversed(ir.tree.ancestors(nid)):
            data = ir.tree.data(anc)
            if isinstance(data, BlockNode):
                return data
        raise TransformLegalityError(f"no enclosing BlockNode for {nid}")

    def _loop_axis(self, ir: KernelIR, loop_nid: int, block: BlockNode) -> str | None:
        """The concrete axis the loop's loop_var binds, via the block's iter_values."""
        loop_var = ir.tree.loop(loop_nid).loop_var
        for iv, value in zip(block.iter_vars, block.iter_values):
            if loop_var in to_affine(value):
                return iv.axis
        return None

    def _role_of(self, block: BlockNode, axis: str) -> AxisRole:
        """Role the block assigns to ``axis`` (default PARALLEL if absent)."""
        for iv in block.iter_vars:
            if iv.axis == axis:
                return iv.role
        return AxisRole.PARALLEL


__all__ = ["RFactor", "RFactorOption"]
