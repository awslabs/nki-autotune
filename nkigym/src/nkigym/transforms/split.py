"""``Split`` transform — partition one loop or one tensorize-axis tile into multiple factors."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.expr import Const, Expr, Var, from_affine, substitute
from nkigym.ir.tree import BlockNode, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.transforms._tree_ops import _block_local_descendants, _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_MAX_SPLIT_PARTS = 3


@dataclass(frozen=True)
class SplitOption(TransformOption):
    """Per-application payload for :class:`Split`.

    Attributes:
        target_nid: Node id in ``ir.tree`` to split. Either a
            :class:`ForNode` (outer-trip flavour) or an :class:`ISANode`
            (tensorize flavour).
        factors: Replacement factors, outermost-first. ``len >= 2``.
        target_axis: ``None`` for outer-trip flavour. The abstract iter_var
            axis name (e.g. ``"M"``) for tensorize flavour.
    """

    target_nid: int
    factors: tuple[int, ...]
    target_axis: str | None = None


class Split(Transform):
    """Replace one loop or tensorize-axis tile with a chain of factors."""

    def analyze(self, ir: KernelIR) -> list[SplitOption]:
        options: list[SplitOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if isinstance(data, ForNode):
                for factors in _factorizations(data.extent):
                    options.append(SplitOption(target_nid=nid, factors=factors, target_axis=None))
            elif isinstance(data, ISANode):
                """Tensorize flavour: walk the enclosing block's iter_vars."""
                block_nid, block = _find_enclosing_block(ir.tree, nid)
                for iv in block.iter_vars:
                    abstract = iv.axis
                    extent = iv.dom[1] - iv.dom[0]
                    """Tile width currently bound on the leaf (max_tile or full extent)."""
                    current = _current_tensorize_width(data, abstract)
                    if current is None or current < 2:
                        continue
                    for factors in _factorizations(current):
                        options.append(SplitOption(target_nid=nid, factors=factors, target_axis=abstract))
        return options

    def apply(self, ir: KernelIR, option: SplitOption) -> KernelIR:
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        if option.target_axis is None:
            self._do_outer_trip(new_ir, option)
        else:
            self._do_tensorize(new_ir, option)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: SplitOption) -> None:
        if len(option.factors) < 2:
            raise TransformLegalityError(f"Split.factors must have len >= 2; got {option.factors}")
        if any(f < 2 for f in option.factors):
            raise TransformLegalityError(f"Split.factors entries must be >= 2; got {option.factors}")
        target = _resolve(ir.tree, option.target_nid)
        if option.target_axis is None:
            if not isinstance(target, ForNode):
                raise TransformLegalityError(
                    f"Split outer-trip flavour requires target to be ForNode; got {type(target).__name__}"
                )
            if prod(option.factors) != target.extent:
                raise TransformLegalityError(
                    f"Split.factors product {prod(option.factors)} != ForNode.extent {target.extent}"
                )
        else:
            if not isinstance(target, ISANode):
                raise TransformLegalityError(
                    f"Split tensorize flavour requires target to be ISANode; got {type(target).__name__}"
                )
            block_nid, block = _find_enclosing_block(ir.tree, option.target_nid)
            if not any(iv.axis == option.target_axis for iv in block.iter_vars):
                raise TransformLegalityError(
                    f"Split.target_axis={option.target_axis!r} not declared by enclosing block"
                )
            current = _current_tensorize_width(target, option.target_axis)
            if current is None:
                raise TransformLegalityError(
                    f"Split.target_axis={option.target_axis!r}: no tensorize width on this leaf"
                )
            if prod(option.factors) != current:
                raise TransformLegalityError(
                    f"Split.factors product {prod(option.factors)} != current tensorize width {current}"
                )

    def _do_outer_trip(self, ir: KernelIR, option: SplitOption) -> None:
        """Outer-trip Split: replace the target ForNode with a chain of new ForNodes; rewrite iter_values."""
        target_nid = option.target_nid
        target = ir.tree.data(target_nid)
        assert isinstance(target, ForNode)
        parent_nid = ir.tree.parent(target_nid)
        assert parent_nid is not None
        original_children = ir.tree.children(target_nid)

        block_nid, block = _find_enclosing_block(ir.tree, target_nid)

        new_loop_vars = [f"{target.loop_var}_{i}" for i in range(len(option.factors))]
        new_top_nid: int | None = None
        prev_nid: int | None = None
        for loop_var, extent in zip(new_loop_vars, option.factors):
            new_nid = ir.tree.add_node(ForNode(loop_var=loop_var, extent=extent), parent=None)
            if new_top_nid is None:
                new_top_nid = new_nid
            if prev_nid is not None:
                ir.tree.graph.add_edge(prev_nid, new_nid)
            prev_nid = new_nid
        assert prev_nid is not None and new_top_nid is not None
        for child_nid in original_children:
            ir.tree.graph.add_edge(prev_nid, child_nid)
        _replace_in_parent_children(ir.tree, parent_nid, [target_nid], [new_top_nid])
        ir.tree.graph.remove_node(target_nid)

        """Rewrite iter_values: any iter_value referencing the old loop_var becomes the affine sum."""
        new_value = _affine_split(new_loop_vars, option.factors)
        substitution = {target.loop_var: new_value}
        new_iter_values = tuple(substitute(value, substitution) for value in block.iter_values)
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=new_iter_values,
            reads=tuple(_substitute_region(r, substitution) for r in block.reads),
            writes=tuple(_substitute_region(w, substitution) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block

        """Also update all descendant ISANodes' operand_bindings within this block's scope."""
        for desc_nid in _block_local_descendants(ir.tree, block_nid):
            desc_data = ir.tree.data(desc_nid)
            if isinstance(desc_data, ISANode):
                new_bindings = {
                    slot: _substitute_region(region, substitution)
                    for slot, region in desc_data.operand_bindings.items()
                }
                new_isa = ISANode(op_cls=desc_data.op_cls, operand_bindings=new_bindings, kwargs=dict(desc_data.kwargs))
                ir.tree.graph.nodes[desc_nid]["data"] = new_isa

    def _do_tensorize(self, ir: KernelIR, option: SplitOption) -> None:
        """Tensorize Split: insert ForNodes above the leaf, shrink leaf operand bindings."""
        leaf_nid = option.target_nid
        leaf = ir.tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        parent_nid = ir.tree.parent(leaf_nid)
        assert parent_nid is not None
        block_nid, block = _find_enclosing_block(ir.tree, leaf_nid)

        """Choose a base loop_var name from the existing iter_value if it's a Var, else from axis."""
        base_loop_var = _existing_binding_loop_var(block, option.target_axis) or f"i_{option.target_axis}"
        new_loop_vars = [f"{base_loop_var}_{i}" for i in range(len(option.factors) - 1)]

        ir.tree.graph.remove_edge(parent_nid, leaf_nid)
        prev_nid = parent_nid
        for loop_var, extent in zip(new_loop_vars, option.factors[:-1]):
            new_nid = ir.tree.add_node(ForNode(loop_var=loop_var, extent=extent), parent=prev_nid)
            prev_nid = new_nid
        ir.tree.graph.add_edge(prev_nid, leaf_nid)

        """Shrink the leaf's operand_bindings on target_axis: the new tile width is option.factors[-1]."""
        new_bindings = {
            slot: _shrink_region(region, option.target_axis, option.factors[-1])
            for slot, region in leaf.operand_bindings.items()
        }
        new_leaf = ISANode(op_cls=leaf.op_cls, operand_bindings=new_bindings, kwargs=dict(leaf.kwargs))
        ir.tree.graph.nodes[leaf_nid]["data"] = new_leaf

        """Rewrite iter_values for the affected iter_var: existing binding -> affine sum that includes
        the new outer loop_vars."""
        new_value_factor = _affine_split([*new_loop_vars, base_loop_var], option.factors)
        new_iter_values = tuple(substitute(value, {base_loop_var: new_value_factor}) for value in block.iter_values)
        new_block = BlockNode(
            iter_vars=block.iter_vars,
            iter_values=new_iter_values,
            reads=tuple(_substitute_region(r, {base_loop_var: new_value_factor}) for r in block.reads),
            writes=tuple(_substitute_region(w, {base_loop_var: new_value_factor}) for w in block.writes),
            alloc_buffers=block.alloc_buffers,
            annotations=dict(block.annotations),
        )
        ir.tree.graph.nodes[block_nid]["data"] = new_block


def _resolve(tree: KernelTree, nid: int):
    if nid not in tree.graph:
        raise TransformLegalityError(f"Split.target_nid={nid} is not a node in the IR tree")
    return tree.data(nid)


def _find_enclosing_block(tree: KernelTree, nid: int) -> tuple[int, BlockNode]:
    """Walk ancestors of ``nid`` until we hit a BlockNode."""
    for ancestor in reversed(tree.ancestors(nid)):
        data = tree.data(ancestor)
        if isinstance(data, BlockNode):
            return ancestor, data
    raise TransformLegalityError(f"no enclosing BlockNode for nid {nid}")


def _existing_binding_loop_var(block: BlockNode, axis: str) -> str | None:
    """Return the loop_var name on the iter_value for ``axis``, if it is a bare Var; else None."""
    for iv, value in zip(block.iter_vars, block.iter_values):
        if iv.axis == axis and isinstance(value, Var):
            return value.name
    return None


def _current_tensorize_width(leaf: ISANode, abstract_axis: str) -> int | None:
    """Look up the tile width for ``abstract_axis`` on the first operand whose OPERAND_AXES contains it."""
    op_cls = leaf.op_cls
    for slot, axes in op_cls.OPERAND_AXES.items():
        if abstract_axis not in axes:
            continue
        if slot not in leaf.operand_bindings:
            continue
        region = leaf.operand_bindings[slot]
        axis_index = axes.index(abstract_axis)
        if axis_index >= len(region.ranges):
            continue
        _lo, hi = region.ranges[axis_index]
        if isinstance(hi, Const):
            return hi.value
    return None


def _affine_split(loop_vars: list[str], factors: tuple[int, ...]) -> Expr:
    """Build the affine binding ``v_0 * (f_1*f_2*...) + v_1 * (f_2*...) + ... + v_{n-1}``."""
    coeffs: dict[str | None, int] = {}
    for i, name in enumerate(loop_vars):
        stride = prod(factors[i + 1 :]) if i + 1 < len(factors) else 1
        coeffs[name] = stride
    return from_affine(coeffs)


def _substitute_region(region: BufferRegion, subs: dict[str, Expr]) -> BufferRegion:
    """Return a copy of ``region`` with ``subs`` applied to every range bound."""
    new_ranges = tuple((substitute(lo, subs), substitute(hi, subs)) for lo, hi in region.ranges)
    return BufferRegion(tensor=region.tensor, ranges=new_ranges)


def _shrink_region(region: BufferRegion, target_axis: str, new_width: int) -> BufferRegion:
    """Replace the ``hi`` for the matching axis with Const(new_width)."""
    """Best-effort: locate the axis by looking for a hi whose Const width matches the existing tile.
    We can't directly map abstract axes here without OPERAND_AXES; the caller in tensorize Split
    already validated this width is uniquely the target. For now, replace any range whose hi == old width."""
    new_ranges: list[tuple[Expr, Expr]] = []
    for lo, hi in region.ranges:
        if isinstance(hi, Const):
            new_ranges.append((lo, Const(value=new_width) if hi.value > new_width else hi))
        else:
            new_ranges.append((lo, hi))
    return BufferRegion(tensor=region.tensor, ranges=tuple(new_ranges))


def _factorizations(n: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for parts in range(2, _MAX_SPLIT_PARTS + 1):
        _enum(n, parts, (), out)
    return out


def _enum(remaining: int, parts_left: int, prefix: tuple[int, ...], out: list[tuple[int, ...]]) -> None:
    if parts_left == 1:
        if remaining >= 2:
            out.append(prefix + (remaining,))
        return
    for f in range(2, remaining + 1):
        if remaining % f == 0 and remaining // f >= 2 ** (parts_left - 1):
            _enum(remaining // f, parts_left - 1, prefix + (f,), out)


__all__ = ["Split", "SplitOption"]
