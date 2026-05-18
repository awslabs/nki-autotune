"""``Split`` transform - partition one axis-chain entry into multiple factors."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import prod

from nkigym.ir import KernelIR
from nkigym.ir.dependency import Dependency
from nkigym.ir.tree import ForNode, ISANode, KernelTree
from nkigym.ops.alloc import NKIAlloc
from nkigym.transforms._tree_ops import _replace_in_parent_children
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption

_MAX_SPLIT_PARTS = 3


@dataclass(frozen=True)
class SplitOption(TransformOption):
    """Per-application payload for :class:`Split`.

    Attributes:
        target_nid: Node id in ``ir.tree`` to split.
        factors: Replacement factors, outermost-first. ``len >= 2``,
            each factor ``>= 2``.
        target_axis: ``None`` selects the outer-trip flavor -
            ``target_nid`` is a :class:`ForNode` and ``factors``
            replaces its trip count. Set to an abstract axis name
            (e.g. ``"M"``) for the tensorize flavor - ``target_nid``
            is an :class:`ISANode` and ``factors`` replaces
            ``tensorize_sizes[target_axis]``.
    """

    target_nid: int
    factors: tuple[int, ...]
    target_axis: str | None = None


class Split(Transform):
    """Replace one axis-chain entry on a leaf with a chain of factors.

    See ``docs/superpowers/specs/2026-05-16-transforms-split-fuse-design.md``.
    """

    def analyze(self, ir: KernelIR) -> list[SplitOption]:
        """Enumerate every legal split option (outer-trip and tensorize)."""
        options: list[SplitOption] = []
        for nid in ir.tree.preorder():
            data = ir.tree.data(nid)
            if isinstance(data, ForNode):
                for factors in _factorizations(data.trip):
                    options.append(SplitOption(target_nid=nid, factors=factors, target_axis=None))
            elif isinstance(data, ISANode):
                if data.op_cls is NKIAlloc:
                    continue
                for axis, current in data.tensorize_sizes.items():
                    if axis not in data.axis_map:
                        continue
                    min_tile = data.op_cls.MIN_TILE_SIZE.get(axis)
                    max_tile = data.op_cls.MAX_TILE_SIZE.get(axis)
                    for factors in _factorizations(current):
                        last = factors[-1]
                        if min_tile is not None and last < min_tile:
                            continue
                        if max_tile is not None and last > max_tile:
                            continue
                        options.append(SplitOption(target_nid=nid, factors=factors, target_axis=axis))
        return options

    def apply(self, ir: KernelIR, option: SplitOption) -> KernelIR:
        """Re-check legality, deep-copy ``ir``, perform the split, return new IR."""
        self._check_legality(ir, option)
        new_ir = copy.deepcopy(ir)
        self._do_apply(new_ir, option)
        new_ir.dependency = Dependency(new_ir.tree)
        return new_ir

    def _check_legality(self, ir: KernelIR, option: SplitOption) -> None:
        """Raise :class:`TransformLegalityError` if ``option`` is invalid for ``ir``."""
        if len(option.factors) < 2:
            raise TransformLegalityError(f"Split.factors must have len >= 2; got {option.factors}")
        if any(f < 2 for f in option.factors):
            raise TransformLegalityError(f"Split.factors entries must be >= 2; got {option.factors}")
        target = _resolve(ir.tree, option.target_nid)
        if option.target_axis is None:
            if not isinstance(target, ForNode):
                raise TransformLegalityError(
                    f"Split outer-trip flavor requires target to be ForNode; got {type(target).__name__}"
                )
            if prod(option.factors) != target.trip:
                raise TransformLegalityError(
                    f"Split.factors product {prod(option.factors)} != ForNode.trip {target.trip}"
                )
        else:
            if not isinstance(target, ISANode):
                raise TransformLegalityError(
                    f"Split tensorize flavor requires target to be ISANode; got {type(target).__name__}"
                )
            if target.op_cls is NKIAlloc:
                raise TransformLegalityError(
                    "Split tensorize flavor: cannot split NKIAlloc tensorize_sizes " "(would change buffer placement)"
                )
            if option.target_axis not in target.axis_map:
                raise TransformLegalityError(
                    f"Split.target_axis={option.target_axis!r} not in leaf.axis_map={list(target.axis_map)}"
                )
            current = target.tensorize_sizes[option.target_axis]
            if prod(option.factors) != current:
                raise TransformLegalityError(
                    f"Split.factors product {prod(option.factors)} != tensorize_sizes[{option.target_axis!r}]={current}"
                )
            min_tile = target.op_cls.MIN_TILE_SIZE.get(option.target_axis)
            max_tile = target.op_cls.MAX_TILE_SIZE.get(option.target_axis)
            if min_tile is not None and option.factors[-1] < min_tile:
                raise TransformLegalityError(
                    f"Split.factors[-1]={option.factors[-1]} < MIN_TILE_SIZE[{option.target_axis!r}]={min_tile}"
                )
            if max_tile is not None and option.factors[-1] > max_tile:
                raise TransformLegalityError(
                    f"Split.factors[-1]={option.factors[-1]} > MAX_TILE_SIZE[{option.target_axis!r}]={max_tile}"
                )

    def _do_apply(self, ir: KernelIR, option: SplitOption) -> None:
        """Mutate ``ir.tree`` in place per ``option``. Caller already checked legality."""
        if option.target_axis is None:
            self._do_apply_outer_trip(ir, option)
        else:
            self._do_apply_tensorize(ir, option)

    def _do_apply_outer_trip(self, ir: KernelIR, option: SplitOption) -> None:
        """Replace a ForNode with a chain of new ForNodes whose trips are ``option.factors``.

        Sibling order under the parent is preserved: the new chain's top occupies
        the original target's slot in the parent's child list.
        """
        target_nid = option.target_nid
        target = ir.tree.data(target_nid)
        assert isinstance(target, ForNode)
        parent_nid = ir.tree.parent(target_nid)
        assert parent_nid is not None
        original_children_of_target = ir.tree.children(target_nid)

        """Build the new chain DETACHED so we can splice its top into target's old slot."""
        new_top_nid: int | None = None
        prev: int | None = None
        for trip in option.factors:
            new_nid = ir.tree.add_node(ForNode(dim=target.dim, trip=trip), parent=None)
            if new_top_nid is None:
                new_top_nid = new_nid
            if prev is not None:
                ir.tree.graph.add_edge(prev, new_nid)
            prev = new_nid

        """Reparent target's original children under the deepest new node."""
        assert prev is not None
        for child in original_children_of_target:
            ir.tree.graph.add_edge(prev, child)

        """Swap target_nid out for new_top_nid at the same position in parent's child list."""
        assert new_top_nid is not None
        _replace_in_parent_children(ir.tree, parent_nid, [target_nid], [new_top_nid])

        """Drop the now-orphaned target node."""
        ir.tree.graph.remove_node(target_nid)

    def _do_apply_tensorize(self, ir: KernelIR, option: SplitOption) -> None:
        """Insert ``len(factors)-1`` ForNodes above the leaf and update tensorize_sizes."""
        leaf_nid = option.target_nid
        leaf = ir.tree.data(leaf_nid)
        assert isinstance(leaf, ISANode)
        parent = ir.tree.parent(leaf_nid)
        assert parent is not None
        assert option.target_axis is not None
        concrete_dim = leaf.axis_map[option.target_axis]

        """Detach the leaf from its parent, then chain new ForNodes from ``parent``, then reattach."""
        ir.tree.graph.remove_edge(parent, leaf_nid)
        prev = parent
        for trip in option.factors[:-1]:
            new_nid = ir.tree.add_node(ForNode(dim=concrete_dim, trip=trip), parent=prev)
            prev = new_nid
        ir.tree.graph.add_edge(prev, leaf_nid)

        """Update tensorize. ISANode is a frozen dataclass - replace the node payload."""
        new_tensorize_sizes = dict(leaf.tensorize_sizes)
        new_tensorize_sizes[option.target_axis] = option.factors[-1]
        new_leaf = ISANode(
            op_cls=leaf.op_cls,
            reads=leaf.reads,
            writes=leaf.writes,
            rmw=leaf.rmw,
            tensorize_sizes=new_tensorize_sizes,
            axis_map=dict(leaf.axis_map),
            kwargs=dict(leaf.kwargs),
            location=leaf.location,
            dtype=leaf.dtype,
        )
        ir.tree.graph.nodes[leaf_nid]["data"] = new_leaf


def _resolve(tree: KernelTree, nid: int):
    """Return the node payload for ``nid`` or raise."""
    if nid not in tree.graph:
        raise TransformLegalityError(f"Split.target_nid={nid} is not a node in the IR tree")
    return tree.data(nid)


def _factorizations(n: int) -> list[tuple[int, ...]]:
    """Return every ordered factorization of ``n`` into 2..``_MAX_SPLIT_PARTS`` parts, each ``>= 2``.

    Example: ``_factorizations(16)`` returns ``(2, 8)``, ``(4, 4)``, ``(8, 2)``,
    ``(2, 2, 4)``, ``(2, 4, 2)``, ``(4, 2, 2)``.
    """
    out: list[tuple[int, ...]] = []
    for parts in range(2, _MAX_SPLIT_PARTS + 1):
        _enum_ordered_factorizations(n, parts, (), out)
    return out


def _enum_ordered_factorizations(
    remaining: int, parts_left: int, prefix: tuple[int, ...], out: list[tuple[int, ...]]
) -> None:
    """Append every ordered factorization of ``remaining`` into exactly ``parts_left`` factors >= 2."""
    if parts_left == 1:
        if remaining >= 2:
            out.append(prefix + (remaining,))
        return
    for f in range(2, remaining + 1):
        if remaining % f == 0 and remaining // f >= 2 ** (parts_left - 1):
            _enum_ordered_factorizations(remaining // f, parts_left - 1, prefix + (f,), out)


__all__ = ["Split", "SplitOption"]
