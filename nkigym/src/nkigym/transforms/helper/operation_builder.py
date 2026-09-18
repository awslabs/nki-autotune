"""Ordinary-IR block emission shared by structural rewrites."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from nkigym.ir.arith.expr import Const
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.base import NKIOp


class NameSupply:
    """Allocate deterministic names without colliding with existing tensors."""

    def __init__(self, names: set[str]) -> None:
        """Initialize the reserved-name set."""
        self._names = set(names)

    def fresh(self, stem: str) -> str:
        """Return and reserve one name derived from ``stem``."""
        candidate = stem
        suffix = 1
        while candidate in self._names:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        self._names.add(candidate)
        return candidate


@dataclass(frozen=True)
class OperationScope:
    """Mapped block and loop geometry for one emitted operation."""

    block: BlockNode
    loops: tuple[ForNode, ...]


@dataclass
class OperationBuilder:
    """Mutable state for appending ordinary ISA blocks."""

    tree: KernelTree
    parent: int | None
    buffers: dict[str, Buffer]
    names: NameSupply
    regions: dict[str, BufferRegion] = field(default_factory=dict)
    scope: OperationScope | None = None
    localize_temps: bool = False

    def region(self, tensor: str) -> BufferRegion:
        """Return an active mapped region or one complete on-chip tile."""
        region = self.regions.get(tensor)
        if region is None:
            buffer = self.buffers[tensor]
            if buffer.location == "shared_hbm" or buffer.shape[0] != 128:
                raise ValueError(f"tensor {tensor!r} has no explicit online-fusion region")
            ranges = [(Const(value=0), Const(value=128))]
            ranges.extend((Const(value=0), Const(value=extent)) for extent in buffer.shape[1:])
            region = BufferRegion(tensor=tensor, ranges=tuple(ranges))
        return region

    def append(
        self,
        op_cls: type[NKIOp],
        bindings: Mapping[str, BufferRegion | str],
        kwargs: dict[str, Any],
        scope: OperationScope | None = None,
    ) -> int:
        """Append one ISA block from explicit regions or named active tensors."""
        regions = {slot: self.region(value) if isinstance(value, str) else value for slot, value in bindings.items()}
        rmw_operands = op_cls.rmw_operands(kwargs)
        reads = tuple(
            region for slot, region in regions.items() if slot in op_cls.INPUT_OPERANDS or slot in rmw_operands
        )
        writes = tuple(region for slot, region in regions.items() if slot not in op_cls.INPUT_OPERANDS)
        active_scope = self.scope if scope is None else scope
        if active_scope is None:
            block = BlockNode(iter_vars=(), iter_values=(), reads=reads, writes=writes, alloc_buffers=())
            loops: tuple[ForNode, ...] = ()
        else:
            block = replace(active_scope.block, reads=reads, writes=writes, alloc_buffers=())
            loops = active_scope.loops
        block_nid = self.tree.add_node(block, parent=self.parent)
        parent = block_nid
        for loop in loops:
            parent = self.tree.add_node(loop, parent=parent)
        self.tree.add_node(ISANode(op_cls=op_cls, operand_bindings=regions, kwargs=kwargs), parent=parent)
        return block_nid

    def temp(self, stem: str, source: str) -> str:
        """Allocate an fp32 SBUF temporary matching one active source region."""
        name = self.names.fresh(f"online_{stem}")
        source_buffer = self.buffers[source]
        source_region = self.region(source)
        shape = source_buffer.shape
        region = replace(source_region, tensor=name)
        if self.localize_temps:
            widths = tuple(width.value for _lower, width in source_region.ranges if isinstance(width, Const))
            if len(widths) != len(source_region.ranges):
                raise ValueError("localized factor widths must be constant")
            shape = widths
            region = BufferRegion(tensor=name, ranges=tuple((Const(value=0), Const(value=width)) for width in widths))
        self.buffers[name] = replace(
            source_buffer, name=name, shape=shape, location="sbuf", storage_dtype="float32", versions=1, list_len=1
        )
        self.regions[name] = region
        return name


__all__ = ["NameSupply", "OperationBuilder", "OperationScope"]
