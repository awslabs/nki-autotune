"""Align one buffer's physical free-axis pitch while preserving logical shape."""

from dataclasses import dataclass, replace
from math import lcm

from nkigym.ir import KernelIR
from nkigym.ir.arith.analyzer import Analyzer
from nkigym.ir.arith.expr import Add, Const, FloorDiv, Mod, Mul
from nkigym.ir.buffer_placement import layout_satisfies_alignment
from nkigym.ir.dependency_rebind import rebind_unchanged_dependency
from nkigym.ir.tree import AccessPattern, Buffer, ISANode
from nkigym.transforms.base import Transform, TransformLegalityError, TransformOption, copy_for_rewrite
from nkigym.transforms.helper.canonical_rewrite import replace_buffer

_ALIGNMENTS = tuple(1 << exponent for exponent in range(13))
_Views = dict[str, list[tuple[int, str, AccessPattern]]]
_Facts = tuple[_Views, dict[str, int]]


@dataclass(frozen=True)
class SetBufferAlignmentOption(TransformOption):
    """Choose a power-of-two free-axis pitch alignment, measured in elements."""

    tensor: str
    alignment: int


class SetBufferAlignment(Transform[SetBufferAlignmentOption]):
    """Change one allocation's physical pitch without adding logical elements."""

    def analyze(self, ir: KernelIR) -> list[SetBufferAlignmentOption]:
        """Enumerate changed physical layouts with exactly representable accesses."""
        buffers = ir.all_buffers()
        facts = _facts(ir)
        return [
            option
            for tensor in buffers
            for alignment in _ALIGNMENTS
            if _resolve(ir, option := SetBufferAlignmentOption(tensor, alignment), buffers, facts) is not None
        ]

    def apply(self, ir: KernelIR, option: SetBufferAlignmentOption) -> KernelIR:
        """Recheck, copy, and update one pitch and its physical access patterns."""
        resolved = _resolve(ir, option, ir.all_buffers(), _facts(ir))
        if resolved is None:
            raise TransformLegalityError(f"illegal SetBufferAlignment option: {option}")
        buffer, patterns = resolved
        result = copy_for_rewrite(ir)
        replace_buffer(result, buffer)
        for nid, slot, pattern in patterns:
            node = result.tree.isa(nid)
            result.tree.graph.nodes[nid]["data"] = replace(
                node, access_patterns={**node.access_patterns, slot: pattern}
            )
        result.dependency = rebind_unchanged_dependency(ir.dependency, result.tree)
        return result


def _facts(ir: KernelIR) -> _Facts:
    """Index physical views and producer alignment requirements in one pass."""
    result: _Views = {}
    alignments: dict[str, int] = {}
    for nid in ir.tree.preorder():
        node = ir.tree.data(nid)
        if isinstance(node, ISANode):
            for slot, pattern in node.access_patterns.items():
                tensor = node.operand_bindings[slot].tensor
                result.setdefault(tensor, []).append((nid, slot, pattern))
            for slot, required in node.op_cls.OUTPUT_TILE_ALIGNMENT_BYTES.items():
                region = node.operand_bindings.get(slot)
                if region is not None:
                    alignments[region.tensor] = lcm(alignments.get(region.tensor, 1), required)
    return result, alignments


def _resolve(
    ir: KernelIR, option: SetBufferAlignmentOption, buffers: dict[str, Buffer], facts: _Facts
) -> tuple[Buffer, list[tuple[int, str, AccessPattern]]] | None:
    """Resolve a pitch change without weakening native output alignment."""
    current = buffers.get(option.tensor)
    if (
        current is None
        or current.location not in {"sbuf", "psum"}
        or option.tensor in ir.return_names
        or type(option.alignment) is not int
        or option.alignment not in _ALIGNMENTS
    ):
        return None
    candidate = replace(current, free_alignment=option.alignment)
    if candidate.physical_shape() == current.physical_shape():
        return None
    if not layout_satisfies_alignment(candidate, facts[1].get(option.tensor, 1)):
        return None
    old_pitch, new_pitch = current.physical_shape()[2], candidate.physical_shape()[2]
    patterns = []
    for nid, slot, pattern in facts[0].get(option.tensor, ()):
        changed = _rebase_pattern(pattern, old_pitch, new_pitch)
        if changed is None:
            return None
        patterns.append((nid, slot, changed))
    return candidate, patterns


def _rebase_pattern(pattern: AccessPattern, old_pitch: int, new_pitch: int) -> AccessPattern | None:
    """Preserve affine tensor views whose free-coordinate span stays within a row.

    Decompose the base and each stride into row and free coordinates. A view
    crossing row boundaries through its free coordinate would need a different
    instruction view rank and is rejected instead of silently changing values.
    """
    analyzer = Analyzer()
    column = analyzer.simplify(Mod(left=pattern.offset, right=Const(value=old_pitch)))
    if not isinstance(column, Const):
        return None
    low = high = column.value
    dimensions = []
    for stride, extent in pattern.pattern:
        stride, extent = analyzer.simplify(stride), analyzer.simplify(extent)
        if not isinstance(stride, Const) or not isinstance(extent, Const) or extent.value < 1:
            return None
        rows = abs(stride.value) // old_pitch * (-1 if stride.value < 0 else 1)
        residual = stride.value - rows * old_pitch
        span = residual * (extent.value - 1)
        low += min(span, 0)
        high += max(span, 0)
        dimensions.append((Const(value=rows * new_pitch + residual), extent))
    if low < 0 or high >= min(old_pitch, new_pitch):
        return None
    offset = analyzer.simplify(
        Add(
            left=Mul(left=FloorDiv(left=pattern.offset, right=Const(value=old_pitch)), right=Const(value=new_pitch)),
            right=column,
        )
    )
    return AccessPattern(pattern=tuple(dimensions), offset=offset)


__all__ = ["SetBufferAlignment", "SetBufferAlignmentOption"]
