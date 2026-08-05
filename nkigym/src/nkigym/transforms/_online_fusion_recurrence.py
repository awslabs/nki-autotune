"""Ordinary ISA emission for online-fusion recurrences."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from nkigym.ir.arith.expr import Const
from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, ForNode, ISANode, KernelTree
from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import NKIOp
from nkigym.ops.memset import NKIMemset
from nkigym.ops.scalar_tensor_tensor import NKIScalarTensorTensor
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_tensor import NKITensorTensor
from nkigym.transforms._online_fusion_types import (
    BinaryFactor,
    ConstantFactor,
    FactorExpression,
    StateFactor,
    UnaryFactor,
)


@dataclass(frozen=True)
class _CompiledFactor:
    """A compiled factor represented by either a tensor or literal."""

    tensor: str | None
    literal: float | None


class NameSupply:
    """Deterministic fresh-name allocator."""

    def __init__(self, names: set[str]) -> None:
        """Initialize from all existing tensor names."""
        self._names = set(names)

    def fresh(self, stem: str) -> str:
        """Return and reserve a name derived from ``stem``."""
        candidate = stem
        suffix = 1
        while candidate in self._names:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        self._names.add(candidate)
        return candidate


@dataclass(frozen=True)
class RecurrenceScope:
    """Mapped block and loop geometry for one recurrence operation."""

    block: BlockNode
    loops: tuple[ForNode, ...]


@dataclass
class RecurrenceIR:
    """Mutable ordinary-IR state used by recurrence emitters."""

    tree: KernelTree
    parent: int | None
    buffers: dict[str, Buffer]
    names: NameSupply
    regions: dict[str, BufferRegion] = field(default_factory=dict)
    scope: RecurrenceScope | None = None
    localize_temps: bool = False

    def region(self, tensor: str) -> BufferRegion:
        """Return the active mapped region or the full one-tile region."""
        region = self.regions.get(tensor)
        if region is None:
            region = full_region(self.buffers[tensor])
        return region


def compile_correction(
    context: RecurrenceIR,
    expression: FactorExpression,
    old_states: dict[int, str],
    new_states: dict[int, str],
    stem: str,
) -> str:
    """Materialize the stable ratio ``g(new) / g(old)``."""
    compiled = _compile_correction_value(context, expression, old_states, new_states, stem)
    if compiled.tensor is None:
        raise ValueError(f"correction for {expression!r} did not materialize a tensor")
    return compiled.tensor


def compile_factor(context: RecurrenceIR, expression: FactorExpression, states: dict[int, str], stem: str) -> str:
    """Materialize one state factor."""
    return _compile_factor(context, expression, states, stem)


def append_corrected_update(context: RecurrenceIR, state: str, correction: str, contribution: str, output: str) -> None:
    """Append ``output = state * correction + contribution``."""
    bindings = {
        "data": context.region(state),
        "operand0": context.region(correction),
        "operand1": context.region(contribution),
        "dst": context.region(output),
    }
    append_manual_block(
        context.tree, context.parent, NKIScalarTensorTensor, bindings, {"op0": "multiply", "op1": "add"}, context.scope
    )


def append_additive_update(context: RecurrenceIR, state: str, contribution: str, output: str) -> None:
    """Append ``output = state + contribution`` with a PSUM-compatible input."""
    bindings = {"data": context.region(state), "operand1": context.region(contribution), "dst": context.region(output)}
    append_manual_block(
        context.tree,
        context.parent,
        NKIScalarTensorTensor,
        bindings,
        {"op0": "multiply", "operand0": 1.0, "op1": "add"},
        context.scope,
    )


def append_scaled_output(context: RecurrenceIR, data: str, factor: str, output: str) -> None:
    """Append ``output = data * factor`` with free-axis broadcasting."""
    append_manual_block(
        context.tree,
        context.parent,
        NKITensorScalar,
        {"data": context.region(data), "operand0": context.region(factor), "dst": context.region(output)},
        {"op0": "multiply"},
        context.scope,
    )


def append_tensor_tensor(context: RecurrenceIR, data1: str, data2: str, output: str, operator: str) -> None:
    """Append one tensor-tensor block."""
    bindings = {"data1": context.region(data1), "data2": context.region(data2), "dst": context.region(output)}
    append_manual_block(context.tree, context.parent, NKITensorTensor, bindings, {"op": operator}, context.scope)


def append_copy(context: RecurrenceIR, source: str, destination: str) -> int:
    """Append one explicit state roll-forward copy and return its block id."""
    return append_manual_block(
        context.tree,
        context.parent,
        NKITensorCopy,
        {"src": context.region(source), "dst": context.region(destination)},
        {},
        context.scope,
    )


def append_initializer(context: RecurrenceIR, tensor: str, value: float) -> int:
    """Append a full-buffer memset block."""
    return append_manual_block(
        context.tree, context.parent, NKIMemset, {"dst": context.region(tensor)}, {"value": value}, context.scope
    )


def append_manual_block(
    tree: KernelTree,
    parent: int | None,
    op_cls: type[NKIOp],
    bindings: dict[str, BufferRegion],
    kwargs: dict[str, Any],
    scope: RecurrenceScope | None = None,
) -> int:
    """Append one ordinary ISA block with optional mapped loop geometry."""
    reads, writes = access_regions(op_cls, bindings, kwargs)
    if scope is None:
        block = BlockNode(iter_vars=(), iter_values=(), reads=reads, writes=writes, alloc_buffers=())
        loops: tuple[ForNode, ...] = ()
    else:
        block = replace(scope.block, reads=reads, writes=writes, alloc_buffers=())
        loops = scope.loops
    block_nid = tree.add_node(block, parent=parent)
    leaf_parent = block_nid
    for loop in loops:
        leaf_parent = tree.add_node(loop, parent=leaf_parent)
    tree.add_node(ISANode(op_cls=op_cls, operand_bindings=bindings, kwargs=kwargs), parent=leaf_parent)
    return block_nid


def access_regions(
    op_cls: type[NKIOp], bindings: dict[str, BufferRegion], kwargs: dict[str, Any]
) -> tuple[tuple[BufferRegion, ...], tuple[BufferRegion, ...]]:
    """Derive block reads and writes from operation operand metadata."""
    reads: list[BufferRegion] = []
    writes: list[BufferRegion] = []
    rmw_operands = op_cls.rmw_operands(kwargs)
    for slot, region in bindings.items():
        if slot in op_cls.INPUT_OPERANDS:
            reads.append(region)
        elif slot in rmw_operands:
            reads.append(region)
            writes.append(region)
        else:
            writes.append(region)
    return tuple(reads), tuple(writes)


def full_region(buffer: Buffer) -> BufferRegion:
    """Return a full logical region for a one-tile on-chip buffer."""
    if buffer.location == "shared_hbm":
        raise ValueError(f"manual online operation cannot use HBM buffer {buffer.name!r}")
    if buffer.shape[0] != 128:
        raise ValueError(f"online state buffer {buffer.name!r} must have leading extent 128")
    ranges = [(Const(value=0), Const(value=128))]
    if len(buffer.shape) == 2:
        ranges.append((Const(value=0), Const(value=buffer.shape[1])))
    return BufferRegion(tensor=buffer.name, ranges=tuple(ranges))


def _compile_correction_value(
    context: RecurrenceIR,
    expression: FactorExpression,
    old_states: dict[int, str],
    new_states: dict[int, str],
    stem: str,
) -> _CompiledFactor:
    """Apply structural ratio rules before falling back to direct division."""
    if isinstance(expression, ConstantFactor):
        result = _CompiledFactor(tensor=None, literal=1.0)
    elif isinstance(expression, BinaryFactor) and expression.operator == "multiply":
        left = _compile_correction_value(context, expression.left, old_states, new_states, f"{stem}_left")
        right = _compile_correction_value(context, expression.right, old_states, new_states, f"{stem}_right")
        result = _compile_binary_factor(context, "multiply", left, right, stem)
    elif isinstance(expression, UnaryFactor) and expression.operator == "rsqrt":
        operand, scale, bias = _flatten_copy_operand(expression)
        old_value = _compile_factor(context, operand, old_states, f"{stem}_old_arg")
        new_value = _compile_factor(context, operand, new_states, f"{stem}_new_arg")
        old_factor = _append_unary_factor(context, old_value, "sqrt", scale, bias, f"{stem}_old_sqrt")
        new_factor = _append_unary_factor(context, new_value, "rsqrt", scale, bias, f"{stem}_new_rsqrt")
        result = _CompiledFactor(tensor=_append_product(context, new_factor, old_factor, stem), literal=None)
    elif isinstance(expression, UnaryFactor) and expression.operator == "exp":
        operand, scale, _bias = _flatten_copy_operand(expression)
        old_value = _compile_factor(context, operand, old_states, f"{stem}_old_arg")
        new_value = _compile_factor(context, operand, new_states, f"{stem}_new_arg")
        difference = _new_temp(context, f"{stem}_difference", new_value)
        append_tensor_tensor(context, new_value, old_value, difference, "subtract")
        output = _new_temp(context, stem, difference)
        kwargs: dict[str, Any] = {"op": "exp"}
        if scale != 1.0:
            kwargs["scale"] = scale
        append_manual_block(
            context.tree,
            context.parent,
            NKIActivation,
            {"data": context.region(difference), "dst": context.region(output)},
            kwargs,
            context.scope,
        )
        result = _CompiledFactor(tensor=output, literal=None)
    elif isinstance(expression, UnaryFactor) and expression.operator == "reciprocal":
        operand, scale, bias = _flatten_copy_operand(expression)
        old_value = _compile_factor(context, operand, old_states, f"{stem}_old_arg")
        new_value = _compile_factor(context, operand, new_states, f"{stem}_new_arg")
        old_affine = _append_affine(context, old_value, scale, bias, f"{stem}_old")
        new_affine = _append_affine(context, new_value, scale, bias, f"{stem}_new")
        result = _CompiledFactor(tensor=_append_ratio(context, old_affine, new_affine, stem), literal=None)
    else:
        old_factor = _compile_factor(context, expression, old_states, f"{stem}_old")
        new_factor = _compile_factor(context, expression, new_states, f"{stem}_new")
        result = _CompiledFactor(tensor=_append_ratio(context, new_factor, old_factor, stem), literal=None)
    return result


def _compile_factor(context: RecurrenceIR, expression: FactorExpression, states: dict[int, str], stem: str) -> str:
    """Materialize one state factor and return its tensor name."""
    compiled = _compile_factor_value(context, expression, states, stem)
    if compiled.tensor is None:
        raise ValueError(f"factor {expression!r} did not materialize a tensor")
    return compiled.tensor


def _compile_factor_value(
    context: RecurrenceIR, expression: FactorExpression, states: dict[int, str], stem: str
) -> _CompiledFactor:
    """Recursively compile a factor expression."""
    if isinstance(expression, StateFactor):
        result = _CompiledFactor(tensor=states[expression.stage], literal=None)
    elif isinstance(expression, ConstantFactor):
        result = _CompiledFactor(tensor=None, literal=expression.value)
    elif isinstance(expression, UnaryFactor):
        operand_expression, scale, bias = _flatten_copy_operand(expression)
        operand = _compile_factor_value(context, operand_expression, states, f"{stem}_arg")
        if operand.tensor is None:
            raise ValueError(f"cannot apply {expression.operator} to a literal factor")
        output = _new_temp(context, f"{stem}_{expression.operator}", operand.tensor)
        kwargs: dict[str, Any] = {"op": expression.operator}
        if scale != 1.0:
            kwargs["scale"] = scale
        if bias != 0.0:
            kwargs["bias"] = bias
        append_manual_block(
            context.tree,
            context.parent,
            NKIActivation,
            {"data": context.region(operand.tensor), "dst": context.region(output)},
            kwargs,
            context.scope,
        )
        result = _CompiledFactor(tensor=output, literal=None)
    elif isinstance(expression, BinaryFactor):
        left = _compile_factor_value(context, expression.left, states, f"{stem}_left")
        right = _compile_factor_value(context, expression.right, states, f"{stem}_right")
        result = _compile_binary_factor(context, expression.operator, left, right, stem)
    else:
        raise TypeError(f"unsupported factor expression {type(expression).__name__}")
    return result


def _flatten_copy_operand(expression: UnaryFactor) -> tuple[FactorExpression, float, float]:
    """Fold nested affine ``copy`` factors into a unary operation."""
    operand = expression.operand
    scale = expression.scale
    bias = expression.bias
    while isinstance(operand, UnaryFactor) and operand.operator == "copy":
        outer_scale = scale
        scale = operand.scale * outer_scale
        bias = operand.bias * outer_scale + bias
        operand = operand.operand
    return operand, scale, bias


def _compile_binary_factor(
    context: RecurrenceIR, operator: str, left: _CompiledFactor, right: _CompiledFactor, stem: str
) -> _CompiledFactor:
    """Compile a binary factor over tensors and literals."""
    if left.tensor is not None and right.tensor is not None:
        output = _new_temp(context, stem, left.tensor)
        append_tensor_tensor(context, left.tensor, right.tensor, output, operator)
        result = _CompiledFactor(tensor=output, literal=None)
    elif left.tensor is not None and right.literal is not None:
        output = _new_temp(context, stem, left.tensor)
        _append_tensor_scalar(context, left.tensor, right.literal, output, operator, False)
        result = _CompiledFactor(tensor=output, literal=None)
    elif right.tensor is not None and left.literal is not None:
        output = _new_temp(context, stem, right.tensor)
        _append_tensor_scalar(context, right.tensor, left.literal, output, operator, True)
        result = _CompiledFactor(tensor=output, literal=None)
    elif left.literal is not None and right.literal is not None:
        functions = {
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "maximum": max,
        }
        result = _CompiledFactor(tensor=None, literal=float(functions[operator](left.literal, right.literal)))
    else:
        raise ValueError("binary factor has neither tensor nor literal operands")
    return result


def _append_ratio(context: RecurrenceIR, numerator: str, denominator: str, stem: str) -> str:
    """Materialize ``numerator / denominator``."""
    inverse = _new_temp(context, f"{stem}_inverse", denominator)
    append_manual_block(
        context.tree,
        context.parent,
        NKIActivation,
        {"data": context.region(denominator), "dst": context.region(inverse)},
        {"op": "reciprocal"},
        context.scope,
    )
    ratio = _new_temp(context, stem, numerator)
    append_tensor_tensor(context, numerator, inverse, ratio, "multiply")
    return ratio


def _append_product(context: RecurrenceIR, left: str, right: str, stem: str) -> str:
    """Materialize the product of two factor tensors."""
    product = _new_temp(context, stem, left)
    append_tensor_tensor(context, left, right, product, "multiply")
    return product


def _append_unary_factor(context: RecurrenceIR, data: str, operator: str, scale: float, bias: float, stem: str) -> str:
    """Materialize one affine unary factor."""
    output = _new_temp(context, stem, data)
    kwargs: dict[str, Any] = {"op": operator}
    if scale != 1.0:
        kwargs["scale"] = scale
    if bias != 0.0:
        kwargs["bias"] = bias
    append_manual_block(
        context.tree,
        context.parent,
        NKIActivation,
        {"data": context.region(data), "dst": context.region(output)},
        kwargs,
        context.scope,
    )
    return output


def _append_affine(context: RecurrenceIR, data: str, scale: float, bias: float, stem: str) -> str:
    """Materialize an affine factor only when it is non-identity."""
    result = data
    if scale != 1.0 or bias != 0.0:
        result = _new_temp(context, stem, data)
        kwargs: dict[str, Any] = {"op": "copy"}
        if scale != 1.0:
            kwargs["scale"] = scale
        if bias != 0.0:
            kwargs["bias"] = bias
        append_manual_block(
            context.tree,
            context.parent,
            NKIActivation,
            {"data": context.region(data), "dst": context.region(result)},
            kwargs,
            context.scope,
        )
    return result


def _append_tensor_scalar(
    context: RecurrenceIR, data: str, operand: float, output: str, operator: str, reverse: bool
) -> None:
    """Append one literal tensor-scalar block."""
    kwargs: dict[str, Any] = {"op0": operator, "operand0": operand}
    if reverse:
        kwargs["reverse0"] = True
    append_manual_block(
        context.tree,
        context.parent,
        NKITensorScalar,
        {"data": context.region(data), "dst": context.region(output)},
        kwargs,
        context.scope,
    )


def _new_temp(context: RecurrenceIR, stem: str, source: str) -> str:
    """Allocate a fresh fp32 SBUF factor tensor."""
    name = context.names.fresh(f"online_{stem}")
    source_buffer = context.buffers[source]
    source_region = context.region(source)
    shape = source_buffer.shape
    region = replace(source_region, tensor=name)
    if context.localize_temps:
        widths: list[int] = []
        for _lower, width in source_region.ranges:
            if not isinstance(width, Const):
                raise ValueError(f"localized factor width must be constant, got {width!r}")
            widths.append(width.value)
        shape = tuple(widths)
        region = BufferRegion(tensor=name, ranges=tuple((Const(value=0), Const(value=width)) for width in widths))
    context.buffers[name] = replace(
        source_buffer, name=name, shape=shape, location="sbuf", storage_dtype="float32", versions=1, list_len=1
    )
    context.regions[name] = region
    return name


__all__ = [
    "NameSupply",
    "RecurrenceIR",
    "RecurrenceScope",
    "access_regions",
    "append_additive_update",
    "append_copy",
    "append_corrected_update",
    "append_initializer",
    "append_manual_block",
    "append_scaled_output",
    "append_tensor_tensor",
    "compile_correction",
    "compile_factor",
    "full_region",
]
