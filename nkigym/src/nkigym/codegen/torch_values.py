"""Source-emission values shared by the Torch frontend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TorchValue:
    """One emitted on-chip value and its logical shape."""

    name: str
    shape: tuple[int, ...]
    transposed: bool = False


@dataclass(frozen=True)
class TorchSegments:
    """Physical tensors comprising one logical output."""

    values: tuple[TorchValue, ...]
    axis: int = -1


def emit_activation(
    source: TorchValue, operation: str, name: str, scale: float, body: list[str], imports: set[str]
) -> TorchValue:
    """Emit one unary activation."""
    if operation == "gelu":
        return _emit_exact_gelu(source, name, body, imports)
    target = TorchValue(name, source.shape, source.transposed)
    imports.add("NKIActivation")
    body.append(f'{target.name} = NKIActivation(op="{operation}", scale={scale!r})(data={source.name})')
    return target


def _emit_exact_gelu(source: TorchValue, name: str, body: list[str], imports: set[str]) -> TorchValue:
    """Emit an accurate normal-CDF approximation for exact GELU."""
    absolute = TorchValue(f"{name}_absolute", source.shape, source.transposed)
    reciprocal = TorchValue(f"{name}_reciprocal", source.shape, source.transposed)
    polynomial = TorchValue(f"{name}_polynomial_0", source.shape, source.transposed)
    imports.update(("NKIActivation", "NKITensorTensor"))
    body.extend(
        (
            f'{absolute.name} = NKIActivation(op="abs", scale=1.0)(data={source.name})',
            f'{reciprocal.name} = NKIActivation(op="reciprocal", scale=0.23164190351963043, bias=1.0)'
            f"(data={absolute.name})",
            f'{polynomial.name} = NKIActivation(op="copy", scale=0.5307027697563171, '
            f"bias=-0.726576030254364)(data={reciprocal.name})",
        )
    )
    for index, coefficient in enumerate((0.710706889629364, -0.14224837720394135, 0.12741480767726898), 1):
        product = TorchValue(f"{name}_product_{index}", source.shape, source.transposed)
        body.append(
            f'{product.name} = NKITensorTensor(op="multiply")(data1={polynomial.name}, data2={reciprocal.name})'
        )
        polynomial = TorchValue(f"{name}_polynomial_{index}", source.shape, source.transposed)
        body.append(f'{polynomial.name} = NKIActivation(op="copy", bias={coefficient!r})(data={product.name})')
    weighted = TorchValue(f"{name}_weighted", source.shape, source.transposed)
    squared = TorchValue(f"{name}_squared", source.shape, source.transposed)
    exponential = TorchValue(f"{name}_exponential", source.shape, source.transposed)
    tail = TorchValue(f"{name}_tail", source.shape, source.transposed)
    delta = TorchValue(f"{name}_delta", source.shape, source.transposed)
    sign = TorchValue(f"{name}_sign", source.shape, source.transposed)
    signed = TorchValue(f"{name}_signed", source.shape, source.transposed)
    cdf = TorchValue(f"{name}_cdf", source.shape, source.transposed)
    target = TorchValue(name, source.shape, source.transposed)
    body.extend(
        (
            f'{weighted.name} = NKITensorTensor(op="multiply")' f"(data1={polynomial.name}, data2={reciprocal.name})",
            f'{squared.name} = NKIActivation(op="square", scale=1.0)(data={source.name})',
            f'{exponential.name} = NKIActivation(op="exp", scale=-0.5)(data={squared.name})',
            f'{tail.name} = NKITensorTensor(op="multiply")(data1={weighted.name}, data2={exponential.name})',
            f'{delta.name} = NKIActivation(op="copy", scale=-1.0, bias=0.5)(data={tail.name})',
            f'{sign.name} = NKIActivation(op="sign", scale=1.0)(data={source.name})',
            f'{signed.name} = NKITensorTensor(op="multiply")(data1={sign.name}, data2={delta.name})',
            f'{cdf.name} = NKIActivation(op="copy", bias=0.5)(data={signed.name})',
            f'{target.name} = NKITensorTensor(op="multiply")(data1={source.name}, data2={cdf.name})',
        )
    )
    return target


def emit_cast(source: TorchValue, class_name: str, name: str, body: list[str], imports: set[str]) -> TorchValue:
    """Emit one activation-backed dtype cast."""
    target = TorchValue(name, source.shape, source.transposed)
    imports.add(class_name)
    body.append(f"{target.name} = {class_name}()(data={source.name})")
    return target


def emit_cumsum(source: TorchValue, name: str, body: list[str], imports: set[str]) -> TorchSegments:
    """Emit a numerically stable chunked FP32 prefix sum."""
    width = min(256, source.shape[1])
    if len(source.shape) != 2 or source.shape[1] % width:
        raise ValueError(f"Torch cumsum requires a rank-two uniformly chunked width, got {source.shape}")
    imports.update(("NKIActivation", "NKIActivationReduce", "NKITensorScalar", "NKITensorSlice", "NKITensorTensorScan"))
    outputs: list[TorchValue] = []
    zero: TorchValue | None = None
    offset: TorchValue | None = None
    for index, start in enumerate(range(0, source.shape[1], width)):
        chunk = TorchValue(f"{name}_chunk_{index}", (source.shape[0], width))
        body.append(f"{chunk.name} = NKITensorSlice(start={start}, width={width})(src={source.name})")
        if zero is None:
            zero = TorchValue(f"{name}_zero", chunk.shape)
            body.append(f'{zero.name} = NKIActivation(op="copy", scale=0.0)(data={chunk.name})')
        scanned = TorchValue(f"{name}_scan_{index}", chunk.shape)
        body.append(
            f'{scanned.name} = NKITensorTensorScan(op0="add", op1="add", initial=0.0)'
            f"(data0={chunk.name}, data1={zero.name})"
        )
        adjusted = scanned
        if offset is not None:
            adjusted = TorchValue(f"{name}_adjusted_{index}", chunk.shape)
            body.append(f'{adjusted.name} = NKITensorScalar(op0="add")(data={scanned.name}, operand0={offset.name})')
        outputs.append(adjusted)
        if start + width < source.shape[1]:
            last = TorchValue(f"{name}_last_{index}", (source.shape[0], 1))
            offset = TorchValue(f"{name}_offset_{index}", (source.shape[0],))
            body.extend(
                (
                    f"{last.name} = NKITensorSlice(start={width - 1}, width=1)(src={adjusted.name})",
                    f'{offset.name} = NKIActivationReduce(op="copy", reduce_op="add")(data={last.name})',
                )
            )
    return TorchSegments(tuple(outputs))


def emit_reduce(
    source: TorchValue, name: str, operation: str, reduction: str, body: list[str], imports: set[str]
) -> TorchValue:
    """Emit one activation-backed free-axis reduction."""
    imports.add("NKIActivationReduce")
    body.append(f'{name} = NKIActivationReduce(op="{operation}", reduce_op="{reduction}")(data={source.name})')
    return TorchValue(name, (source.shape[0],))


def emit_slice(
    value: TorchValue, start: int, width: int, base: str, suffix: str, body: list[str], imports: set[str]
) -> TorchValue:
    """Emit one contiguous free-axis copy."""
    if len(value.shape) != 2 or start < 0 or width < 1 or start + width > value.shape[1]:
        raise ValueError(f"Torch tensor slice [{start}:{start + width}] is invalid for {value.shape}")
    target = TorchValue(f"{base}{suffix}", (value.shape[0], width))
    imports.add("NKITensorSlice")
    body.append(f"{target.name} = NKITensorSlice(start={start}, width={width})(src={value.name})")
    return target


def emit_topk(
    source: TorchValue, k: int, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchSegments, TorchSegments]:
    """Emit repeated native top-eight selection rounds."""
    if source.shape[1] > 16384:
        raise ValueError("Torch topk free-axis extent exceeds the max8 hardware limit of 16384")
    value_chunks: list[TorchValue] = []
    index_chunks: list[TorchValue] = []
    working = source
    imports.update(("NKIFindIndex8", "NKIMatchReplace8", "NKIMax8"))
    for offset in range(0, k, 8):
        width = min(8, k - offset)
        values = TorchValue(f"sbuf_{stem}_values_{offset}", (source.shape[0], 8))
        indices = TorchValue(f"sbuf_{stem}_indices_{offset}", (source.shape[0], 8))
        body.extend(
            (
                f"{values.name} = NKIMax8()(src={working.name})",
                f"{indices.name} = NKIFindIndex8()(data={working.name}, vals={values.name})",
            )
        )
        if width < 8:
            imports.add("NKITensorSlice")
            value_slice = TorchValue(f"{values.name}_{width}", (source.shape[0], width))
            index_slice = TorchValue(f"{indices.name}_{width}", (source.shape[0], width))
            body.extend(
                (
                    f"{value_slice.name} = NKITensorSlice(start=0, width={width})(src={values.name})",
                    f"{index_slice.name} = NKITensorSlice(start=0, width={width})(src={indices.name})",
                )
            )
            values, indices = value_slice, index_slice
        value_chunks.append(values)
        index_chunks.append(indices)
        if offset + width < k:
            working = TorchValue(f"sbuf_{stem}_remaining_{offset}", source.shape)
            prior = source.name if offset == 0 else f"sbuf_{stem}_remaining_{offset - 8}"
            body.append(
                f"{working.name} = NKIMatchReplace8(imm=float('-inf'))"
                f"(data={prior}, vals=sbuf_{stem}_values_{offset})"
            )
    return TorchSegments(tuple(value_chunks)), TorchSegments(tuple(index_chunks))


__all__ = [
    "TorchSegments",
    "TorchValue",
    "emit_activation",
    "emit_cast",
    "emit_cumsum",
    "emit_reduce",
    "emit_slice",
    "emit_topk",
]
