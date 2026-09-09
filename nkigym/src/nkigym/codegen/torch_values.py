"""Source-emission values shared by the Torch frontend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TorchValue:
    """One emitted on-chip value and its logical shape."""

    name: str
    shape: tuple[int, ...]
    transposed: bool = False
    is_hbm: bool = False
    storage_dtype: str | None = None


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
    """Emit native GELU plus a decaying polynomial accuracy correction."""
    squared = TorchValue(f"{name}_squared", source.shape, source.transposed)
    polynomial_0 = TorchValue(f"{name}_polynomial_0", source.shape, source.transposed)
    product = TorchValue(f"{name}_product", source.shape, source.transposed)
    polynomial_1 = TorchValue(f"{name}_polynomial_1", source.shape, source.transposed)
    exponential = TorchValue(f"{name}_exponential", source.shape, source.transposed)
    correction = TorchValue(f"{name}_correction", source.shape, source.transposed)
    approximate = TorchValue(f"{name}_approximate", source.shape, source.transposed)
    target = TorchValue(name, source.shape, source.transposed)
    imports.update(("NKIActivation", "NKIScalarTensorTensor", "NKITensorTensor"))
    body.extend(
        (
            f'{squared.name} = NKIActivation(op="square", scale=1.0)(data={source.name})',
            f'{polynomial_0.name} = NKIActivation(op="copy", scale=-0.00012687359622193486, '
            f"bias=0.00046550246152931305)(data={squared.name})",
            f'{product.name} = NKITensorTensor(op="multiply")' f"(data1={polynomial_0.name}, data2={squared.name})",
            f'{polynomial_1.name} = NKIScalarTensorTensor(op0="add", op1="multiply")'
            f"(data={product.name}, operand0=-6.871650057892193e-05, operand1={squared.name})",
            f'{exponential.name} = NKIActivation(op="exp", scale=-0.5461375669722928)(data={squared.name})',
            f'{correction.name} = NKITensorTensor(op="multiply")'
            f"(data1={polynomial_1.name}, data2={exponential.name})",
            f'{approximate.name} = NKIActivation(op="gelu", scale=1.0)(data={source.name})',
            f'{target.name} = NKITensorTensor(op="add")(data1={approximate.name}, data2={correction.name})',
        )
    )
    return target


def emit_cast(source: TorchValue, class_name: str, name: str, body: list[str], imports: set[str]) -> TorchValue:
    """Emit one activation-backed dtype cast."""
    storage_dtype = "float8_e4m3" if class_name == "NKIFloat8Cast" else None
    target = TorchValue(name, source.shape, source.transposed, storage_dtype=storage_dtype)
    imports.add(class_name)
    body.append(f"{target.name} = {class_name}()(data={source.name})")
    return target


def emit_cumsum(source: TorchValue, name: str, body: list[str], imports: set[str]) -> TorchSegments:
    """Emit a numerically stable chunked FP32 prefix sum."""
    width = min(2048, source.shape[1])
    if len(source.shape) != 2 or source.shape[1] % width:
        raise ValueError(f"Torch cumsum requires a rank-two uniformly chunked width, got {source.shape}")
    slice_operation = "NKIHBMFreeSlice" if source.is_hbm else "NKITensorSlice"
    imports.update(
        ("NKIActivationReduce", "NKITensorScalar", "NKITensorScalarCumulative", "NKITensorSlice", slice_operation)
    )
    outputs: list[TorchValue] = []
    offset: TorchValue | None = None
    for index, start in enumerate(range(0, source.shape[1], width)):
        chunk = TorchValue(f"{name}_chunk_{index}", (source.shape[0], width))
        body.append(f"{chunk.name} = {slice_operation}(start={start}, width={width})(src={source.name})")
        scanned = TorchValue(f"{name}_scan_{index}", chunk.shape)
        body.append(f'{scanned.name} = NKITensorScalarCumulative(op0="add", op1="add", imm0=0.0)(src={chunk.name})')
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
