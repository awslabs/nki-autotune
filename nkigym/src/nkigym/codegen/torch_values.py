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
    target = TorchValue(name, source.shape, source.transposed, storage_dtype=source.storage_dtype)
    imports.add("NKIActivation")
    body.append(f'{target.name} = NKIActivation(op="{operation}", scale={scale!r})(data={source.name})')
    return target


def emit_cast(source: TorchValue, class_name: str, name: str, body: list[str], imports: set[str]) -> TorchValue:
    """Emit one activation-backed dtype cast."""
    storage_dtype = {"NKIFloat8Cast": "float8_e4m3", "NKIFloat32Cast": "float32"}.get(class_name)
    target = TorchValue(name, source.shape, source.transposed, storage_dtype=storage_dtype)
    imports.add(class_name)
    body.append(f"{target.name} = {class_name}()(data={source.name})")
    return target


def emit_cumsum(source: TorchValue, name: str, body: list[str], imports: set[str]) -> TorchSegments:
    """Emit a numerically stable chunked FP32 prefix sum."""
    if len(source.shape) != 2 or source.shape[1] < 1:
        raise ValueError(f"Torch cumsum requires a nonempty rank-two input, got {source.shape}")
    width = min(256, source.shape[1])
    imports.update(("NKITensorScalar", "NKITensorScalarCumulative"))
    outputs: list[TorchValue] = []
    offset: TorchValue | None = None
    for index, start in enumerate(range(0, source.shape[1], width)):
        chunk_width = min(width, source.shape[1] - start)
        chunk = emit_slice(source, start, chunk_width, name, f"_chunk_{index}", body, imports)
        chunk = emit_cast(chunk, "NKIFloat32Cast", f"{name}_float_chunk_{index}", body, imports)
        scanned = TorchValue(f"{name}_scan_{index}", chunk.shape, storage_dtype="float32")
        body.append(f'{scanned.name} = NKITensorScalarCumulative(op0="add", op1="add", imm0=0.0)(src={chunk.name})')
        adjusted = scanned
        if offset is not None:
            adjusted = TorchValue(f"{name}_adjusted_{index}", chunk.shape, storage_dtype="float32")
            body.append(f'{adjusted.name} = NKITensorScalar(op0="add")(data={scanned.name}, operand0={offset.name})')
        outputs.append(adjusted)
        if start + chunk_width < source.shape[1]:
            last = emit_slice(adjusted, chunk_width - 1, 1, name, f"_last_{index}", body, imports)
            offset = emit_reduce(last, f"{name}_offset_{index}", "copy", "add", body, imports)
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
    target = TorchValue(f"{base}{suffix}", (value.shape[0], width), storage_dtype=value.storage_dtype)
    operation = "NKIHBMFreeSlice" if value.is_hbm else "NKITensorSlice"
    imports.add(operation)
    body.append(f"{target.name} = {operation}(start={start}, width={width})(src={value.name})")
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
            values, indices = (
                emit_slice(value, 0, width, value.name, f"_{width}", body, imports) for value in (values, indices)
            )
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
