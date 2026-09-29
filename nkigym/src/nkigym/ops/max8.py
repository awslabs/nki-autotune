"""Find the eight largest values with ``nisa.max8``."""

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue, emit_topk
from nkigym.ops.base import AxisRole, NKIOp, _operand_role
from nkigym.ops.index_iota import emit_packed_topk_indices, native_prefix
from nkigym.ops.iota import emit_first_sorted_value
from nkigym.ops.register_load import ControlEmitter
from nkigym.ops.transpose import emit_partition_sum
from nkigym.ops.uint32_cast import emit_uniform_prefix


class NKIMax8(NKIOp):
    """Return the eight largest values in each source partition."""

    NAME: ClassVar[str] = "max8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "K")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 8}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 8, "K": 8}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 16384, "K": 8}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source tensor."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKIMax8(src=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return descending top-eight values for CPU validation."""
        source = np.asarray(kwargs["src"])
        return np.sort(source, axis=1)[:, -8:][:, ::-1].copy()


def emit_batched_topk(
    emit: ControlEmitter,
    source: str,
    shape: tuple[int, int],
    selection: tuple[int, bool, bool],
    fallback: Callable[[tuple[str, str, str] | None], tuple[tuple[str, str], bool]],
) -> tuple[TorchValue, TorchValue]:
    """Select rows together and enter fallback only if at least one is invalid."""
    rows, width = shape
    count, sorted_output, full_sort = selection
    if full_sort and count == 1 and 1 <= rows <= 128 and 1 <= width <= 16:
        loaded = emit.cast("NKIFloat32Cast", emit.emit("NKILoad", f"src={source}"))
        return emit_first_sorted_value(emit, loaded, shape)
    if not (sorted_output and 1 <= rows <= 128 and (rows > 1 or full_sort) and (count // 8 + 1) * 8 <= width <= 16384):
        outputs, is_hbm = fallback(None)
    else:
        loaded = emit.cast("NKIFloat32Cast", emit.emit("NKILoad", f"src={source}"))
        values, indices, valid = native_prefix(emit, loaded, width, count, rows, per_partition=True)
        if full_sort:
            values, indices, valid = emit_uniform_prefix(
                emit, loaded, (rows, width, count, full_sort), (values, indices, valid)
            )
        outputs = (values, indices)
        emit.imports.add("NKIInplaceTensorCopy")
        copy_config = f"groups=1, partitions={rows}, start=0, width={count}, engine='vector'"
        flags = emit.emit("NKIStore", f"src={valid}")
        complete = emit_partition_sum(emit, valid)
        rejected = emit.scalar("less", complete, float(rows))
        with emit.guard(rejected):
            tables = tuple(emit.emit("NKIStore", f"src={value}") for value in outputs)
            recovered, _ = fallback((flags, tables[0], tables[1]))
            for destination, table in zip(outputs, recovered, strict=True):
                value = emit.emit("NKILoad", f"src={table}")
                emit.line(f"{destination} = NKIInplaceTensorCopy({copy_config})(src={value}, dst={destination})")
        is_hbm = False
    names = tuple(emit.emit("NKILoad", f"src={value}") if is_hbm else value for value in outputs)
    output_shape = (rows, count)
    return (
        TorchValue(names[0], output_shape, storage_dtype="float32"),
        TorchValue(names[1], output_shape, storage_dtype="uint32"),
    )


def emit_native_topk(
    source: TorchValue, k: int, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchValue, TorchValue]:
    """Select values and indices when equal-value index choices are interchangeable."""
    rows, width = source.shape
    if not (rows <= 128 and k <= width <= 16384):
        raise ValueError("native top-k requires at most 128 rows and a max8-compatible width")
    if source.is_hbm:
        imports.add("NKILoad")
        loaded = TorchValue(f"sbuf_{stem}_input", source.shape, storage_dtype=source.storage_dtype)
        body.append(f"{loaded.name} = NKILoad()(src={source.name})")
        source = loaded
    if k % 8:
        values, indices = emit_topk(source, k, stem, body, imports)
        if len(values.values) != 1:
            raise ValueError("a non-aligned native prefix must fit one top-eight result")
        return values.values[0], indices.values[0]
    working = TorchValue(f"sbuf_{stem}_working", source.shape, storage_dtype=source.storage_dtype)
    imports.update(("NKITensorCopy", "NKINCGather"))
    body.append(f"{working.name} = NKITensorCopy(engine='vector')(src={source.name})")
    positions = emit_packed_topk_indices(working, k, stem, body, imports).values[0]
    positions = TorchValue(positions.name, positions.shape, storage_dtype="uint32")
    values = TorchValue(f"sbuf_{stem}_selected", positions.shape, storage_dtype=source.storage_dtype)
    body.append(f"{values.name} = NKINCGather()(data={source.name}, indices={positions.name})")
    return values, positions


__all__ = ["NKIMax8"]
