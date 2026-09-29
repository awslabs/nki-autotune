"""Native index search with one broadcast search value per partition."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import AxisRole, NKIOp, _operand_role
from nkigym.ops.nc_gather import _emit_max_with_indices
from nkigym.ops.stream_shuffle import emit_partition_reshape


class NKIBroadcastFindIndex8(NKIOp):
    """Find up to eight successive occurrences of one value in each partition."""

    NAME: ClassVar[str] = "nc_find_index8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "vals": ("P",), "dst": ("P", "K")}
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"vals": (("P",), ("K",))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "vals"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf"}), "vals": frozenset({"sbuf"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"data": "float32", "vals": "float32"}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 8}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 8, "K": 8}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 16384, "K": 8}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip inputs."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in ("data", "vals")):
            raise TypeError("broadcast index search requires SBUF inputs")
        if any(np.asarray(kwargs[name]).dtype != np.dtype("float32") for name in ("data", "vals")):
            raise TypeError("broadcast index search requires float32 inputs")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return increasing match positions, followed by unmatched sentinels."""
        data = np.asarray(kwargs["data"])
        values = np.asarray(kwargs["vals"]).reshape(data.shape[0])
        result = np.full((data.shape[0], 8), np.iinfo(np.uint32).max, dtype=np.uint32)
        for row in range(data.shape[0]):
            matches = np.flatnonzero(data[row] == values[row])[:8]
            result[row, : matches.size] = matches
        return result


def emit_indexed_max(
    source: TorchValue, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchSegments, TorchSegments]:
    """Find the first maximum or first NaN with independently selected row indices.

    A comparison with negative infinity marks every non-NaN value, including
    infinities. Its row minimum supplies a present search value and indicates
    whether NaNs exist. Gather the original value at the selected valid index.
    """
    rows, width = source.shape
    if source.storage_dtype != "float32" or rows < 1 or not 8 <= width <= 16384:
        return _emit_max_with_indices(source, stem, body, imports)
    emit = TorchArithmetic(f"sbuf_{stem}_indexed", body, imports)
    maximum = emit.emit("NKITensorReduce", f"data={source.name}", "op='max', axis=1")
    positions = emit.emit("NKIBroadcastFindIndex8", f"data={source.name}, vals={maximum}")
    indices = emit.emit("NKITensorSlice", f"src={positions}", "start=0, width=1, engine='vector'")
    valid = emit.scalar("greater_equal", source.name, "float('-inf')")
    all_valid = emit.emit("NKITensorReduce", f"data={valid}", "op='minimum', axis=1")
    nan_positions = emit.emit("NKIBroadcastFindIndex8", f"data={valid}, vals={all_valid}")
    first_nan = emit.emit("NKITensorSlice", f"src={nan_positions}", "start=0, width=1, engine='vector'")
    predicate = emit.cast("NKIUInt32Cast", all_valid)
    imports.add("NKITensorCopyPredicated")
    emit.body.append(
        f"{indices} = NKITensorCopyPredicated(reverse_pred=True)(src={first_nan}, predicate={predicate}, dst={indices})"
    )
    values = emit.emit("NKINCGather", f"data={source.name}, indices={indices}")
    return (
        TorchSegments((TorchValue(values, (rows, 1), storage_dtype=source.storage_dtype),)),
        TorchSegments((TorchValue(indices, (rows, 1), storage_dtype="uint32"),)),
    )


def emit_grouped_maxima(
    emit: TorchArithmetic, values: tuple[TorchValue, TorchValue], shape: tuple[int, int], index_bound: int
) -> tuple[str, str]:
    """Pack local maxima and their exactly representable indices into row tables."""
    rows, parts = shape
    if not 0 < index_bound <= 1 << 24:
        raise ValueError("local maximum indices must be exactly representable in float32")
    on_chip = rows > 1 and values[0].storage_dtype == "float32"
    tables = []
    for value in values:
        if on_chip:
            source = emit.cast("NKIFloat32Cast", value.name) if value.storage_dtype == "uint32" else value.name
            table = emit_partition_reshape(emit, source, (rows * parts, 1), rows)
        else:
            table = emit.emit("NKIDMATranspose", f"src={value.name}")
            if rows > 1:
                stored = emit.emit("NKIStore", f"src={table}")
                table = emit.emit("NKIGroupedLoad", f"src={stored}", f"groups=1, rows=1, stages={rows}")
        tables.append(table)
    return tables[0], tables[1]


__all__ = ["NKIBroadcastFindIndex8"]
