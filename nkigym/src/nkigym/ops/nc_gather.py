"""Within-partition SBUF gathering with ``nisa.nc_n_gather``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, _operand_role


class NKINCGather(NKIOp):
    """Gather free-axis elements independently in each partition."""

    NAME: ClassVar[str] = "nc_n_gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "indices": ("P", "N"), "dst": ("P", "N")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "indices"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf"}), "indices": frozenset({"sbuf"})}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"indices": "uint32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "N": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF data and indices with matching partitions."""
        for slot in ("data", "indices"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKINCGather({slot}=<role={role}>) expects SBUF")
        if np.asarray(kwargs["data"]).shape[0] != np.asarray(kwargs["indices"]).shape[0]:
            raise ValueError("nc_n_gather requires matching partition extents")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Gather flattened free-axis indices for CPU validation."""
        data = np.asarray(kwargs["data"])
        indices = np.asarray(kwargs["indices"]).astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= data.shape[1]):
            raise ValueError("nc_n_gather indices exceed the source free-axis extent")
        return np.take_along_axis(data, indices, axis=1)


def emit_clamped_gather(emit: TorchArithmetic, source: str, position: str, width: int) -> str:
    """Gather a fixed-width table using a bounded speculative index."""
    bounded = emit.emit(
        "NKITensorScalarSequence",
        f"data={position}, operand0=0.0, operand1={float(width - 1)}",
        "op0='maximum', op1='minimum', engine='vector'",
    )
    indices = emit.cast("NKIUInt32Cast", bounded)
    return emit.emit("NKINCGather", f"data={source}, indices={indices}")


def emit_pair_cursors(
    emit: TorchArithmetic,
    tables: tuple[str, str],
    counts: tuple[str, str],
    starts: tuple[str, str],
    bounds: tuple[str, str],
    swapped: str,
    width: int,
) -> tuple[str, str, str]:
    """Read the next unmatched positions and the last matched right position."""
    low, high = bounds
    left = emit.scalar("add", emit_clamped_gather(emit, tables[0], swapped, width), starts[0])
    left = emit.select(
        emit.binary("greater", counts[0], swapped),
        left,
        emit.binary("minimum", high, emit.scalar("add", starts[0], float(width))),
    )
    rank = emit.scalar("subtract", emit.binary("subtract", counts[1], swapped), 1.0)
    right = emit.scalar("add", emit_clamped_gather(emit, tables[1], rank, width), starts[1])
    right = emit.select(
        emit.binary("greater", counts[1], swapped),
        emit.scalar("add", right, 1.0),
        emit.binary("maximum", low, starts[1]),
    )
    previous = emit.scalar(
        "add", emit_clamped_gather(emit, tables[1], emit.binary("subtract", counts[1], swapped), width), starts[1]
    )
    right = emit.select(emit.scalar("greater", swapped, 0.0), emit.binary("minimum", right, previous), right)
    crossing = emit.binary(
        "multiply", emit.binary("greater", counts[0], swapped), emit.binary("greater", counts[1], swapped)
    )
    return left, emit.select(crossing, left, right), previous


__all__ = ["NKINCGather"]
