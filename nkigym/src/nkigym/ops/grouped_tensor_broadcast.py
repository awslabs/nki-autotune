"""Binary operations with a broadcast scalar for each contiguous free-axis group."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role
from nkigym.ops.tensor_tensor import _OPS


class NKIGroupedTensorBroadcast(NKIOp):
    """Apply FP32 binary arithmetic over grouped free-axis views."""

    NAME: ClassVar[str] = "tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data1": ("P", "G", "T"),
        "data2": ("P", "G"),
        "dst": ("P", "G", "T"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data1": (("P",), ("G", "T")),
        "data2": (("P",), ("G",)),
        "dst": (("P",), ("G", "T")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G",), ("T",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data1", "data2"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "data1": frozenset({"sbuf", "psum"}),
        "data2": frozenset({"sbuf"}),
    }
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        slot: frozenset({"float16", "bfloat16", "float32"}) for slot in INPUT_OPERANDS
    }
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "partitions", "G": "groups", "T": "chunks"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = dict.fromkeys(("P", "G", "T"), 1)
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "G": None, "T": None}
    TILABLE_FIXED_AXES: ClassVar[frozenset[str]] = frozenset({"G"})
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"partitions", "groups", "chunks"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, partitions: int, groups: int, chunks: int, op: str) -> None:
        """Configure the physical groups and binary operator."""
        if min(partitions, groups, chunks) < 1 or op not in _OPS:
            raise ValueError("grouped tensor broadcast requires positive extents and a supported operator")
        super().__init__(partitions=partitions, groups=groups, chunks=chunks, op="maximum" if op == "max" else op)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Declare the exact binary operation and its broadcast operand."""
        return PointwiseContract(
            operator=str(kwargs["op"]),
            input_operands=("data1", "data2"),
            output_operand="dst",
            broadcast_operands=frozenset({"data2"}),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require supported on-chip locations and matching grouped shapes."""
        p, g, t = (int(kwargs[name]) for name in ("partitions", "groups", "chunks"))
        for slot, shape in (("data1", (p, g * t)), ("data2", (p, g))):
            role = _operand_role(kwargs[slot])
            if role is not None and role not in self.INPUT_LOCATIONS[slot]:
                raise TypeError(f"grouped tensor broadcast {slot} has unsupported location {role}")
            if np.asarray(kwargs[slot]).shape != shape:
                raise ValueError(f"grouped tensor broadcast {slot} requires shape {shape}")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Evaluate the broadcast in its logical three-dimensional shape."""
        p, g, t = (int(kwargs[name]) for name in ("partitions", "groups", "chunks"))
        left = np.asarray(kwargs["data1"], dtype=np.float32).reshape(p, g, t)
        right = np.asarray(kwargs["data2"], dtype=np.float32).reshape(p, g, 1)
        return _OPS[str(kwargs["op"])](left, right).reshape(p, g * t)


__all__ = ["NKIGroupedTensorBroadcast"]
