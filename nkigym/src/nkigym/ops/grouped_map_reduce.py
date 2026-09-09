"""Grouped activation and free-axis reduction."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator

_MAPS = {"copy": lambda data: data, "exp": np.exp}
_REDUCTIONS = {"add": np.sum, "max": np.max}


class NKIGroupedMapReduce(NKIOp):
    """Map and reduce uniform chunks while preserving row groups."""

    NAME: ClassVar[str] = "activation_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "G", "T", "F"),
        "bias": ("P", "G"),
        "dst": ("P", "G", "T", "F"),
        "reduce_res": ("P", "G", "T"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("P",), ("G", "T", "F")),
        "bias": (("P",), ("G",)),
        "dst": (("P",), ("G", "T", "F")),
        "reduce_res": (("P",), ("G", "T")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "bias"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "T": "chunks", "F": "width"}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GPTF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "T": 1, "F": None}
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "chunks", "width"})
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {"reduce_res": "float32"}
    OUTPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"reduce_res": "float32"}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int, chunks: int, width: int, op: str, reduce_op: str) -> None:
        """Configure one grouped mapped reduction."""
        if op not in _MAPS or reduce_op not in _REDUCTIONS:
            raise ValueError(f"unsupported grouped map/reduce {op!r}/{reduce_op!r}")
        super().__init__(groups=groups, partitions=partitions, chunks=chunks, width=width, op=op, reduce_op=reduce_op)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the configured mapped reduction contract."""
        return ReductionContract(
            input_operand="data",
            output_operand="reduce_res",
            reduction_axis="F",
            combinator=reduction_combinator(str(kwargs["reduce_op"])),
            map_operator=str(kwargs["op"]),
            bias_operand="bias",
            mapped_output_operand="dst",
            mapped_op_cls=NKIActivation,
            mapped_input_operands=("data", "bias"),
            mapped_excluded_kwargs=frozenset({"reduce_op"}),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require on-chip source and optional bias tensors."""
        if any(_operand_role(kwargs.get(name)) not in {None, "sbuf", "psum"} for name in ("data", "bias")):
            raise TypeError("NKIGroupedMapReduce expects on-chip operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one row-major partial reduction per chunk."""
        g, p, t, f = (int(kwargs[name]) for name in ("groups", "partitions", "chunks", "width"))
        data = np.asarray(kwargs["data"], dtype=np.float32).reshape(p, g, t, f)
        if isinstance(bias := kwargs.get("bias"), np.ndarray):
            data = data + np.asarray(bias, dtype=np.float32).reshape(p, g, 1, 1)
        mapped = _MAPS[str(kwargs["op"])](data)
        return np.asarray(_REDUCTIONS[str(kwargs["reduce_op"])](mapped, axis=3), dtype=np.float32).reshape(p, g * t)


__all__ = ["NKIGroupedMapReduce"]
