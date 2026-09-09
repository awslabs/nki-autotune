"""Transpose grouped count vectors into one packed PSUM row."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PermutationContract, _operand_role


class NKIGroupedCountsTranspose(NKIOp):
    """Transpose each grouped ``(P, 1)`` count tile into one row slice."""

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("G", "P", "O"), "dst": ("O", "G", "P")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("G", "P"), ("O",)),
        "dst": (("O",), ("G", "P")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "O": 1}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 128, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "O": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"

    def __init__(self, groups: int, partitions: int) -> None:
        """Configure the grouped count-row extents."""
        super().__init__(groups=groups, partitions=partitions)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Return the grouped row-transpose contract."""
        _ = kwargs
        return PermutationContract(input_operand="data", output_operand="dst", permutation=(2, 0, 1))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one SBUF source."""
        if (role := _operand_role(kwargs["data"])) is not None and role != "sbuf":
            raise TypeError(f"NKIGroupedCountsTranspose(data=<role={role}>) expects SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return grouped counts as one contiguous row."""
        groups, partitions = int(kwargs["groups"]), int(kwargs["partitions"])
        return np.asarray(kwargs["data"]).reshape(groups, partitions).reshape(1, groups * partitions)


__all__ = ["NKIGroupedCountsTranspose"]
