"""Small float32 SBUF transposes executed by the Vector Engine."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PermutationContract, _operand_role


class NKIVectorTranspose(NKIOp):
    """Transpose ``data(P, F)`` directly to ``dst(F, P)`` in SBUF."""

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F"), "dst": ("F", "P")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"float32"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 32, "F": 32}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self) -> None:
        """Select the Vector Engine and its direct SBUF output."""
        super().__init__(engine="vector")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Describe the permutation without casting or arithmetic."""
        _ = kwargs
        return PermutationContract(input_operand="data", output_operand="dst", permutation=(1, 0))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a rank-two float32 SBUF tile."""
        value = np.asarray(kwargs["data"])
        if _operand_role(kwargs["data"]) not in {None, "sbuf"} or value.ndim != 2:
            raise TypeError("NKIVectorTranspose requires a rank-two SBUF source")
        if value.dtype != np.float32 or kwargs["engine"] != "vector":
            raise TypeError("NKIVectorTranspose requires float32 storage and the Vector Engine")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the transposed values with their float32 bits unchanged."""
        return np.asarray(kwargs["data"]).T.copy()


__all__ = ["NKIVectorTranspose"]
