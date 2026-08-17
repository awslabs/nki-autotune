"""Within-partition SBUF gathering with ``nisa.nc_n_gather``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKINCGather(NKIOp):
    """Gather free-axis elements independently in each partition."""

    NAME: ClassVar[str] = "nc_n_gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "indices": ("P", "N"), "dst": ("P", "N")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "indices"})
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


__all__ = ["NKINCGather"]
