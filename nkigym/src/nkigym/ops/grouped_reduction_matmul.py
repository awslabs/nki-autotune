"""Grouped tiled reduction through ``nisa.nc_matmul``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, BilinearReductionContract, NKIOp, ReduceCombinator, _operand_role


class NKIGroupedReductionMatmul(NKIOp):
    """Accumulate grouped PV products into packed output tiles."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("T", "D", "K", "G", "Q", "P"),
        "moving": ("K", "G", "T", "D", "H"),
        "dst": ("G", "Q", "P", "H"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "stationary": (("T", "D", "K"), ("G", "Q", "P")),
        "moving": (("K",), ("G", "T", "D", "H")),
        "dst": (("G", "Q", "P"), ("H",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    FIRST_WRITE_AXES: ClassVar[tuple[str, ...]] = ("T", "D")
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "G": "groups",
        "Q": "queries",
        "T": "tiles",
        "D": "subtiles",
        "K": "partition_width",
        "P": "partitions",
        "H": "output_width",
    }
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {
        "K": AxisRole.ACCUMULATION,
        "T": AxisRole.ACCUMULATION,
        "D": AxisRole.ACCUMULATION,
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "GQTDKPH"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "Q": 1, "T": 1, "D": 1, "K": 128, "P": 128, "H": 512}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {"groups", "queries", "tiles", "subtiles", "partition_width", "partitions", "output_width"}
    )
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def __init__(
        self, groups: int, queries: int, tiles: int, subtiles: int, partitions: int, output_width: int
    ) -> None:
        """Configure grouped PV reduction extents."""
        super().__init__(
            groups=groups,
            queries=queries,
            tiles=tiles,
            subtiles=subtiles,
            partition_width=128,
            partitions=partitions,
            output_width=output_width,
        )

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> BilinearReductionContract:
        """Return the grouped additive matrix reduction contract."""
        _ = kwargs
        return BilinearReductionContract("stationary", "moving", "dst", "K", ReduceCombinator("add", 0.0))

    @classmethod
    def first_write_overwrites(cls, operand: str, kwargs: Mapping[str, Any]) -> bool:
        """Return the configured PSUM first-write behavior."""
        return operand == "dst" and kwargs.get("accumulate") is not True

    @classmethod
    def rmw_operands(cls, kwargs: Mapping[str, Any]) -> frozenset[str]:
        """Treat the first grouped matmul as a write-only destination."""
        return frozenset() if kwargs.get("accumulate") is False else cls.RMW_OPERANDS

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF operands."""
        for slot in ("stationary", "moving"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKIGroupedReductionMatmul({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return grouped PV reductions."""
        g, q, t, d, p, h = (
            int(kwargs[key]) for key in ("groups", "queries", "tiles", "subtiles", "partitions", "output_width")
        )
        stationary = np.asarray(kwargs["stationary"], dtype=np.float32).reshape(t, d, 128, g, q, p)
        moving = np.asarray(kwargs["moving"], dtype=np.float32).reshape(128, g, t, d, h)
        output = np.einsum("tdkgqp,kgtdh->gqph", stationary, moving)
        return output.reshape(g * q * p, h)


__all__ = ["NKIGroupedReductionMatmul"]
