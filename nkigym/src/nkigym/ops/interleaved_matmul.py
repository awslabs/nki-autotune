"""Matrix products whose contraction tiles occupy packed free-axis groups."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from nkigym.ops.base import AxisRole, BilinearReductionContract, NKIOp, ReduceCombinator, _operand_role


class NKIInterleavedMatmul(NKIOp):
    """Sum native partition contractions over a separate packed tile axis."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("P", "K", "M"),
        "moving": ("P", "K", "N"),
        "dst": ("M", "N"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "stationary": (("P",), ("K", "M")),
        "moving": (("P",), ("K", "N")),
        "dst": (("M",), ("N",)),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "stationary": (("P",), ("M",)),
        "moving": (("P",), ("N",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {operand: frozenset({"sbuf"}) for operand in INPUT_OPERANDS}
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "rmw"
    REDUCE_COMBINATOR: ClassVar[ReduceCombinator | None] = ReduceCombinator("add", 0.0)
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION, "P": AxisRole.ACCUMULATION}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": 128, "K": "tiles"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "K": 1, "M": 128, "N": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "K": 1, "M": 128, "N": 512}
    TENSORIZE_MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"N": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"tiles"})
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def __init__(self, tiles: int) -> None:
        """Configure a positive number of native partition contractions."""
        if tiles < 1:
            raise ValueError("interleaved matmul requires a positive tile count")
        super().__init__(tiles=tiles)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> BilinearReductionContract:
        """Describe the additive reduction over packed matrix-product tiles."""
        _ = kwargs
        return BilinearReductionContract("stationary", "moving", "dst", "K", ReduceCombinator("add", 0.0))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require complete on-chip packed matrices."""
        for slot in self.INPUT_OPERANDS:
            source = np.asarray(kwargs[slot])
            if _operand_role(kwargs[slot]) not in {None, "sbuf"}:
                raise TypeError("interleaved matmul requires SBUF inputs")
            if source.ndim != 2 or source.shape[0] != 128 or source.shape[1] % int(kwargs["tiles"]):
                raise ValueError("interleaved matmul requires complete packed contraction groups")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Sum each native contraction in FP32."""
        tiles = int(kwargs["tiles"])
        left = np.asarray(kwargs["stationary"], dtype=np.float32).reshape(128, tiles, -1)
        right = np.asarray(kwargs["moving"], dtype=np.float32).reshape(128, tiles, -1)
        result = np.zeros((left.shape[2], right.shape[2]), dtype=np.float32)
        for tile in range(tiles):
            result += left[:, tile, :].T @ right[:, tile, :]
        return result


__all__ = ["NKIInterleavedMatmul"]
