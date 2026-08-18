"""Packed independent matrix products through ``nisa.nc_matmul``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKIGroupedMatmul(NKIOp):
    """Compute group-wise products into chunked PSUM tiles."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("K", "G", "C", "T", "M"),
        "moving": ("K", "G", "N"),
        "dst": ("C", "G", "M", "T", "N"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "stationary": (("K",), ("G", "C", "T", "M")),
        "moving": (("K",), ("G", "N")),
        "dst": (("C", "G", "M"), ("T", "N")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "C": "chunks",
        "G": "groups",
        "T": "tiles",
        "M": "partition",
        "N": "queries",
    }
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "CGTN"}
    MIN_TILE_SIZE.update({"K": 128, "M": 128})
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"C": 1, "G": 1, "T": 1, "K": 128, "M": 128, "N": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"chunks", "groups", "tiles", "partition", "queries"})
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_TILE_ALIGNMENT_BYTES: ClassVar[dict[str, int]] = {"dst": 256 * 1024}

    def __init__(self, chunks: int, groups: int, tiles: int, partition: int, queries: int) -> None:
        """Configure the packed group and tile extents."""
        if min(chunks, groups, tiles, partition, queries) < 1:
            raise ValueError("grouped matmul extents must be positive")
        super().__init__(
            chunks=chunks, groups=groups, tiles=tiles, partition=partition, queries=queries, accumulate=False
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require both packed inputs in SBUF."""
        for slot in ("stationary", "moving"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKIGroupedMatmul({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return packed group-wise products for CPU validation."""
        c, g, t, m, n = (int(kwargs[key]) for key in ("chunks", "groups", "tiles", "partition", "queries"))
        stationary, moving = (np.asarray(kwargs[name], dtype=np.float32) for name in ("stationary", "moving"))
        k = stationary.shape[0]
        stationary = stationary.reshape(k, g, c, t, m)
        moving = moving.reshape(k, g, n)
        output = np.empty((c, g, m, t, n), dtype=np.float32)
        for chunk in range(c):
            for group in range(g):
                for tile in range(t):
                    output[chunk, group, :, tile, :] = stationary[:, group, chunk, tile, :].T @ moving[:, group]
        return output.reshape(c * g * m, t * n)


def _emit_grouped_attention_prefix(
    q: TorchValue,
    k: TorchValue,
    mask: TorchValue,
    dimensions: tuple[int, int, int, int, int],
    base: str,
    body: list[str],
) -> tuple[str, str]:
    """Emit grouped QK and one shared stable softmax."""
    chunks, groups, tiles, width, queries = dimensions
    config = f"chunks={chunks}, groups={groups}, tiles={tiles}"
    body.extend(
        (
            f"{base}_q = NKIBF16Cast()(data={q.name})",
            f"{base}_score_ps = NKIGroupedMatmul({config}, partition={width}, queries={queries})"
            f"(stationary={k.name}, moving={base}_q)",
            f"{base}_scores = NKIGroupedTensorCopy({config}, partition={width}, queries={queries})"
            f"(src={base}_score_ps)",
            f'{base}_masked = NKITensorTensor(op="add")(data1={base}_scores, data2={mask.name})',
            f"{base}_maximum_parts = NKITiledTensorReduce(chunks={chunks}, tiles={tiles}, groups={groups}, "
            f'queries={queries}, op="max")(data={base}_masked)',
            f"{base}_maximum_ps = NKITranspose()(data={base}_maximum_parts)",
            f'{base}_maximum = NKITensorReduce(op="max", axis=1)(data={base}_maximum_ps)',
            f"{base}_maximum_broadcast_ps = NKITransposeBroadcast(partitions={width})(data={base}_maximum)",
            f"{base}_maximum_broadcast = NKITensorCopy()(src={base}_maximum_broadcast_ps)",
            f"{base}_maximum_full = NKITileBroadcast(chunks={chunks}, tiles={tiles}, groups={groups}, "
            f"queries={queries})(data={base}_maximum_broadcast)",
            f'{base}_centered = NKITensorTensor(op="subtract")(data1={base}_masked, data2={base}_maximum_full)',
            f'{base}_exponential = NKIActivation(op="exp")(data={base}_centered)',
            f"{base}_one_iota = NKIIota(partitions={width}, width=1, pattern=[[0, 1]], "
            f"offset=1, channel_multiplier=0)()",
            f"{base}_ones = NKIBF16Cast()(data={base}_one_iota)",
            f"{base}_sums_ps = NKITiledSumMatmul(chunks={chunks}, tiles={tiles}, groups={groups}, "
            f"queries={queries})(stationary={base}_exponential, moving={base}_ones)",
            f'{base}_sums = NKITensorReduce(op="add", axis=1)(data={base}_sums_ps)',
            f'{base}_reciprocal = NKIActivation(op="reciprocal")(data={base}_sums)',
            f"{base}_reciprocal_ps = NKITransposeBroadcast(partitions={width})(data={base}_reciprocal)",
            f"{base}_reciprocal_broadcast = NKITensorCopy()(src={base}_reciprocal_ps)",
        )
    )
    return f"{base}_exponential", f"{base}_reciprocal_broadcast"


__all__ = ["NKIGroupedMatmul"]
