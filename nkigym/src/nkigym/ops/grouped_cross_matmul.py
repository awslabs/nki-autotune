"""Grouped query/key products through ``nisa.nc_matmul``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKIGroupedCrossMatmul(NKIOp):
    """Compute independent query/key tiles into packed PSUM storage."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("K", "G", "Q", "P"),
        "moving": ("K", "G", "T", "F"),
        "dst": ("G", "Q", "P", "T", "F"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "stationary": (("K",), ("G", "Q", "P")),
        "moving": (("K",), ("G", "T", "F")),
        "dst": (("G", "Q", "P"), ("T", "F")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "G": "groups",
        "Q": "queries",
        "T": "tiles",
        "P": "partitions",
        "F": "width",
    }
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "KGQTPF"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"K": 128, "G": 1, "Q": 1, "T": 1, "P": 128, "F": 512}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "queries", "tiles", "partitions", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_TILE_ALIGNMENT_BYTES: ClassVar[dict[str, int]] = {"dst": 256 * 1024}

    def __init__(self, groups: int, queries: int, tiles: int, partitions: int, width: int) -> None:
        """Configure the packed query/key geometry."""
        super().__init__(
            groups=groups, queries=queries, tiles=tiles, partitions=partitions, width=width, accumulate=False
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF operands."""
        for slot in ("stationary", "moving"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKIGroupedCrossMatmul({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return packed independent query/key products."""
        g, q, t, p, f = (int(kwargs[key]) for key in ("groups", "queries", "tiles", "partitions", "width"))
        stationary = np.asarray(kwargs["stationary"], dtype=np.float32).reshape(-1, g, q, p)
        moving = np.asarray(kwargs["moving"], dtype=np.float32).reshape(-1, g, t, f)
        output = np.einsum("kgqp,kgtf->gqptf", stationary, moving)
        return output.reshape(g * q * p, t * f)


def lower_grouped_context_attention(
    node: Any, values: Mapping[Any, object], body: list[str], imports: set[str]
) -> TorchValue:
    """Emit one grouped context-attention operation graph."""
    inputs = tuple(value for argument in node.args if isinstance((value := values[argument]), TorchValue))
    keys = ("groups", "queries", "tiles", "reduction", "partitions", "width", "output_width")
    g, qn, t, r, p, w, h = (int(node.kwargs[key]) for key in keys)
    if tuple(value.shape for value in inputs) != (
        (r, g * qn * p),
        (r, g * t * w),
        (128, t * 4 * g * h),
        (p, g * qn),
        (p, g * qn),
    ):
        raise ValueError("grouped context-attention input shapes are inconsistent")
    q_value, k_value, v_value, lower, upper = inputs
    base, config = f"sbuf_{node.name}", f"groups={g}, queries={qn}, tiles={t}, partitions={p}"
    imports.update(
        "NKIFoldedLoad NKIFoldedStore NKIGroupedActivationReduce NKIGroupedCrossMatmul NKIGroupedDMATranspose "
        "NKIGroupedQueryReduce NKIGroupedRangeSelectReduce NKIGroupedReciprocal "
        "NKIGroupedReductionMatmul NKIGroupedVectorScale".split()
    )
    loaded = []
    for value, tiles, suffix in zip(inputs, (qn, t, t * 4, qn, qn), ("q", "k", "v", "lower", "upper"), strict=True):
        if not value.is_hbm:
            raise ValueError("grouped context-attention inputs must remain in HBM until folded loading")
        target = TorchValue(f"{base}_{suffix}", value.shape)
        body.append(f"{target.name} = NKIFoldedLoad(groups={g}, tiles={tiles})(src={value.name})")
        loaded.append(target)
    q_value, k_value, v_value, lower, upper = loaded
    body.extend(
        (
            f"{base}_scores = NKIGroupedCrossMatmul({config}, width={w})"
            f"(stationary={q_value.name}, moving={k_value.name})",
            f"{base}_masked, {base}_max_parts = NKIGroupedRangeSelectReduce({config}, width={w})"
            f"(on_true_tile={base}_scores, bound0={lower.name}, bound1={upper.name})",
            f"{base}_negative = NKIGroupedQueryReduce(groups={g}, queries={qn}, tiles={t}, partitions={p}, "
            f'op="max", negate=True)(data={base}_max_parts)',
            f"{base}_exp, {base}_sum_parts = NKIGroupedActivationReduce({config}, width={w})"
            f"(data={base}_masked, bias={base}_negative)",
            f"{base}_total = NKIGroupedQueryReduce(groups={g}, queries={qn}, tiles={t}, partitions={p}, "
            f'op="add")(data={base}_sum_parts)',
            f"{base}_reciprocal = NKIGroupedReciprocal(groups={g}, queries={qn}, partitions={p})"
            f"(data={base}_total)",
            f"{base}_stationary = NKIGroupedDMATranspose({config}, subtiles=4)(src={base}_exp)",
            f"{base}_output_ps = NKIGroupedReductionMatmul({config}, subtiles=4, output_width={h})"
            f"(stationary={base}_stationary, moving={v_value.name})",
            f"{base}_output = NKIGroupedVectorScale(groups={g}, queries={qn}, partitions={p}, width={h})"
            f"(data={base}_output_ps, operand0={base}_reciprocal)",
            f"hbm_{node.name} = NKIFoldedStore(groups={g}, tiles={qn}, partitions={p})(src={base}_output)",
        )
    )
    return TorchValue(f"hbm_{node.name}", (g * qn * p, h), is_hbm=True)


__all__ = ["NKIGroupedCrossMatmul", "lower_grouped_context_attention"]
