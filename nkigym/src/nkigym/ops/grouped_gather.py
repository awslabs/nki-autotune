"""Independent within-partition gathers through ``nisa.nc_n_gather``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.inplace_match_replace8 import emit_inplace_selection


class NKIGroupedGather(NKIOp):
    """Gather free-axis values within each packed program group."""

    NAME: ClassVar[str] = "nc_n_gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("G", "P", "F"),
        "indices": ("G", "P", "N"),
        "dst": ("G", "P", "N"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("G", "P"), (axis,)) for slot, axis in {"data": "F", "indices": "N", "dst": "N"}.items()
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "indices"})
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"indices": "uint32"}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None, "N": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF operands."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in ("data", "indices")):
            raise TypeError("NKIGroupedGather expects SBUF operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Gather the requested free-axis positions."""
        data = np.asarray(kwargs["data"])
        indices = np.asarray(kwargs["indices"], dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= data.shape[1]):
            raise ValueError("NKIGroupedGather indices exceed the source width")
        return np.take_along_axis(data, indices, axis=1)


def emit_rotational_topk_outputs(
    selected: TorchValue,
    selected_indices: TorchValue,
    k: int,
    config: tuple[int, int, int, int, int],
    stem: str,
    body: list[str],
    imports: set[str],
) -> tuple[TorchSegments, TorchSegments]:
    """Flatten stage candidates, select the final top-k, and store outputs."""
    groups, rows, stages, _stage_width, local_k = config
    padded_k = stages * local_k
    hbm_values = TorchValue(f"hbm_{stem}_candidates", (groups * rows, padded_k), is_hbm=True)
    hbm_indices = TorchValue(f"hbm_{stem}_candidate_indices", hbm_values.shape, is_hbm=True)
    candidates = TorchValue(f"sbuf_{stem}_candidates", hbm_values.shape)
    candidate_indices = TorchValue(f"sbuf_{stem}_candidate_indices", hbm_values.shape)
    result_values = TorchValue(f"sbuf_{stem}_result_values", (groups * rows, k))
    result_positions = TorchValue(f"sbuf_{stem}_result_positions", result_values.shape)
    imports.update(("NKIGroupedIota", "NKIGroupedLoad", "NKIGroupedStore", "NKIIndexIota"))
    body.extend(
        (
            f"{hbm_values.name} = NKIGroupedStore(groups={groups}, rows={rows}, stages={stages})"
            f"(src={selected.name})",
            f"{hbm_indices.name} = NKIGroupedStore(groups={groups}, rows={rows}, stages={stages})"
            f"(src={selected_indices.name})",
            f"{candidates.name} = NKIGroupedLoad(groups={groups}, rows={rows}, stages=1)(src={hbm_values.name})",
            f"{candidate_indices.name} = NKIGroupedLoad(groups={groups}, rows={rows}, stages=1)"
            f"(src={hbm_indices.name})",
            f"{result_values.name} = NKIGroupedIota(groups={groups}, partitions={rows}, width={k}, "
            f"pattern=[[0, {k}]], channel_multiplier=0)()",
            f"{result_positions.name} = NKIIndexIota(groups={groups}, partitions={rows}, width={k}, "
            f"pattern=[[0, {k}]], channel_multiplier=0)()",
        )
    )
    result_indices = emit_inplace_selection(
        candidates,
        candidate_indices,
        result_values,
        result_positions,
        groups,
        rows,
        padded_k,
        f"{stem}_final",
        body,
        imports,
    )
    integer_indices = TorchValue(f"sbuf_{stem}_integer_indices", result_indices.shape)
    imports.add("NKIGroupedInt32Cast")
    body.append(
        f"{integer_indices.name} = NKIGroupedInt32Cast(groups={groups}, partitions={rows})"
        f"(data={result_indices.name})"
    )
    outputs = []
    for suffix, value in (("values", result_values), ("indices", integer_indices)):
        output = TorchValue(f"hbm_{stem}_{suffix}", value.shape, is_hbm=True)
        body.append(f"{output.name} = NKIGroupedStore(groups={groups}, rows={rows}, stages=1)(src={value.name})")
        outputs.append(TorchSegments((output,)))
    return outputs[0], outputs[1]


__all__ = ["NKIGroupedGather", "emit_rotational_topk_outputs"]
