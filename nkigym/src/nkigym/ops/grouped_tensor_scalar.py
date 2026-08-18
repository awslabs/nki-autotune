"""Repeated tensor-scalar operations through ``nisa.tensor_scalar``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, _operand_role


class NKIGroupedTensorScalar(NKIOp):
    """Apply one partition-vector operation to independent packed groups."""

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("G", "P", "F"),
        "operand0": ("P", "O"),
        "dst": ("G", "P", "F"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "data": (("G", "P"), ("F",)),
        "operand0": (("P",), ("O",)),
        "dst": (("G", "P"), ("F",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "O": 1}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None, "O": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF inputs."""
        for slot in ("data", "operand0"):
            if (role := _operand_role(kwargs[slot])) is not None and role not in {"sbuf", "psum"}:
                raise TypeError(f"NKIGroupedTensorScalar({slot}=<role={role}>) expects on-chip data")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply the configured operation to every packed group."""
        groups, partitions = int(kwargs["groups"]), int(kwargs["partitions"])
        data = np.asarray(kwargs["data"]).reshape(groups, partitions, -1)
        vector = np.asarray(kwargs["operand0"]).reshape(1, partitions, 1)
        operations = {"add": np.add, "multiply": np.multiply}
        return operations[str(kwargs["op0"])](data, vector).reshape(groups * partitions, -1)


def prepare_rotational_topk(
    source: TorchValue,
    config: tuple[int, int, int, int, int],
    rotation: str,
    stem: str,
    body: list[str],
    imports: set[str],
) -> tuple[TorchValue, TorchValue, TorchValue, TorchValue]:
    """Emit packed working values, indices, rotation, and selection storage."""
    groups, rows, stages, stage_width, local_k = config
    partitions, width = rows * stages, stage_width + stages * local_k
    leading = groups * partitions
    working = TorchValue(f"sbuf_{stem}_working", (leading, width))
    free = TorchValue(f"sbuf_{stem}_free_indices", working.shape)
    offset_row = TorchValue(f"sbuf_{stem}_offset_row", (1, partitions))
    offset_ps = TorchValue(f"sbuf_{stem}_offset_ps", (partitions, 1))
    offsets = TorchValue(f"sbuf_{stem}_offsets", offset_ps.shape)
    indices = TorchValue(f"sbuf_{stem}_indices", working.shape)
    selected = TorchValue(f"sbuf_{stem}_selected_values", (leading, local_k))
    positions = TorchValue(f"sbuf_{stem}_selected_positions", selected.shape)
    imports.update(
        "NKIGroupedIota NKIGroupedLoad NKIGroupedTensorScalar NKIIndexIota NKIIota NKILoad "
        "NKITensorCopy NKITranspose".split()
    )
    body.extend(
        (
            f"{working.name} = NKIGroupedLoad(groups={groups}, rows={rows}, stages={stages})(src={source.name})",
            f"{free.name} = NKIGroupedIota(groups={groups}, partitions={partitions}, width={width}, "
            f"pattern=[[1, {width}]], channel_multiplier=0)()",
            f"{offset_row.name} = NKIIota(partitions=1, width={partitions}, "
            f"pattern=[[0, {rows}], [{stage_width}, {stages}]])()",
            f"{offset_ps.name} = NKITranspose()(data={offset_row.name})",
            f"{offsets.name} = NKITensorCopy()(src={offset_ps.name})",
            f"{indices.name} = NKIGroupedTensorScalar(groups={groups}, partitions={partitions}, op0='add')"
            f"(data={free.name}, operand0={offsets.name})",
            f"sbuf_{stem}_rotation = NKILoad()(src={rotation})",
            f"{selected.name} = NKIGroupedIota(groups={groups}, partitions={partitions}, width={local_k}, "
            f"pattern=[[0, {local_k}]], channel_multiplier=0)()",
            f"{positions.name} = NKIIndexIota(groups={groups}, partitions={partitions}, width={local_k}, "
            f"pattern=[[0, {local_k}]], channel_multiplier=0)()",
        )
    )
    return working, indices, selected, positions


__all__ = ["NKIGroupedTensorScalar", "prepare_rotational_topk"]
