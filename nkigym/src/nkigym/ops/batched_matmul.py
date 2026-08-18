"""Independent packed matrix products through ``nisa.nc_matmul``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import AxisRole, NKIOp, _operand_role
from nkigym.ops.inplace_match_replace8 import emit_inplace_selection


class NKIBatchedMatmul(NKIOp):
    """Apply one shared square matrix to independent packed groups."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("K", "M"),
        "moving": ("G", "K", "N"),
        "dst": ("G", "M", "N"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "stationary": (("K",), ("M",)),
        "moving": (("G", "K"), ("N",)),
        "dst": (("G", "M"), ("N",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "K": "partitions", "M": "partitions"}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"K": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "K": 1, "M": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "K": 128, "M": 128, "N": 512}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def __init__(self, groups: int, partitions: int) -> None:
        """Configure group and square-matrix extents."""
        super().__init__(groups=groups, partitions=partitions, accumulate=False)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF matrix operands."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in ("stationary", "moving")):
            raise TypeError("NKIBatchedMatmul expects SBUF operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply the shared matrix to every packed group."""
        groups, partitions = int(kwargs["groups"]), int(kwargs["partitions"])
        matrix = np.asarray(kwargs["stationary"], dtype=np.float32)
        moving = np.asarray(kwargs["moving"], dtype=np.float32).reshape(groups, partitions, -1)
        return np.concatenate([matrix.T @ moving[group] for group in range(groups)], axis=0)


def emit_rotational_topk_stages(
    working: TorchValue,
    indices: TorchValue,
    selected: TorchValue,
    positions: TorchValue,
    config: tuple[int, int, int, int, int],
    stem: str,
    body: list[str],
    imports: set[str],
) -> tuple[TorchValue, TorchValue]:
    """Emit selection, cyclic rotation, and insertion for every stage."""
    groups, rows, stages, stage_width, local_k = config
    partitions, selected_indices = rows * stages, positions
    imports.update(("NKIBatchedMatmul", "NKIInplaceTensorCopy"))
    for stage in range(stages):
        selected_indices = emit_inplace_selection(
            working,
            indices,
            selected,
            positions,
            groups,
            partitions,
            stage_width + stage * local_k,
            f"{stem}_{stage}",
            body,
            imports,
        )
        if stage < stages - 1:
            rotated_values = TorchValue(f"sbuf_{stem}_rotated_values_{stage}", selected.shape)
            rotated_indices = TorchValue(f"sbuf_{stem}_rotated_indices_{stage}", selected.shape)
            insertion = stage_width + stage * local_k
            body.extend(
                (
                    f"{rotated_values.name} = NKIBatchedMatmul(groups={groups}, partitions={partitions})"
                    f"(stationary=sbuf_{stem}_rotation, moving={selected.name})",
                    f"{rotated_indices.name} = NKIBatchedMatmul(groups={groups}, partitions={partitions})"
                    f"(stationary=sbuf_{stem}_rotation, moving={selected_indices.name})",
                    f"{working.name} = NKIInplaceTensorCopy(groups={groups}, partitions={partitions}, "
                    f"start={insertion}, width={local_k})(src={rotated_values.name}, dst={working.name})",
                    f"{indices.name} = NKIInplaceTensorCopy(groups={groups}, partitions={partitions}, "
                    f"start={insertion}, width={local_k})(src={rotated_indices.name}, dst={indices.name})",
                )
            )
    return selected, selected_indices


__all__ = ["NKIBatchedMatmul", "emit_rotational_topk_stages"]
