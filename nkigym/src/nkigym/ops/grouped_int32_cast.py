"""Grouped int32 conversion through ``nisa.activation``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role
from nkigym.ops.batched_matmul import emit_rotational_topk_stages
from nkigym.ops.grouped_tensor_scalar import prepare_rotational_topk


def emit_rotational_selection(
    source: TorchValue,
    count: int,
    config: tuple[int, int, int, int, int],
    stem: str,
    body: list[str],
    imports: set[str],
    specs: dict[str, tuple[tuple[int, ...], str]],
) -> tuple[TorchValue, TorchValue]:
    """Emit an unordered exact prefix using existing rotational primitives."""
    groups, rows, stages, _width, _local = config
    rotation = f"rotation_topk_{stages}"
    specs[rotation] = ((rows * stages, rows * stages), "float32")
    state = prepare_rotational_topk(source, config, rotation, stem, body, imports)
    selected, positions = emit_rotational_topk_stages(*state, config, stem, body, imports)
    imports.update(("NKIGroupedInt32Cast", "NKIGroupedStore", "NKILoad"))
    integer = f"sbuf_{stem}_global_indices"
    body.append(f"{integer} = NKIGroupedInt32Cast(groups={groups}, partitions={rows * stages})(data={positions.name})")
    outputs = []
    for suffix, value, dtype in (("values", selected.name, "float32"), ("indices", integer, "int32")):
        stored, loaded = f"hbm_{stem}_{suffix}", f"sbuf_{stem}_flat_{suffix}"
        body.append(f"{stored} = NKIGroupedStore(groups={groups}, rows={rows}, stages={stages})(src={value})")
        body.append(f"{loaded} = NKILoad()(src={stored})")
        outputs.append(TorchValue(loaded, (groups * rows, count), storage_dtype=dtype))
    return outputs[0], outputs[1]


class NKIGroupedInt32Cast(NKIOp):
    """Copy packed groups into an int32 destination."""

    NAME: ClassVar[str] = "activation"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("G", "P", "F"), "dst": ("G", "P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("G", "P"), ("F",)) for slot in ("data", "dst")
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "int32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int) -> None:
        """Configure packed groups and native copy conversion."""
        super().__init__(groups=groups, partitions=partitions, op="copy")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the value-preserving cast contract."""
        _ = kwargs
        return PointwiseContract(operator="copy", input_operands=("data",), output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source tile."""
        if _operand_role(kwargs["data"]) not in {None, "sbuf", "psum"}:
            raise TypeError("NKIGroupedInt32Cast.data expects SBUF or PSUM")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return packed values represented as int32."""
        return np.asarray(kwargs["data"], dtype=np.int32)


__all__ = ["NKIGroupedInt32Cast"]
