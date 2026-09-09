"""Drain grouped count-row PSUM slices and emit histogram metadata."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKIGroupedCountsCopy(NKIOp):
    """Copy each grouped count-row slice from PSUM into SBUF."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {slot: ("O", "G", "P") for slot in ("src", "dst")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("O",), ("G", "P")) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "O": 1}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 128, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "O": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, groups: int, partitions: int) -> None:
        """Configure the grouped count-row extents."""
        super().__init__(groups=groups, partitions=partitions)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving grouped drain contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a PSUM or SBUF source."""
        if (role := _operand_role(kwargs["src"])) is not None and role not in {"psum", "sbuf"}:
            raise TypeError(f"NKIGroupedCountsCopy(src=<role={role}>) expects PSUM or SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one value-preserving count-row copy."""
        return np.array(kwargs["src"], copy=True)


def emit_bincount_source(source: TorchValue, stem: str, body: list[str], imports: set[str]) -> TorchValue:
    """Load one metadata source directly into fp32 SBUF."""
    if not source.is_hbm:
        raise ValueError("grouped bincount source must remain in HBM")
    target = TorchValue(f"sbuf_{stem}_data", source.shape)
    imports.add("NKIFloat32Load")
    body.append(f"{target.name} = NKIFloat32Load()(src={source.name})")
    return target


def emit_grouped_bincount(
    source: TorchValue, stem: str, groups: int, experts: int, body: list[str], imports: set[str]
) -> tuple[TorchSegments, TorchSegments, TorchSegments]:
    """Emit grouped expert counts and derive metadata rows from one scan."""
    partitions = 128
    if groups % partitions or experts != groups:
        raise ValueError("wide grouped bincount requires one expert per destination and 128-way tiles")
    group_tiles, base = groups // partitions, f"sbuf_{stem}"
    config = f"groups={group_tiles}, partitions={partitions}"
    lower = TorchValue(f"{base}_lower", (groups, 1))
    partial = TorchValue(f"{base}_count_parts", (groups, 1))
    transposed = TorchValue(f"psum_{stem}_counts", (1, groups))
    counts = TorchValue(f"{base}_counts", transposed.shape)
    cumulative = TorchValue(f"{base}_cumulative", counts.shape)
    displacements, zeros = TorchValue(f"{base}_displacements", counts.shape), TorchValue(f"{base}_zeros", counts.shape)
    imports.update(
        "NKIActivation NKIGroupedCountsCopy NKIGroupedCountsTranspose NKIGroupedIota "
        "NKIGroupedTensorScalarReduce NKIInt32Cast NKIStore NKITensorScalarCumulative NKITensorTensor".split()
    )
    body.extend(
        (
            f"{lower.name} = NKIGroupedIota({config}, width=1, pattern=[[0, 1]], " "offset=0, channel_multiplier=1)()",
            f'{partial.name} = NKIGroupedTensorScalarReduce({config}, op0="equal", reduce_op="add")'
            f"(data={source.name}, operand0={lower.name})",
            f"{transposed.name} = NKIGroupedCountsTranspose({config})(data={partial.name})",
            f"{counts.name} = NKIGroupedCountsCopy({config})(src={transposed.name})",
            f'{cumulative.name} = NKITensorScalarCumulative(op0="add", op1="add", imm0=0.0)(src={counts.name})',
            f'{displacements.name} = NKITensorTensor(op="subtract")' f"(data1={cumulative.name}, data2={counts.name})",
            f'{zeros.name} = NKIActivation(op="copy", scale=0.0)(data={counts.name})',
        )
    )
    outputs: list[TorchValue] = []
    for suffix, value in (("counts", counts), ("displacements", displacements), ("zeros", zeros)):
        converted, stored = TorchValue(f"{value.name}_int32", value.shape), TorchValue(
            f"hbm_{stem}_{suffix}", value.shape, is_hbm=True
        )
        body.extend(
            (
                f"{converted.name} = NKIInt32Cast()(data={value.name})",
                f"{stored.name} = NKIStore()(src={converted.name})",
            )
        )
        outputs.append(stored)
    return (TorchSegments((outputs[0],)), TorchSegments((outputs[1],)), TorchSegments((outputs[2],)))


__all__ = ["NKIGroupedCountsCopy", "emit_bincount_source", "emit_grouped_bincount"]
