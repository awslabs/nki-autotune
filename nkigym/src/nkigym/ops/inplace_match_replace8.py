"""In-place replacement and index return through ``nisa.nc_match_replace8``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import AxisRole, NKIOp, _operand_role


class NKIInplaceMatchReplace8(NKIOp):
    """Replace eight values and write their original positions in place."""

    NAME: ClassVar[str] = "nc_match_replace8"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("G", "P", "F"),
        "vals": ("G", "P", "K"),
        "dst": ("G", "P", "F"),
        "dst_idx": ("G", "P", "O"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("G", "P"), (axis,)) for slot, axis in {"data": "F", "vals": "K", "dst": "F", "dst_idx": "O"}.items()
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "vals"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst", "dst_idx"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst_idx"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {
        "data": ((1, "source_start", "source_width"),),
        "dst": ((1, "source_start", "source_width"),),
        "vals": ((1, "value_start", "output_width"),),
        "dst_idx": ((1, "output_start", "output_width"),),
    }
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {"groups", "partitions", "source_start", "source_width", "value_start", "output_start", "output_width"}
    )
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 8, "K": 8, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None, "K": None, "O": None}

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF operands and aliased data destinations."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in ("data", "vals", "dst", "dst_idx")):
            raise TypeError("NKIInplaceMatchReplace8 expects SBUF operands")
        if kwargs["data"] is not kwargs["dst"]:
            raise ValueError("NKIInplaceMatchReplace8 requires data and dst to alias")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Replace selected values and write their original positions."""
        data, positions = kwargs["dst"], kwargs["dst_idx"]
        value_start, output_start, width = (
            int(kwargs[name]) for name in ("value_start", "output_start", "output_width")
        )
        active = int(kwargs["source_width"])
        for row, values in enumerate(np.asarray(kwargs["vals"])[:, value_start : value_start + width]):
            for column, value in enumerate(values, output_start):
                matches = np.flatnonzero(data[row, :active] == value)
                positions[row, column] = matches[0]
                data[row, matches[0]] = kwargs["imm"]
        return positions


def emit_inplace_selection(
    working: TorchValue,
    indices: TorchValue,
    selected: TorchValue,
    positions: TorchValue,
    groups: int,
    partitions: int,
    active_width: int,
    stem: str,
    body: list[str],
    imports: set[str],
) -> TorchValue:
    """Emit packed repeated top-eight selection and one final gather."""
    imports.update(("NKIGroupedGather", "NKIInplaceMatchReplace8", "NKIInplaceMax8"))
    for offset in range(0, selected.shape[1], 8):
        common = (
            f"groups={groups}, partitions={partitions}, source_start=0, source_width={active_width}, "
            f"output_start={offset}, output_width=8"
        )
        body.extend(
            (
                f"{selected.name} = NKIInplaceMax8({common})(src={working.name}, dst={selected.name})",
                f"{positions.name} = NKIInplaceMatchReplace8({common}, value_start={offset}, imm=float('-inf'))"
                f"(data={working.name}, vals={selected.name}, dst={working.name}, dst_idx={positions.name})",
            )
        )
    result = TorchValue(f"sbuf_{stem}_selected_indices", selected.shape)
    body.append(
        f"{result.name} = NKIGroupedGather(groups={groups}, partitions={partitions})"
        f"(data={indices.name}, indices={positions.name})"
    )
    return result


__all__ = ["NKIInplaceMatchReplace8", "emit_inplace_selection"]
