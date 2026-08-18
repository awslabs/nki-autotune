"""DMA-engine ``nisa.dma_transpose`` operation.

Same math as :class:`NKITranspose` (which uses Tensor Engine
``nisa.nc_transpose``), but runs on the DMA engine so it doesn't
contend with matmul for TE cycles. Useful when the matmul is TE-bound
and an explicit DMA transpose is cheaper than a round-trip through
PSUM. The ``src`` input may be an HBM parameter or an SBUF buffer.
"""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import BatchedPermutationContract, NKIOp, PermutationContract, _operand_role


def emit_oriented_value(
    source: TorchValue, target: TorchValue, intermediate: str, body: list[str], imports: set[str]
) -> None:
    """Emit one supported physical transpose path."""
    if source.storage_dtype is None or not source.storage_dtype.startswith("float8"):
        body.append(f"{target.name} = NKIDMATranspose()(src={source.name})")
        imports.add("NKIDMATranspose")
        return
    loaded = f"{target.name}_loaded"
    if source.is_hbm:
        body.append(f"{loaded} = NKILoad()(src={source.name})")
        imports.add("NKILoad")
    source_name = loaded if source.is_hbm else source.name
    casted = f"{target.name}_float32"
    body.extend(
        (
            f"{casted} = NKIFloat32Cast()(data={source_name})",
            f"{intermediate} = NKITranspose()(data={casted})",
            f"{target.name} = NKIFloat8Cast()(data={intermediate})",
        )
    )
    imports.update(("NKIFloat32Cast", "NKITranspose", "NKIFloat8Cast"))


class NKIDMATranspose(NKIOp):
    """DMA-engine transpose ``src(P, F) -> dst(F, P)``."""

    NAME: ClassVar[str] = "dma_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"src": ("P", "F"), "dst": ("F", "P")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"shared_hbm", "sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {
        "src": frozenset({"bfloat16", "float16", "float32", "int32", "tfloat32", "uint32"})
    }
    """Both abstract axes become a partition axis on one side of the
    transpose, so one instruction covers at most 128 elements of each.
    Larger free dimensions are represented by outer loops."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": 128}
    HBM_SOURCE_MAX_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 512, "F": 128}
    """Location-specific limits for a direct HBM-to-SBUF transpose.

    HBM input does not occupy the source partition axis, so the logical P
    extent becomes a packed SBUF free axis and may cover up to 512 elements.
    The logical F extent still becomes the destination partition axis and
    remains capped at 128.
    """
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_TILE_ALIGNMENT_BYTES: ClassVar[dict[str, int]] = {"dst": 32}

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Return the two-axis transpose contract."""
        _ = kwargs
        return PermutationContract(
            input_operand="src",
            output_operand="dst",
            permutation=(1, 0),
            batching=BatchedPermutationContract(permutation=(3, 1, 2, 0), input_axes=(0, 3), batch_axis=2),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an HBM parameter or SBUF source."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"param", "sbuf"}:
            raise TypeError(f"NKIDMATranspose(src=<role={role}>) expects param or sbuf")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return ``src.T``."""
        return np.array(kwargs["src"]).T
