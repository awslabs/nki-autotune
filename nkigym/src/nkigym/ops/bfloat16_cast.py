"""Cast one local tile to bfloat16 with Vector Engine ``nisa.tensor_copy``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_values import TorchValue, emit_activation
from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role
from nkigym.ops.float32_cast import _matmul_only_uses, _vector_copy_parameters
from nkigym.ops.reciprocal import emit_activation_or_reciprocal


def emit_activation_with_storage(
    source: TorchValue, node: Node, operation: str, scale: float, body: list[str], imports: set[str]
) -> TorchValue:
    """Emit an activation with an optional reference-declared physical dtype."""
    dtype = node.meta.get("activation_storage_dtype")
    if dtype not in {None, "bfloat16"}:
        raise ValueError(f"Unsupported activation storage dtype {dtype!r}")
    name = f"sbuf_{node.name}"
    native = operation == "sqrt" and node.meta.get("arithmetic") == "native"
    activation = emit_activation if native else emit_activation_or_reciprocal
    target = activation(source, operation, f"{name}_fp32" if dtype else name, scale, body, imports)
    if dtype:
        imports.add("NKIBF16Cast")
        body.append(f"{name} = NKIBF16Cast()(data={target.name})")
        target = TorchValue(name, target.shape, target.transposed, storage_dtype=dtype)
    residency = node.meta.get("activation_residency", "auto")
    if residency not in {"auto", "sbuf", "shared_hbm"}:
        raise ValueError(f"Unsupported activation residency {residency!r}")
    spill = residency == "shared_hbm" or (
        residency == "auto" and operation == "gelu_apprx_tanh" and int(np.prod(target.shape)) > 1 << 20
    )
    if spill:
        imports.add("NKIStore")
        stored = TorchValue(
            f"hbm_{name.removeprefix('sbuf_')}", target.shape, target.transposed, True, target.storage_dtype
        )
        body.append(f"{stored.name} = NKIStore()(src={target.name})")
        target = stored
    return target


def cast_matmul_producer(source: TorchValue, node: Node, body: list[str], imports: set[str]) -> TorchValue:
    """Round an explicitly configured matmul-only producer before its layout views."""
    dtype = node.meta.get("matmul_input_dtype")
    if dtype not in {None, "bfloat16"}:
        raise ValueError(f"Unsupported matmul input dtype {dtype!r}")
    if dtype is None or source.storage_dtype == dtype or not _matmul_only_uses(node):
        return source
    name = f"{source.name}_bf16"
    imports.add("NKIBF16Cast")
    body.append(f"{name} = NKIBF16Cast()(data={source.name})")
    return TorchValue(name, source.shape, source.transposed, storage_dtype=dtype)


class NKIBF16Cast(NKIOp):
    """Copy one tile into a bfloat16 destination."""

    NAME: ClassVar[str] = "activation"
    ISA_OPERAND_NAMES: ClassVar[dict[str, str]] = {"data": "src"}
    native_parameters = staticmethod(_vector_copy_parameters)
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self) -> None:
        """Configure the native activation as an explicit bfloat16 copy."""
        super().__init__(op="copy")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseContract:
        """Return the value-preserving cast contract."""
        _ = kwargs
        return PointwiseContract(operator="copy", input_operands=("data",), output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source tile."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKIBF16Cast(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the value-preserving FP32 simulation result."""
        return np.asarray(kwargs["data"], dtype=np.float32)


__all__ = ["NKIBF16Cast"]
