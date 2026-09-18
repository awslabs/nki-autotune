"""Cast one local tile to float32 with Vector Engine ``nisa.tensor_copy``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import torch
from torch.fx import Node

from nkigym.codegen.torch_values import TorchSegments, TorchValue, emit_cast
from nkigym.ops.base import NKIOp, PointwiseContract, _operand_role


def _matmul_only_uses(node: Node) -> bool:
    """Recognize value-preserving views ending only in FP32-accumulating matmuls."""
    if not node.users:
        return False
    for user in node.users:
        name = str(getattr(user.target, "__name__", user.target)).removeprefix("wrapped_")
        if name == "matmul":
            continue
        if user.target is getattr and user.args[1] in {"dtype", "shape"}:
            continue
        view = name in {"transpose", "permute", "reshape", "view", "view_as", "detach", "unsqueeze"}
        view |= user.target is getattr and user.args[1] == "T"
        view |= name == "as_tensor" and len(user.args) == 1 and user.kwargs.get("dtype") is None
        if not view or not _matmul_only_uses(user):
            return False
    return True


def emit_torch_cast(
    source: TorchValue | TorchSegments, node: Node, body: list[str], imports: set[str]
) -> TorchValue | TorchSegments:
    """Preserve explicit FP32 arithmetic while folding redundant low-precision matmul casts."""
    name = str(getattr(node.target, "__name__", node.target)).removeprefix("wrapped_")
    dtype = node.kwargs.get("dtype", node.args[1] if len(node.args) > 1 else None)
    metadata = node.meta.get("tensor_meta", node.meta.get("example_value"))
    requested = name == "float" or dtype is torch.float32 or isinstance(dtype, str) and dtype == "float32"
    requested |= name == "to" and getattr(metadata, "dtype", None) == torch.float32
    values = source.values if isinstance(source, TorchSegments) else (source,)
    retained_fp8 = node.meta.get("matmul_input_dtype") == "bfloat16" and all(
        value.storage_dtype in {"float8_e4m3", "float8_e5m2"} for value in values
    )
    if (
        not requested
        or (retained_fp8 or all(value.storage_dtype in {"bfloat16", "float16", "float32"} for value in values))
        and _matmul_only_uses(node)
    ):
        return source
    converted = []
    for index, value in enumerate(values):
        target = f"sbuf_{node.name}" + (f"_{index}" if isinstance(source, TorchSegments) else "")
        if target == value.name:
            target = f"{target}_cast"
        if value.storage_dtype == "float32":
            converted.append(value)
        elif value.is_hbm:
            imports.add("NKIFloat32Load")
            body.append(f"{target} = NKIFloat32Load()(src={value.name})")
            converted.append(TorchValue(target, value.shape, value.transposed, storage_dtype="float32"))
        else:
            converted.append(emit_cast(value, "NKIFloat32Cast", target, body, imports))
    return TorchSegments(tuple(converted), source.axis) if isinstance(source, TorchSegments) else converted[0]


def _vector_copy_parameters(kwargs: dict[str, Any], operands: frozenset[str]) -> tuple[str, dict[str, Any]]:
    """Lower an explicit float dtype conversion to one Vector Engine copy."""
    if kwargs.get("op") != "copy" or operands != {"data", "dst"}:
        raise ValueError("dtype-copy lowering requires one data operand and an identity activation")
    return "tensor_copy", {key: value for key, value in kwargs.items() if key != "op"} | {"engine": "vector"}


class NKIFloat32Cast(NKIOp):
    """Copy one tile into a float32 destination."""

    NAME: ClassVar[str] = "activation"
    ISA_OPERAND_NAMES: ClassVar[dict[str, str]] = {"data": "src"}
    native_parameters = staticmethod(_vector_copy_parameters)
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self) -> None:
        """Configure the native activation as an explicit float32 copy."""
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
            raise TypeError(f"NKIFloat32Cast(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the source values represented as float32."""
        return np.asarray(kwargs["data"], dtype=np.float32)


__all__ = ["NKIFloat32Cast"]
