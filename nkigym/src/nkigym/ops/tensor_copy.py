"""SBUF/PSUM → SBUF copy op: maps to ``nisa.tensor_copy``."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKITensorCopy(NKIOp):
    """Copy ``src`` into ``dst`` element-wise."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf", "psum"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    INPLACE_OPERANDS: ClassVar[dict[str, frozenset[str]]] = {"dst": frozenset({"src"})}

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving tensor-copy contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF or PSUM source."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorCopy(src=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> Any:
        """Return a copy of ``src``."""
        return np.array(kwargs["src"])


def emit_matmul_drain(source: str, target: TorchValue, fp32: bool, body: list[str], imports: set[str]) -> TorchValue:
    """Drain one PSUM while optionally deferring its physical BF16 rounding."""
    class_name, operand = ("NKIFloat32Cast", "data") if fp32 else ("NKIBF16Cast", "data")
    result = replace(target, storage_dtype="float32" if fp32 else "bfloat16")
    imports.add(class_name)
    body.append(f"{result.name} = {class_name}()({operand}={source})")
    return result
