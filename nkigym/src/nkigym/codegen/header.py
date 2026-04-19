"""Kernel header: imports, inlined gadget functions, signature, assertions, HBM output allocation."""

import inspect

from nkigym.codegen import gadgets
from nkigym.kernel_ir.dim_analysis import DimAnalysis

_INLINED_GADGETS = "\n\n".join(
    inspect.getsource(fn)
    for fn in (
        gadgets.load_block,
        gadgets.stage_block,
        gadgets.store_block,
        gadgets._is_single_tile,
        gadgets._stage_single,
        gadgets._stage_along_partition,
        gadgets._stage_along_free,
    )
)


def render_header(da: DimAnalysis) -> str:
    """Emit NKI kernel header from dimension analysis.

    Produces imports, inlined top-level gadget functions
    (``load_block``, ``stage_block``, ``store_block``) so the
    kernel file is self-contained and does not depend on the
    ``nkigym`` package on the executor, the ``@nki.jit``
    decorator, function signature, input shape assertions, and
    the output HBM allocation.

    Args:
        da: Complete dimension analysis result.

    Returns:
        NKI source code for the kernel header.
    """
    lines: list[str] = []

    lines.append("import nki")
    lines.append("import nki.isa as nisa")
    lines.append("import nki.language as nl")
    lines.append("import numpy as np")
    lines.append("from typing import Any")
    lines.append("")
    lines.append(_INLINED_GADGETS)
    lines.append("")
    lines.append("@nki.jit")
    params = ", ".join(da.param_names)
    lines.append(f"def {da.func_name}({params}):")

    for name in da.param_names:
        shape = da.tensors[name].shape
        lines.append(f"    assert {name}.shape == {shape}")

    ret = da.return_name
    ret_tensor = da.tensors[ret]
    lines.append(f"    {ret} = nl.ndarray({ret_tensor.shape}," f" dtype=nl.{ret_tensor.dtype}, buffer=nl.shared_hbm)")

    return "\n".join(lines)


def render_return(da: DimAnalysis) -> str:
    """Emit the return statement for the kernel.

    Args:
        da: Complete dimension analysis result.

    Returns:
        Indented return statement.
    """
    return f"    return {da.return_name}"
