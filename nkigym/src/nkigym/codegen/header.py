"""Kernel header: imports, inlined gadget functions, signature, assertions, HBM output allocation."""

import inspect

from nkigym.codegen import gadgets
from nkigym.kernel_ir import KernelIR

_INLINED_GADGETS = "\n\n".join(
    inspect.getsource(fn)
    for fn in (gadgets.load_block, gadgets.memset_block, gadgets.stage_block, gadgets.store_block, gadgets.matmul_block)
)


def render_header(ir: KernelIR) -> str:
    """Emit NKI kernel header from the kernel ir."""
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
    params = ", ".join(ir.param_names)
    lines.append(f"def {ir.func_name}({params}):")
    for name in ir.param_names:
        shape = ir.logical_tensors[name].shape
        lines.append(f"    assert {name}.shape == {shape}")
    ret = ir.return_name
    ret_tinfo = ir.logical_tensors[ret]
    lines.append(f"    {ret} = nl.ndarray({ret_tinfo.shape}, dtype=nl.{ret_tinfo.dtype}, buffer=nl.shared_hbm)")
    return "\n".join(lines)


def render_return(ir: KernelIR) -> str:
    """Emit the return statement for the kernel."""
    return f"    return {ir.return_name}"
