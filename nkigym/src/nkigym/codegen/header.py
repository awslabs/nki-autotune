"""Kernel header: imports, signature, assertions, HBM output allocation."""

from nkigym.dim_analysis.dim_analysis import DimAnalysis


def render_header(da: DimAnalysis) -> str:
    """Emit NKI kernel header from dimension analysis.

    Produces imports, @nki.jit decorator, function signature,
    input shape assertions, and output HBM allocation.

    Args:
        da: Complete dimension analysis result.

    Returns:
        NKI source code for the kernel header.
    """
    lines: list[str] = []

    lines.append("import nki")
    lines.append("import nki.isa as nisa")
    lines.append("import nki.language as nl")
    lines.append("")
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


def render_full(da: DimAnalysis) -> str:
    """Emit complete kernel header with return statement.

    Args:
        da: Complete dimension analysis result.

    Returns:
        NKI source code for imports, signature, assertions,
        HBM allocation, and return statement.
    """
    return render_header(da) + "\n" + render_return(da) + "\n"
