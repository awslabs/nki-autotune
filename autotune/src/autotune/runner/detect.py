"""AST-based kernel source analysis and hardware detection."""

import ast
import json
import subprocess


def _is_hbm_ndarray(node: ast.AST) -> bool:
    """Check if an AST node is an nl.ndarray(..., buffer=nl.shared_hbm) call."""
    return (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "nl.ndarray"
        and any(kw.arg == "buffer" and ast.unparse(kw.value) == "nl.shared_hbm" for kw in node.keywords)
    )


def _extract_shape_tuple(shape_node: ast.expr) -> tuple[int, ...]:
    """Extract integer dimensions from an AST Tuple node.

    Raises:
        ValueError: If a dimension is not a constant integer.
    """
    if not isinstance(shape_node, ast.Tuple):
        raise ValueError(f"Expected Tuple shape, got {type(shape_node).__name__}")
    dims: list[int] = []
    for elt in shape_node.elts:
        if not (isinstance(elt, ast.Constant) and isinstance(elt.value, int)):
            raise ValueError(f"Non-constant dimension in output shape: {ast.unparse(elt)}")
        dims.append(elt.value)
    return tuple(dims)


def _find_jit_name(node: ast.AST) -> str:
    """Return the function name if node is an @nki.jit function, else empty."""
    name = ""
    if isinstance(node, ast.FunctionDef):
        jit_decs = [d for d in node.decorator_list if ast.unparse(d) in ("nki.jit", "nki.jit()")]
        if jit_decs:
            name = node.name
    return name


def detect_kernel_info(source: str) -> tuple[str, tuple[int, ...]]:
    """Detect function name and output shape from kernel source in one parse.

    Raises:
        ValueError: If no @nki.jit function or HBM output tensor is found.
    """
    tree = ast.parse(source)
    func_name = ""
    output_shape: tuple[int, ...] = ()
    for node in ast.walk(tree):
        if not func_name:
            func_name = _find_jit_name(node)
        if not output_shape and _is_hbm_ndarray(node) and node.args:
            output_shape = _extract_shape_tuple(node.args[0])
    if not func_name:
        raise ValueError("No @nki.jit decorated function found in source")
    if not output_shape:
        raise ValueError("No nl.ndarray(..., buffer=nl.shared_hbm) found in source")
    return func_name, output_shape


def detect_func_name(source: str) -> str:
    """Detect the @nki.jit decorated function name from kernel source.

    Raises:
        ValueError: If no @nki.jit decorated function is found.
    """
    return detect_kernel_info(source)[0]


def detect_output_spec(source: str) -> tuple[int, ...]:
    """Detect the output tensor shape from kernel source.

    Raises:
        ValueError: If no HBM output tensor is found.
    """
    return detect_kernel_info(source)[1]


def detect_neuron_cores() -> int:
    """Detect available Neuron cores via neuron-ls.

    Raises:
        RuntimeError: If neuron-ls fails or reports 0 cores.
    """
    result = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RuntimeError(f"neuron-ls failed: {result.stderr[:500]}")
    devices = json.loads(result.stdout)
    total_cores = sum(d["nc_count"] for d in devices)
    if total_cores == 0:
        raise RuntimeError("neuron-ls reports 0 Neuron cores")
    return total_cores
