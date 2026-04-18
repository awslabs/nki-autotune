"""AST-based kernel source analysis and hardware detection."""

import ast
import json
import logging
import subprocess

logger = logging.getLogger(__name__)


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
        if not output_shape and _is_hbm_ndarray(node) and isinstance(node, ast.Call) and node.args:
            output_shape = _extract_shape_tuple(node.args[0])
    if not func_name:
        raise ValueError("No @nki.jit decorated function found in source")
    if not output_shape:
        raise ValueError("No nl.ndarray(..., buffer=nl.shared_hbm) found in source")
    return func_name, output_shape


def _try_parse(source: str) -> ast.Module | None:
    """Try to parse source code, returning None on SyntaxError."""
    result: ast.Module | None = None
    try:
        result = ast.parse(source)
    except SyntaxError:
        pass
    return result


def _slice_size(node: ast.expr) -> int:
    """Extract the size of a slice dimension.

    Handles constant bounds (``0:128 → 128``) and the common
    loop-indexed form ``expr:expr + N`` that codegen emits for
    in-flight tile views (e.g. ``i_block_d0 * 128:i_block_d0 *
    128 + 128``). For the loop-indexed form the ``expr`` must
    match literally (``ast.unparse``) on both sides; ``N`` is
    then the size.
    """
    size: int | None = None
    if isinstance(node, ast.Slice) and node.upper is not None:
        lower = node.lower if node.lower is not None else ast.Constant(value=0)
        upper = node.upper
        if (
            isinstance(upper, ast.Constant)
            and isinstance(upper.value, int)
            and isinstance(lower, ast.Constant)
            and isinstance(lower.value, int)
        ):
            size = upper.value - lower.value
        elif (
            isinstance(upper, ast.BinOp)
            and isinstance(upper.op, ast.Add)
            and isinstance(upper.right, ast.Constant)
            and isinstance(upper.right.value, int)
            and ast.unparse(upper.left) == ast.unparse(lower)
        ):
            size = upper.right.value
    if size is None:
        raise ValueError(f"Cannot extract slice size from {ast.unparse(node)}")
    return size


def _range_dims(subscript: ast.Subscript) -> list[int]:
    """Extract sizes of all range (slice) dimensions from a subscript.

    For sbuf_a[0:128, 0, 0, 0, 0, 0:128], returns [128, 128].
    Skips single-index dimensions (block/tile selectors).
    """
    if not isinstance(subscript.slice, ast.Tuple):
        raise ValueError(f"Expected tuple subscript, got {type(subscript.slice).__name__}")
    sizes: list[int] = []
    for elt in subscript.slice.elts:
        if isinstance(elt, ast.Slice):
            sizes.append(_slice_size(elt))
    return sizes


def _enclosing_loop_product(node: ast.AST, parents: dict[int, ast.AST]) -> int:
    """Walk up from node through parents, multiplying range(N) trip counts."""
    product = 1
    current = node
    while id(current) in parents:
        parent = parents[id(current)]
        if isinstance(parent, ast.For) and isinstance(parent.iter, ast.Call):
            call_str = ast.unparse(parent.iter.func) if isinstance(parent.iter.func, ast.Attribute) else ""
            if not call_str:
                call_str = ast.unparse(parent.iter.func) if isinstance(parent.iter.func, ast.Name) else ""
            if call_str in ("range",) and parent.iter.args:
                arg = parent.iter.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    product *= arg.value
        current = parent
    return product


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Build a mapping from child node id to parent node."""
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def detect_mac_count(source: str) -> int:
    """Auto-detect total MAC count from nisa.nc_matmul calls in kernel source.

    For each nc_matmul call:
      - Extracts stationary and moving operand slice dimensions
      - Per-tile MAC = partition × stationary_free × moving_free
      - Multiplies by enclosing loop trip counts
      - Sums across all nc_matmul calls

    Returns:
        Total MAC count, or 0 if no nc_matmul found or parsing fails.
    """
    tree = _try_parse(source)
    total_mac = 0
    if tree is None:
        logger.warning("Failed to parse kernel source for MAC detection")
    else:
        parents = _build_parent_map(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "nisa.nc_matmul"):
                continue
            operands = _nc_matmul_operands(node)
            if operands is None:
                continue
            stat_val, mov_val = operands
            try:
                stat_dims = _range_dims(stat_val)
                mov_dims = _range_dims(mov_val)
                if len(stat_dims) < 2 or len(mov_dims) < 2:
                    continue
                partition = stat_dims[0]
                stat_free = stat_dims[-1]
                mov_free = mov_dims[-1]
                tile_mac = partition * stat_free * mov_free
                loop_mult = _enclosing_loop_product(node, parents)
                total_mac += tile_mac * loop_mult
            except (ValueError, IndexError):
                continue
    return total_mac


def _nc_matmul_operands(node: ast.Call) -> tuple[ast.Subscript, ast.Subscript] | None:
    """Return ``(stationary, moving)`` operand subscripts from an ``nc_matmul`` call.

    Accepts either kwargs (``stationary=..., moving=...``) or
    the positional ``nc_matmul(dst, stationary, moving)`` form.
    Returns ``None`` when operands aren't subscripts (e.g. bare
    names referring to already-sliced buffers).
    """
    stat_kw = [kw for kw in node.keywords if kw.arg == "stationary"]
    mov_kw = [kw for kw in node.keywords if kw.arg == "moving"]
    pair: tuple[ast.expr, ast.expr] | None = None
    if stat_kw and mov_kw:
        pair = (stat_kw[0].value, mov_kw[0].value)
    elif len(node.args) >= 3:
        pair = (node.args[1], node.args[2])
    result: tuple[ast.Subscript, ast.Subscript] | None = None
    if pair is not None and isinstance(pair[0], ast.Subscript) and isinstance(pair[1], ast.Subscript):
        result = (pair[0], pair[1])
    return result


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
        raise RuntimeError(f"neuron-ls failed: {result.stderr}")
    devices = json.loads(result.stdout)
    total_cores = sum(d["nc_count"] for d in devices)
    if total_cores == 0:
        raise RuntimeError("neuron-ls reports 0 Neuron cores")
    return total_cores
