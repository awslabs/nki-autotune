"""AST-based codegen: callable -> NKIKernel in one step.

Parses the user function AST and builds NKI-level blocks with
concrete tile slices.
"""

import ast
from collections.abc import Callable

import numpy as np

from nkigym.codegen.analysis import (
    _Analysis,
    _OpCall,
    analyze_dims,
    compute_slices,
    iter_reduction_positions,
    iter_tile_positions,
)
from nkigym.codegen.types import NKIBlock, NKIKernel
from nkigym.ir.tensor import TensorRef
from nkigym.ops.activation import NKIActivation
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_copy import NKIDmaCopy
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.utils.source import callable_to_source

_OUTPUT_NAME = "output"

_OP_REGISTRY: dict[str, type[NKIOp]] = {"nc_matmul": NKIMatmul, "activation": NKIActivation}


def codegen(func: Callable, kwargs: dict[str, np.ndarray]) -> NKIKernel:
    """Generate an NKIKernel from a user function and concrete inputs.

    Args:
        func: User function using ``nkigym.<op>(...)`` calls.
        kwargs: Input arrays keyed by parameter name.

    Returns:
        Complete NKIKernel with unrolled parallel blocks.
    """
    source = callable_to_source(func)
    func_def = _find_func_def(source)
    params = tuple(arg.arg for arg in func_def.args.args)
    op_calls = _parse_body(func_def)
    input_shapes = tuple(kwargs[p].shape for p in params)
    dtype_name = next(iter(kwargs.values())).dtype.name
    analysis = analyze_dims(op_calls, params, input_shapes)
    blocks = _build_all_blocks(op_calls, analysis, params, dtype_name)
    output_shape = analysis.var_shapes[analysis.return_var]
    return NKIKernel(
        name=func_def.name,
        params=params,
        input_shapes=input_shapes,
        dtype=f"nl.{dtype_name}",
        output_shape=output_shape,
        blocks=tuple(blocks),
    )


def _find_func_def(source: str) -> ast.FunctionDef:
    """Find the first FunctionDef in parsed source.

    Args:
        source: Python source code string.

    Returns:
        The AST FunctionDef node.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("Expected a function definition")


def _is_nkigym_call(call: ast.Call) -> bool:
    """Check if a call is to ``nkigym.<op>``.

    Args:
        call: AST Call node.

    Returns:
        True if the call targets ``nkigym.<attr>``.
    """
    func = call.func
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "nkigym"


def _eval_expr(node: ast.expr) -> object:
    """Evaluate an AST expression to a Python object.

    Resolves ``np.X`` attribute accesses and literal constants.

    Args:
        node: AST expression node.

    Returns:
        The resolved Python object.
    """
    result = None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "np":
            result = getattr(np, node.attr)
    elif isinstance(node, ast.Constant):
        result = node.value
    if result is None:
        raise ValueError(f"Unsupported kwarg expression: {ast.dump(node)}")
    return result


def _arg_name(node: ast.expr) -> str:
    """Extract variable name from an AST Name node.

    Args:
        node: AST expression node.

    Returns:
        Variable name string.
    """
    if not isinstance(node, ast.Name):
        raise ValueError(f"Expected a variable name, got {ast.dump(node)}")
    return node.id


def _parse_body(func_def: ast.FunctionDef) -> list[_OpCall]:
    """Parse function body into a list of _OpCall.

    Args:
        func_def: The parsed AST FunctionDef node.

    Returns:
        List of parsed operation calls.
    """
    op_calls: list[_OpCall] = []
    counter: list[int] = [0]
    for node in func_def.body:
        if not _try_parse_node(node, op_calls, counter):
            raise ValueError(f"Unsupported statement: {ast.dump(node)}")
    return op_calls


def _try_parse_node(node: ast.stmt, op_calls: list[_OpCall], counter: list[int]) -> bool:
    """Try to parse a single AST statement node.

    Args:
        node: AST statement node.
        op_calls: Accumulator for parsed op calls.
        counter: Mutable counter for intermediate names.

    Returns:
        True if the node was successfully parsed.
    """
    result = False
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        result = True
    elif isinstance(node, ast.Return):
        result = _try_parse_return(node, op_calls, counter)
    elif isinstance(node, ast.Assign) and len(node.targets) == 1:
        result = _try_parse_assign(node, op_calls, counter)
    return result


def _try_parse_assign(node: ast.Assign, op_calls: list[_OpCall], counter: list[int]) -> bool:
    """Try to parse an assignment statement as an nkigym op call.

    Args:
        node: AST Assign node.
        op_calls: Accumulator for parsed op calls.
        counter: Mutable counter for intermediate names.

    Returns:
        True if the assignment was a recognized nkigym call.
    """
    target = node.targets[0]
    result = False
    if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
        if _is_nkigym_call(node.value):
            _flatten_call(node.value, target.id, op_calls, counter)
            result = True
    return result


def _try_parse_return(node: ast.Return, op_calls: list[_OpCall], counter: list[int]) -> bool:
    """Try to parse a return statement.

    Args:
        node: AST Return node.
        op_calls: Accumulator for parsed op calls.
        counter: Mutable counter for intermediate names.

    Returns:
        True if the return was successfully parsed.
    """
    result = False
    if isinstance(node.value, ast.Name):
        result = True
    elif isinstance(node.value, ast.Call) and _is_nkigym_call(node.value):
        _flatten_call(node.value, "_return", op_calls, counter)
        result = True
    return result


def _flatten_call(call: ast.Call, output: str, op_calls: list[_OpCall], counter: list[int]) -> None:
    """Flatten a nkigym call (possibly nested) into _OpCall entries.

    Args:
        call: AST Call node for a ``nkigym.<op>(...)`` call.
        output: Variable name to assign the result to.
        op_calls: Accumulator list for emitted op calls.
        counter: Mutable counter for intermediate variable names.
    """
    assert isinstance(call.func, ast.Attribute)
    op_name = call.func.attr
    if op_name not in _OP_REGISTRY:
        raise ValueError(f"Unknown op: {op_name!r}")
    stmt_type = _OP_REGISTRY[op_name]
    resolved_args = _resolve_call_args(call, op_calls, counter)
    config_kwargs: list[tuple[str, object]] = []
    for kw in call.keywords:
        assert kw.arg is not None
        config_kwargs.append((kw.arg, _eval_expr(kw.value)))
    op_calls.append(
        _OpCall(
            stmt_type=stmt_type, input_vars=tuple(resolved_args), config_kwargs=tuple(config_kwargs), output_var=output
        )
    )


def _resolve_call_args(call: ast.Call, op_calls: list[_OpCall], counter: list[int]) -> list[str]:
    """Resolve positional arguments of a nkigym call.

    Args:
        call: AST Call node.
        op_calls: Accumulator for nested op calls.
        counter: Mutable counter for intermediate names.

    Returns:
        List of resolved variable names.
    """
    resolved: list[str] = []
    for arg in call.args:
        if isinstance(arg, ast.Call) and _is_nkigym_call(arg):
            tmp_name = f"_nested_{counter[0]}"
            counter[0] += 1
            _flatten_call(arg, tmp_name, op_calls, counter)
            resolved.append(tmp_name)
        else:
            resolved.append(_arg_name(arg))
    return resolved


class _NameGen:
    """Sequential tensor_N name generator."""

    def __init__(self) -> None:
        """Initialize counter at 0."""
        self._counter: int = 0

    def next(self) -> str:
        """Return the next tensor_N name.

        Returns:
            Sequential tensor variable name.
        """
        name = f"tensor_{self._counter}"
        self._counter += 1
        return name


def _full_slices(shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Build full-range slices from a shape.

    Args:
        shape: Tensor shape tuple.

    Returns:
        Per-axis (0, size) bounds.
    """
    return tuple((0, s) for s in shape)


def _ref(name: str, shape: tuple[int, ...], slices: tuple[tuple[int, int], ...]) -> TensorRef:
    """Create a TensorRef.

    Args:
        name: Variable name.
        shape: Tensor shape.
        slices: Per-axis (start, stop) bounds.

    Returns:
        New TensorRef.
    """
    return TensorRef(name, shape, slices)


def _emit_load(
    stmts: list[NKIOp], param: str, slices: tuple[tuple[int, int], ...], dtype_str: str, name_gen: _NameGen
) -> TensorRef:
    """Emit SBUF alloc + DMA load for one operand tile.

    Args:
        stmts: Mutable statement accumulator.
        param: HBM parameter name to load from.
        slices: Source slice bounds in HBM.
        dtype_str: NKI dtype string.
        name_gen: Name generator.

    Returns:
        TensorRef for the loaded SBUF tile.
    """
    tile_shape = tuple(e - s for s, e in slices)
    sbuf_name = name_gen.next()
    sbuf_slices = _full_slices(tile_shape)
    stmts.append(NKIAlloc(dst=sbuf_name, shape=tile_shape, dtype=dtype_str, buffer="sbuf"))
    stmts.append(NKIDmaCopy(dst=_ref(sbuf_name, tile_shape, sbuf_slices), src=_ref(param, tile_shape, slices)))
    return _ref(sbuf_name, tile_shape, sbuf_slices)


def _emit_matmul_group(
    stmts: list[NKIOp],
    op_call: _OpCall,
    analysis: _Analysis,
    position: dict[str, int],
    psum_ref: TensorRef,
    dtype_str: str,
    name_gen: _NameGen,
) -> None:
    """Emit loads + matmul for one reduction step.

    Args:
        stmts: Mutable statement accumulator.
        op_call: The matmul op call.
        analysis: Dimension analysis.
        position: Combined parallel + reduction position.
        psum_ref: PSUM accumulator reference.
        dtype_str: NKI dtype string.
        name_gen: Name generator.
    """
    stat_var = op_call.input_vars[0]
    mov_var = op_call.input_vars[1]
    stat_slices = compute_slices(stat_var, position, analysis)
    mov_slices = compute_slices(mov_var, position, analysis)
    stat_ref = _emit_load(stmts, stat_var, stat_slices, dtype_str, name_gen)
    mov_ref = _emit_load(stmts, mov_var, mov_slices, dtype_str, name_gen)
    stmts.append(NKIMatmul(dst=psum_ref, stationary=stat_ref, moving=mov_ref))


def _emit_staging(stmts: list[NKIOp], psum_ref: TensorRef, dtype_str: str, name_gen: _NameGen) -> TensorRef:
    """Emit PSUM->SBUF tensor copy for post-reduction compute.

    Args:
        stmts: Mutable statement accumulator.
        psum_ref: PSUM source reference.
        dtype_str: NKI dtype string.
        name_gen: Name generator.

    Returns:
        TensorRef for the staged SBUF tile.
    """
    sbuf_name = name_gen.next()
    sbuf_slices = _full_slices(psum_ref.shape)
    stmts.append(NKIAlloc(dst=sbuf_name, shape=psum_ref.shape, dtype=dtype_str, buffer="sbuf"))
    stmts.append(NKITensorCopy(dst=_ref(sbuf_name, psum_ref.shape, sbuf_slices), src=psum_ref))
    return _ref(sbuf_name, psum_ref.shape, sbuf_slices)


def _resolve_activation_op(op_call: _OpCall) -> str:
    """Resolve activation function to NKI string.

    Args:
        op_call: The activation op call with config kwargs.

    Returns:
        NKI activation function string.
    """
    result = "nl.identity"
    for key, value in op_call.config_kwargs:
        if key == "op" and isinstance(value, np.ufunc):
            result = f"nl.{value.__name__}"
    return result


def _emit_activation(
    stmts: list[NKIOp], src_ref: TensorRef, op_call: _OpCall, dtype_str: str, name_gen: _NameGen
) -> TensorRef:
    """Emit SBUF alloc + activation for one post-reduction op.

    Args:
        stmts: Mutable statement accumulator.
        src_ref: Source SBUF reference.
        op_call: The activation op call.
        dtype_str: NKI dtype string.
        name_gen: Name generator.

    Returns:
        TensorRef for the activation output.
    """
    op_str = _resolve_activation_op(op_call)
    dst_name = name_gen.next()
    dst_slices = _full_slices(src_ref.shape)
    stmts.append(NKIAlloc(dst=dst_name, shape=src_ref.shape, dtype=dtype_str, buffer="sbuf"))
    stmts.append(NKIActivation(dst=_ref(dst_name, src_ref.shape, dst_slices), src=src_ref, op=op_str))
    return _ref(dst_name, src_ref.shape, dst_slices)


def _emit_store(
    stmts: list[NKIOp], src_ref: TensorRef, output_shape: tuple[int, ...], output_slices: tuple[tuple[int, int], ...]
) -> None:
    """Emit DMA store from SBUF to HBM output.

    Args:
        stmts: Mutable statement accumulator.
        src_ref: Source SBUF reference.
        output_shape: Full output tensor shape.
        output_slices: Destination slice bounds in HBM output.
    """
    stmts.append(NKIDmaCopy(dst=_ref(_OUTPUT_NAME, output_shape, output_slices), src=src_ref))


def _build_block(
    block_idx: int,
    op_calls: list[_OpCall],
    analysis: _Analysis,
    parallel_pos: dict[str, int],
    params: tuple[str, ...],
    dtype_str: str,
    name_gen: _NameGen,
) -> NKIBlock:
    """Build one NKIBlock for a parallel tile position.

    Args:
        block_idx: Block index for naming.
        op_calls: All parsed op calls.
        analysis: Dimension analysis.
        parallel_pos: Parallel tile position.
        params: Kernel parameter names.
        dtype_str: NKI dtype string.
        name_gen: Name generator.

    Returns:
        Complete NKIBlock.
    """
    stmts: list[NKIOp] = []
    matmul_ops = [op for op in op_calls if op.stmt_type is NKIMatmul]
    post_ops = [op for op in op_calls if op.stmt_type is not NKIMatmul]
    output_shape = analysis.var_shapes[analysis.return_var]
    output_slices = compute_slices(analysis.return_var, parallel_pos, analysis)
    tile_shape = tuple(e - s for s, e in output_slices)
    psum_name = name_gen.next()
    psum_slices = _full_slices(tile_shape)
    stmts.append(NKIAlloc(dst=psum_name, shape=tile_shape, dtype="nl.float32", buffer="psum"))
    psum_ref = _ref(psum_name, tile_shape, psum_slices)
    for red_pos in iter_reduction_positions(analysis):
        combined = {**parallel_pos, **red_pos}
        for matmul_op in matmul_ops:
            _emit_matmul_group(stmts, matmul_op, analysis, combined, psum_ref, dtype_str, name_gen)
    src_ref = _emit_staging(stmts, psum_ref, dtype_str, name_gen)
    for post_op in post_ops:
        src_ref = _emit_activation(stmts, src_ref, post_op, dtype_str, name_gen)
    _emit_store(stmts, src_ref, output_shape, output_slices)
    block_params = tuple(list(params) + [_OUTPUT_NAME])
    return NKIBlock(name=f"_block_{block_idx}", params=block_params, body=tuple(stmts))


def _build_all_blocks(
    op_calls: list[_OpCall], analysis: _Analysis, params: tuple[str, ...], dtype_name: str
) -> list[NKIBlock]:
    """Build all NKI blocks for every parallel tile position.

    Args:
        op_calls: All parsed op calls.
        analysis: Dimension analysis.
        params: Kernel parameter names.
        dtype_name: Numpy dtype name (e.g. ``"float16"``).

    Returns:
        List of NKIBlocks.
    """
    dtype_str = f"nl.{dtype_name}"
    name_gen = _NameGen()
    blocks: list[NKIBlock] = []
    for block_idx, parallel_pos in iter_tile_positions(analysis):
        block = _build_block(block_idx, op_calls, analysis, parallel_pos, params, dtype_str, name_gen)
        blocks.append(block)
    return blocks
