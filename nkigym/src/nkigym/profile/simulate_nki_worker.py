"""Standalone remote worker for batched fp32 NKI simulation."""

from __future__ import annotations

import ast
import copy
import importlib
import json
import multiprocessing
import pickle
import re
import sys
import threading
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import cache
from math import erf
from pathlib import Path
from typing import Any, cast

import ml_dtypes
import nki
import numpy as np

_FP_DTYPES_NON_FP32 = """
bfloat16 float16 float8_e4m3 float8_e4m3fn float8_e4m3fn_x4
float8_e5m2 float8_e5m2_x4 float4_e2m1fn_x4 tfloat32
""".split()
ArrayResult = np.ndarray | tuple[np.ndarray, ...]
_SerializedCase = tuple[int, str, str, str, dict[str, np.ndarray], ArrayResult, tuple[str, dict[str, object]] | None]
_FailurePayload = dict[str, int | str]
_Provenance = dict[tuple[object, ...], dict[str, Any]]
_ProvenanceStore = dict[tuple[object, ...], _Provenance]
_WORKER_CASES: list[_SerializedCase] = []
_WORKER_ATOL, _WORKER_RTOL = 0.0, 0.0
_SIMULATION_LOCK = threading.Lock()
_LNC2 = re.compile(r"(?s)(?:(\w+)=nl\.program_id\(0\).*?\[[^\n]*\b\1|\[[^\n]*nl\.program_id\(0\)[^\n]*)")
_OUTPUT = re.compile(r"(?m)^\s+(\w+) = nl\.ndarray\([^\n]*dtype=nl\.(\w+), buffer=nl\.shared_hbm\)")


def _fp32_source(source: str) -> str:
    """Rewrite reduced-precision NKI language dtypes to fp32."""
    preserved = set(re.findall(r"\b(sbuf_semantic_\w+|\w+(?=\s*=.*float8_\w+.*shared_hbm))\s*=", source))
    pairs = re.findall(r"nisa\.(?:dma_transpose|nc_matmul)\((?:src|stationary)=(\w+).*?(?:dst|moving)=(\w+)", source)
    preserved.update(right for left, right in pairs if left in preserved)
    for name in preserved:
        source = re.sub(rf"(\b{re.escape(name)}\s*=.*?\bdtype=)nl\.", rf"\1nkigym_nl.", source)
    for dtype in _FP_DTYPES_NON_FP32:
        source = re.sub(rf"\bnl\.{re.escape(dtype)}\b", "nl.float32", source)
    return _coalesce_dma_copies(source.replace("nkigym_nl.", "nl."))


def _output_dtypes(source: str, func_name: str) -> tuple[str, ...]:
    """Read physical output dtypes from one rendered kernel."""
    declarations = dict(_OUTPUT.findall(source))
    returned = re.findall(r"(?m)^\s+return (.+)$", source)
    names = tuple(name.strip() for name in returned[0].strip("()").split(",")) if len(returned) == 1 else ()
    if f"def {func_name}(" not in source or not names or not set(names) <= declarations.keys():
        raise ValueError(f"generated kernel {func_name!r} has invalid output declarations")
    return tuple(declarations[name] for name in names)


def _node_names(node: ast.AST, excluded: ast.AST | None = None) -> set[str]:
    """Return referenced names outside an optional excluded subtree."""
    nodes = set(ast.walk(node)) - (set(ast.walk(excluded)) if excluded is not None else set())
    return {item.id for item in nodes if isinstance(item, ast.Name)}


def _sliced_names(node: ast.expr) -> set[str]:
    """Return names whose occurrences are confined to one indexed slice."""
    slices = (part for part in ast.walk(node) if isinstance(part, ast.Slice))
    return set().union(*(_node_names(part) - _node_names(node, part) for part in slices))


def _array_root(node: ast.expr) -> str | None:
    """Return the named allocation underlying a simple indexed operand."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _affine_form(node: ast.AST) -> dict[str | None, int] | None:
    """Return one generated affine expression as variable coefficients."""
    if isinstance(node, ast.Name):
        return {node.id: 1}
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return {None: int(node.value)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = _affine_form(node.left), _affine_form(node.right)
        if left is None or right is None:
            return None
        sign = 1 if isinstance(node.op, ast.Add) else -1
        for name, value in right.items():
            left[name] = left.get(name, 0) + sign * value
        return {name: value for name, value in left.items() if value}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        constant = node.left if isinstance(node.left, ast.Constant) else node.right
        expression = node.right if constant is node.left else node.left
        if isinstance(constant, ast.Constant) and isinstance(constant.value, int):
            form = _affine_form(expression)
            return None if form is None else {name: int(constant.value) * value for name, value in form.items()}
    return None


class _ZeroLoops(ast.NodeTransformer):
    """Replace collapsed loop variables with zero."""

    def __init__(self, names: set[str]) -> None:
        """Store the loop variables removed by coalescing."""
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Replace one collapsed loop variable."""
        return ast.copy_location(ast.Constant(value=0), node) if node.id in self.names else node


def _coalesced_operand(node: ast.expr, loops: tuple[tuple[str, int], ...]) -> tuple[ast.expr, tuple[int, ...]] | None:
    """Merge contiguous loop digits while proving disjoint retained intervals."""
    result, names = copy.deepcopy(node), {name for name, _ in loops}
    candidates = [item for item in ast.walk(result) if isinstance(item, ast.Slice) and names & _node_names(item)]
    if not isinstance(result, ast.Subscript) or len(candidates) != 1 or not names <= _sliced_names(result):
        return None
    match target := candidates[0]:
        case ast.Slice(ast.expr() as lower, ast.BinOp(repeated, ast.Add(), ast.Constant(value=int() as width)), None):
            if width <= 0 or ast.dump(repeated) != ast.dump(lower):
                return None
        case _:
            return None
    form = _affine_form(lower)
    coefficients = [None if form is None else form.get(name, 0) for name, _ in loops]
    if any(value is None or value <= 0 or value % width for value in coefficients):
        return None
    digits = sorted(
        (cast(int, coefficient) // width, extent) for coefficient, (_, extent) in zip(coefficients, loops, strict=True)
    )
    span, covered = 1, 1
    for stride, extent in digits:
        if stride < covered:
            return None
        span *= extent if stride == span else 1
        covered = stride * extent
    if span == 1:
        return None
    names = {name for (name, _), coefficient in zip(loops, coefficients) if cast(int, coefficient) < width * span}
    target.lower = cast(ast.expr, _ZeroLoops(names).visit(lower))
    target.upper = ast.BinOp(left=copy.deepcopy(target.lower), op=ast.Add(), right=ast.Constant(value=width * span))
    return result, (*cast(tuple[int, ...], tuple(coefficients)), width, span)


def _loop_info(node: ast.For) -> tuple[str, int] | None:
    """Return one simple positive constant-range loop."""
    match node:
        case ast.For(ast.Name(id=name), ast.Call(ast.Name(id="range"), [ast.Constant(value=value)]), _, []):
            return (name, int(value)) if isinstance(value, int) and value > 0 else None
    return None


def _coalescible_call(node: ast.stmt) -> tuple[str, ...] | None:
    """Return one direct NKI call and its contiguous tensor operands."""
    match node:
        case ast.Expr(ast.Call(ast.Attribute(ast.Name(id="nisa"), attr=operation)) as call):
            operands = {"memset": ("dst",), "range_select": ("on_true_tile", "dst")}.get(operation)
            if operation in {"dma_copy", "tensor_copy"}:
                operands = ("src", "dst")
            if operation == "activation" and {k.arg for k in call.keywords} == {"data", "dst", "op"}:
                operands = ("data", "dst")
            return operands
    return None


def _coalesced_call(node: ast.For) -> ast.stmt | None:
    """Merge equal copy mappings, retaining loops over disjoint strided intervals."""
    loops, nests, statement = [], [], cast(ast.stmt, node)
    while isinstance(statement, ast.For) and len(statement.body) == 1 and (loop := _loop_info(statement)) is not None:
        nests.append(statement)
        loops.append(loop)
        statement = statement.body[0]
    if (operand_names := _coalescible_call(statement)) is None:
        return None
    result = copy.deepcopy(statement)
    bindings = {keyword.arg: keyword for keyword in cast(ast.Call, cast(ast.Expr, result).value).keywords}
    left, right = (bindings[name].value for name in (operand_names[0], operand_names[-1]))
    roots = (_array_root(left), _array_root(right))
    if None in roots or roots[0] == roots[1] and ast.dump(left) != ast.dump(right):
        return None
    names = {name for name, _ in loops} & _sliced_names(left) & _sliced_names(right)
    loops = [(name, extent) for name, extent in loops if name in names]
    fixed = bindings.keys() - set(operand_names) - {"range_start"}
    if any(names & _node_names(bindings[name].value) for name in fixed):
        return None
    if "range_start" in bindings:
        source = bindings[operand_names[0]].value
        slices = [item for item in ast.walk(source) if isinstance(item, ast.Slice) and names & _node_names(item)]
        if len(slices) != 1 or _affine_form(cast(ast.expr, slices[0].lower)) != _affine_form(
            bindings["range_start"].value
        ):
            return None
    operands = {name: _coalesced_operand(bindings[name].value, tuple(loops)) for name in operand_names}
    layouts = {None if value is None else value[1] for value in operands.values()}
    if None in layouts or len(layouts) != 1:
        return None
    for name, value in operands.items():
        bindings[name].value = cast(tuple[ast.expr, tuple[int, ...]], value)[0]
    names.difference_update(*(_node_names(bindings[name].value) for name in operand_names))
    if "range_start" in bindings:
        bindings["range_start"].value = cast(ast.expr, _ZeroLoops(names).visit(bindings["range_start"].value))
    for nest in reversed(nests):
        if cast(ast.Name, nest.target).id not in names:
            result = ast.For(nest.target, nest.iter, [result], [])
    return ast.copy_location(result, node)


class _CoalesceDMACopies(ast.NodeTransformer):
    """Coalesce contiguous CPU simulator calls."""

    def visit_For(self, node: ast.For) -> ast.stmt:
        """Collapse complete contiguous copy nests without moving memory accesses."""
        return _coalesced_call(cast(ast.For, self.generic_visit(node))) or node


def _coalesce_dma_copies(source: str) -> str:
    """Coalesce proven contiguous tensor calls in standalone simulator source."""
    tree = _CoalesceDMACopies().visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def _view_key(view: Any) -> tuple[object, ...]:
    """Return the simulator's identity for one tensor view."""
    identity = (view.tensor_id, view.tensor.__array_interface__["data"][0])
    return identity if view._is_identity() else identity + (view.offset, tuple(tuple(row) for row in view.pattern))


@contextmanager
def _regular_tensor_views() -> Iterator[None]:
    """Use checked NumPy strides for regular views, retaining SDK fallbacks."""
    view_type = importlib.import_module("nki._backends.simulator.tensor_view").SimulatorTensorView
    from_numpy_dtype = importlib.import_module("nki._backends.simulator.dtypes").from_numpy_dtype
    original_get, original_set = view_type.get_data, view_type.set_data

    def array(view: Any) -> np.ndarray | None:
        """Return an in-bounds, injective view with unchanged element dtype."""
        indexed = any(value is not None for value in (view.ti_state, view.scalar_offset, view.vector_offset))
        compatible = view.tensor.flags.c_contiguous and from_numpy_dtype(view.tensor.dtype) == view.dtype
        if view.pattern is None or indexed or not compatible:
            return None
        span = 1
        for step, count in sorted(view.pattern):
            if count <= 0 or step < span:
                return None
            span += step * (count - 1)
        if view.offset < 0 or view.offset + span > view.tensor.size:
            return None
        storage, size = view.tensor, view.tensor.itemsize
        strides = tuple(step * size for step, _ in view.pattern)
        return np.ndarray(view.view_shape, storage.dtype, buffer=storage, offset=view.offset * size, strides=strides)

    def get_data(view: Any) -> np.ndarray:
        """Preserve the SDK's nonidentity read snapshot semantics."""
        target = array(view)
        return original_get(view) if target is None else target.copy()

    def set_data(view: Any, value: object) -> None:
        """Write regular views with the SDK's flattened assignment semantics."""
        target, values = array(view), np.asarray(value).ravel()
        if target is None or values.size not in (1, target.size):
            original_set(view, value)
        else:
            target[...] = values.reshape(target.shape) if values.size == target.size else values[0]

    view_type.get_data, view_type.set_data = get_data, set_data
    try:
        yield
    finally:
        view_type.get_data, view_type.set_data = original_get, original_set


def _outer_fma(left: np.ndarray, right: np.ndarray, addend: np.ndarray) -> np.ndarray:
    """Evaluate one float32 outer-product FMA with one final rounding."""
    product = left.astype(np.float64)[:, None] * right.astype(np.float64)[None, :]
    return np.asarray(product + addend.astype(np.float64), dtype=np.float32)


def _mkl_gemv_result(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Match four MKL thread chunks, each using the AVX-512 eight-term tree."""
    partials = np.zeros((4, left.shape[1], right.shape[1]), dtype=np.float32)
    for start in range(0, left.shape[0], 8):
        chunk = start // (left.shape[0] // 4)
        tile_left, tile_right = left[start : start + 8], right[start : start + 8]
        products = np.asarray(tile_left[:, :, None] * tile_right[:, None, :], dtype=np.float32)
        pairs = [_outer_fma(tile_left[i], tile_right[i], products[j]) for i, j in ((5, 7), (0, 2), (1, 3))]
        carried = _outer_fma(tile_left[4], tile_right[4], _outer_fma(tile_left[6], tile_right[6], partials[chunk]))
        partials[chunk] = np.add(np.add(pairs[1], pairs[2]), np.add(carried, pairs[0]), dtype=np.float32)
    return np.add.accumulate(partials, axis=0, dtype=np.float32)[-1]


def _torch_add_reduce(values: np.ndarray) -> np.ndarray:
    """Match PyTorch's AVX2 cascade sum over one free axis."""
    units = values.reshape(values.shape[0], -1, 4, 8)
    parts = []
    while units.shape[1] >= 16:
        full = units.shape[1] // 16 * 16
        if full < units.shape[1]:
            parts.append(np.add.accumulate(units[:, full:], axis=1, dtype=np.float32)[:, -1])
        groups = units[:, :full].reshape(values.shape[0], -1, 16, 4, 8)
        units = np.add.accumulate(groups, axis=2, dtype=np.float32)[:, :, -1]
    parts.append(np.add.accumulate(units, axis=1, dtype=np.float32)[:, -1])
    total = np.add.accumulate(np.stack(parts, axis=1), axis=1, dtype=np.float32)[:, -1]
    total = np.add.accumulate(total, axis=1, dtype=np.float32)[:, -1]
    return np.add.accumulate(total, axis=1, dtype=np.float32)[:, -1, None]


@contextmanager
def _grouped_matmul_accumulation() -> Iterator[None]:
    """Preserve logical contractions and the CPU reference's blocked sum order."""
    simulator = cast(Any, importlib.import_module("nki._backends.simulator"))
    to_numpy_dtype = importlib.import_module("nki._backends.simulator.dtypes").to_numpy_dtype
    LncContext = importlib.import_module("nki._backends.simulator.lnc").LncContext
    _flatten_to_2d = importlib.import_module("nki._backends.simulator.matmul")._flatten_to_2d
    get_current_context = importlib.import_module("nki._backends.simulator.state").get_current_context
    SimulatorTensorView = importlib.import_module("nki._backends.simulator.tensor_view").SimulatorTensorView
    language_ops = cast(Any, importlib.import_module("nki.language._ops"))

    original_activation = simulator.activation
    original_matmul, original_copy, original_sendrecv = simulator.nc_matmul, simulator.tensor_copy, simulator.sendrecv
    original_tensor_tensor, original_reduce_op = simulator.tensor_tensor_arith, language_ops.get_numpy_reduce_op
    original_get, original_set = SimulatorTensorView.get_data, SimulatorTensorView.set_data
    pending, symbolic = cast(tuple[_ProvenanceStore, _ProvenanceStore], ({}, {}))
    materializing: set[tuple[object, ...]] = set()
    inverse_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    peer_panels: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}

    def allocation(items: _ProvenanceStore, view: Any) -> _Provenance:
        """Return provenance views belonging to one allocation."""
        return items.get((view.tensor_id, view.tensor.__array_interface__["data"][0]), {})

    def view_indices(view: Any) -> np.ndarray:
        """Return flattened storage indices in view iteration order."""
        indices = np.arange(view.tensor.size, dtype=np.int64) if view._is_identity() else view._get_indices()
        return np.asarray(indices, dtype=np.int64).reshape(-1)

    def view_span(view: Any) -> tuple[int, int]:
        """Return inclusive storage bounds for one regular tensor view."""
        deltas, offset = [int(step) * (int(count) - 1) for step, count in view._get_pattern()], int(view.offset)
        return offset + sum(min(0, delta) for delta in deltas), offset + sum(max(0, delta) for delta in deltas)

    def item_indices(item: dict[str, Any]) -> np.ndarray:
        """Return and cache one provenance view's storage indices."""
        if (indices := item.get("indices")) is None:
            item["indices"] = indices = view_indices(item["view"])
        return cast(np.ndarray, indices)

    item_overlaps = lambda item, target: (span := view_span(item["view"]))[0] <= target[1] and target[0] <= span[1]

    def item_pages(item: dict[str, Any]) -> frozenset[int]:
        """Return and cache storage pages touched by one provenance view."""
        pages = item.get("pages")
        if pages is None:
            pages = frozenset(item_indices(item) // (1 << 13))
            item["pages"] = pages
        return cast(frozenset[int], pages)

    def remap_positions(view: Any, absolute: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map storage indices to positions in one target view."""
        key = _view_key(view)
        if key not in inverse_cache:
            order = np.argsort((pattern := np.asarray(view._get_pattern(), dtype=np.int64))[:, 0])
            steps, counts = pattern[order].T
            strides = np.asarray([np.prod(pattern[dim + 1 :, 1]) for dim in order], dtype=np.int64)
            valid = view.scalar_offset is None and view.vector_offset is None and np.all(steps > 0)
            valid &= np.all(steps[1:] % (steps[:-1] * counts[:-1]) == 0)
            inverse_cache[key] = (steps, counts, strides) if valid else None
        if (inverse := inverse_cache[key]) is not None:
            steps, counts, strides = inverse
            relative = absolute - int(view.offset)
            if len(steps) == 2 and steps[0] == 1:
                rows, columns = np.divmod(relative, steps[1])
                selected = (relative >= 0) & (rows < counts[1]) & (columns < counts[0])
                return selected, np.where(selected, rows * counts[0] + columns, 0)
            coordinates = relative[:, None] // steps % counts
            selected = (relative >= 0) & (coordinates @ steps == relative)
            positions = coordinates @ strides
            return selected, np.where(selected, positions, 0)
        order = np.argsort(indices := view_indices(view), kind="stable")
        indices = indices[order]
        locations = np.searchsorted(indices, absolute, side="right") - 1
        safe = np.maximum(locations, 0)
        selected = (locations >= 0) & (indices[safe] == absolute)
        return selected, order[safe]

    def grouped_result(stationary: list[np.ndarray], moving: list[np.ndarray], names: list[str]) -> np.ndarray:
        """Match blocked BLAS sums, reducing K workers before the main thread."""
        ordered = sorted(zip(names, stationary, moving), key=lambda item: int(item[0].rsplit("_", 1)[1]))
        names, stationary, moving = (list(values) for values in zip(*ordered))
        name = names[0].rsplit("_", 1)[0]
        count, widths = len(stationary), (stationary[0].shape[1], moving[0].shape[1])
        left, right = (np.concatenate(parts, axis=0) for parts in (stationary, moving))
        if name.startswith("conv_"):
            active, channels = int(np.flatnonzero(np.any(right, axis=1))[-1]) + 1, int(name.split("_")[1])
            kernel, result = active // channels, np.zeros(widths, dtype=np.float32)
            for start in range(0, channels, 16):
                stop = min(channels, start + 16)
                indices = np.arange(start * kernel, stop * kernel).reshape(-1, kernel).T.reshape(-1)
                result += np.matmul(left[indices].T, right[indices])
            return result
        if count == 32 and min(widths) == 1 and max(widths) % 16 == 0:
            return _mkl_gemv_result(left, right)
        threads = 4 if left.shape[0] >= 4096 and left.shape[0] % 512 == 0 and min(widths) < 128 else 1
        group = {4: 4, 8: 3, 24: 2, 32: 3, 64: 1 + 2 * (widths[0] >= 128), 128: 2}.get(count, 1)
        group = 384 if threads == 4 else group * stationary[0].shape[0]
        partials = np.zeros((threads, *widths), dtype=np.float32)
        for thread in range(threads):
            start, stop = thread * (left.shape[0] // threads), (thread + 1) * (left.shape[0] // threads)
            for index in range(start, stop, group):
                end = min(index + group, stop)
                partials[thread] += np.matmul(left[index:end].T, right[index:end])
        return partials[0] if threads == 1 else ((partials[1] + partials[2]) + partials[3]) + partials[0]

    def write_materialized(key: tuple[object, ...], view: Any, value: np.ndarray) -> None:
        """Write one value without invalidating its provenance."""
        materializing.add(key)
        try:
            original_set(view, value)
        finally:
            materializing.remove(key)

    def flush(view: Any) -> None:
        """Write every pending subview overlapping one read or write."""
        target_span = view_span(view)
        for key, item in tuple(allocation(pending, view).items()):
            if not item_overlaps(item, target_span) or not item["dirty"]:
                continue
            result = grouped_result(item["stationary"], item["moving"], item["name"])
            if item["base"] is not None:
                result = item["base"] + result
            target = item["view"]
            write_materialized(key, target, result.reshape(target.view_shape).astype(to_numpy_dtype(target.dtype)))
            item["dirty"] = False

    def flush_symbolic(view: Any) -> None:
        """Materialize only symbolic panels selected by one read."""
        panels = take_symbolic(view)
        if not panels:
            return
        data = original_get(view).copy().reshape(-1)
        grouped: dict[bytes, list[dict[str, Any]]] = {}
        for panel in panels:
            grouped.setdefault(panel["positions"].tobytes(), []).append(panel)
        for selected in grouped.values():
            selected.sort(key=lambda panel: panel.get("rank", 0))
            stationary = [array for panel in selected for array in panel["stationary"]]
            moving = [array for panel in selected for array in panel["moving"]]
            names = [name for panel in selected for name in panel["name"]]
            result = grouped_result(stationary, moving, names)
            bases = [panel["base"] for panel in selected if panel["base"] is not None]
            if bases:
                result += sum(bases[1:], start=bases[0].copy())
            data[selected[0]["positions"]] = result.reshape(-1)
        write_materialized(_view_key(view), view, data.reshape(view.view_shape))

    def slice_panel(panel: dict[str, Any], columns: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
        """Slice one symbolic matmul panel to complete result columns."""
        base = None if panel["base"] is None else panel["base"][:, columns]
        moving = [array[:, columns] for array in panel["moving"]]
        return panel | {"base": base, "moving": moving, "positions": positions}

    def pending_panels(view: Any) -> list[dict[str, Any]]:
        """Return pending contraction panels selected by one result subview."""
        panels = []
        target_span = view_span(view)
        for item in allocation(pending, view).values():
            if item_overlaps(item, target_span):
                selected, remapped = remap_positions(view, item_indices(item))
                selected = selected.reshape(item["view"].view_shape)
                columns = np.flatnonzero(selected[0])
                complete = bool(columns.size and np.all(selected[:, columns]))
                complete &= np.count_nonzero(selected) == selected.shape[0] * columns.size
                if complete:
                    panel = {key: item[key] for key in ("base", "moving", "name", "stationary")}
                    panel["name"], panel["stationary"] = list(panel["name"]), list(panel["stationary"])
                    panels.append(slice_panel(panel, columns, remapped[selected.reshape(-1)]))
        return panels

    def matching_pending_panel(candidates: list[dict[str, Any]], peer: dict[str, Any]) -> dict[str, Any] | None:
        """Return the local contraction columns corresponding to one peer panel."""
        for panel in candidates:
            rows = panel["stationary"][0].shape[1]
            selected = np.isin(panel["positions"], peer["positions"]).reshape(rows, -1)
            if np.all(selected == selected[0]) and np.count_nonzero(selected) == peer["positions"].size:
                return slice_panel(panel, np.flatnonzero(selected[0]), peer["positions"])
        return None

    def take_symbolic(view: Any) -> list[dict[str, Any]]:
        """Remove and remap symbolic subviews contained by ``view``."""
        items = allocation(symbolic, view)
        if not items:
            return []
        key = _view_key(view)
        exact = items.pop(key, None)
        panels = [] if exact is None else list(exact["panels"])
        target_span = view_span(view)
        matches = [(item_key, item) for item_key, item in items.items() if item_overlaps(item, target_span)]
        if len(matches) > 1:
            target_pages = frozenset(view_indices(view) // (1 << 13))
            matches = [(item_key, item) for item_key, item in matches if not target_pages.isdisjoint(item_pages(item))]
        for key, item in matches:
            positions = tuple(panel["positions"] for panel in item["panels"])
            ends = np.cumsum([value.size for value in positions])
            selected, remapped = remap_positions(view, item_indices(item)[np.concatenate(positions)])
            remaining = []
            for panel, selected, remapped in zip(
                item["panels"], np.split(selected, ends[:-1]), np.split(remapped, ends[:-1]), strict=True
            ):
                if not np.any(selected):
                    remaining.append(panel)
                    continue
                rows, columns = panel["stationary"][0].shape[1], panel["moving"][0].shape[1]
                selected = selected.reshape(rows, columns)
                if not np.all(selected == selected[0]):
                    raise RuntimeError("symbolic matmul views must select complete result columns")
                chosen, retained = np.flatnonzero(selected[0]), np.flatnonzero(~selected[0])
                panels.append(slice_panel(panel, chosen, remapped[selected.reshape(-1)]))
                if retained.size:
                    remaining_mask = ~selected
                    remaining.append(slice_panel(panel, retained, panel["positions"][remaining_mask.reshape(-1)]))
            if remaining:
                item["panels"] = remaining
            else:
                del allocation(symbolic, view)[key]
        return panels

    def get_data(view: Any) -> np.ndarray:
        """Materialize pending matmuls before reading one allocation."""
        flush(view)
        flush_symbolic(view)
        return cast(np.ndarray, original_get(view))

    def invalidate_written_view(view: Any) -> None:
        """Preserve untouched data and discard provenance overlapped by a new write."""
        target_span = view_span(view)
        candidates = [
            (pending_key, item)
            for pending_key, item in allocation(pending, view).items()
            if item_overlaps(item, target_span)
        ]
        if candidates:
            flush(view)
            for pending_key, item in candidates:
                if np.any(remap_positions(view, item_indices(item))[0]):
                    del allocation(pending, view)[pending_key]
        take_symbolic(view)

    def set_data(view: Any, value: object) -> None:
        """Preserve pending subviews before an explicit write resets them."""
        if _view_key(view) not in materializing:
            invalidate_written_view(view)
        original_set(view, value)

    def tensor_copy(dst: Any, src: Any, engine: object, name: object) -> None:
        """Match native integer rounding or retain provenance across an RFactor drain."""
        integer = str(engine) == "engine.vector" and str(dst.dtype).startswith(("int", "uint"))
        panels = [] if integer else pending_panels(src)
        if integer and (values := get_data(src)).dtype.kind == "f":
            set_data(dst, np.rint(values).astype(to_numpy_dtype(dst.dtype)))
        else:
            original_copy(dst, src, engine, name)
        if panels:
            key = _view_key(dst)
            symbolic.setdefault(key[:2], {})[key] = {"dirty": True, "panels": panels, "view": dst}

    def sendrecv(src: Any, dst: Any, *arguments: object, **keywords: object) -> None:
        """Exchange peer data and exact symbolic matmul provenance."""
        send_to, recv_from, pipe = (cast(int, keywords[name]) for name in ("send_to_rank", "recv_from_rank", "pipe_id"))
        if (context := LncContext.get_current()) is None:
            raise RuntimeError("sendrecv requires an active LNC context")
        rank = context.get_program_id()
        sequence = getattr(getattr(context, "_thread_local"), f"sendrecv_seq_{pipe}", 0)
        peer_panels[(rank, send_to, pipe, sequence)] = panels = [panel | {"rank": rank} for panel in take_symbolic(src)]
        original_sendrecv(src, dst, *arguments, **keywords)
        if panels:
            symbolic.setdefault((key := _view_key(src))[:2], {})[key] = {"dirty": True, "panels": panels, "view": src}
        if received := peer_panels.pop((recv_from, rank, pipe, sequence)):
            symbolic.setdefault((key := _view_key(dst))[:2], {})[key] = {"dirty": True, "panels": received, "view": dst}

    def tensor_tensor(dst: Any, data1: Any, data2: Any, op: object, engine: object, name: object) -> None:
        """Preserve RFactor provenance across its in-place SBUF add."""
        destination, panels = _view_key(dst), []
        if str(op) == "add" and destination == _view_key(data1):
            if right := take_symbolic(data2):
                panels.extend(take_symbolic(data1) + right)
        elif str(op) == "add":
            local = pending_panels(data1)
            if local and (right := take_symbolic(data2)) and "rank" in right[0]:
                for peer in right:
                    if (panel := matching_pending_panel(local, peer)) is not None:
                        panels.extend((panel | {"rank": 1 - peer["rank"]}, peer))
        original_tensor_tensor(dst, data1, data2, op, engine, name)
        if panels:
            symbolic.setdefault(destination[:2], {})[destination] = {"dirty": True, "panels": panels, "view": dst}

    def activation(**keywords: object) -> None:
        """Evaluate erf-based activations in FP32 when simulator approximations are too coarse."""
        operation = str(keywords["op"])
        if operation not in {"erf", "gelu"} or keywords["reduce_op"] is not None:
            original_activation(**keywords)
            return
        values = get_data(keywords["data"]).astype(np.float32) * cast(float, keywords["scale"])
        bias = keywords["bias"]
        if bias is not None:
            values = values + (get_data(bias).astype(np.float32) if hasattr(bias, "tensor") else cast(float, bias))
        transformed = values / np.sqrt(2.0) if operation == "gelu" else values
        result = np.asarray(np.frompyfunc(erf, 1, 1)(transformed), dtype=np.float32)
        result = 0.5 * values * (1.0 + result) if operation == "gelu" else result
        set_data(keywords["dst"], result)

    def matmul(**keywords: object) -> None:
        """Record one contraction tile or delegate unsupported matmul modes."""
        dst, stationary, moving = (cast(Any, keywords[name]) for name in ("dst", "stationary", "moving"))
        if keywords["is_transpose"] or stationary.ti_state is not None or moving.ti_state is not None:
            original_matmul(**keywords)
            return
        left, right = (
            _flatten_to_2d(original_get(operand).astype(np.float32), keywords["perf_mode"])
            for operand in (stationary, moving)
        )
        key, written = _view_key(dst), get_current_context().psum_written
        bucket = pending.setdefault(key[:2], {})
        should_accumulate = keywords["accumulate"] if keywords["accumulate"] is not None else key in written
        if keywords["accumulate"] is False or key not in bucket:
            invalidate_written_view(dst)
            bucket[key] = {
                "base": original_get(dst).copy() if should_accumulate else None,
                "dirty": False,
                "moving": [],
                "name": [],
                "stationary": [],
                "view": dst,
            }
        bucket[key]["stationary"].append(left.copy())
        bucket[key]["moving"].append(right.copy())
        bucket[key]["name"].append(str(keywords["name"]))
        bucket[key]["dirty"] = True
        written[key] = True

    def numpy_reduce_op(operation: object) -> Any:
        """Use Torch's additive reduction order for complete free axes."""
        fallback = original_reduce_op(operation)
        if str(operation) != "add":
            return fallback

        def reduce(values: np.ndarray, axis: tuple[int, ...], keepdims: bool = False) -> np.ndarray:
            width = values.size // values.shape[0]
            if axis != tuple(range(1, values.ndim)) or width % 32:
                return cast(np.ndarray, fallback(values, axis=axis, keepdims=keepdims))
            result = _torch_add_reduce(values.reshape(values.shape[0], width))
            return result.reshape((values.shape[0],) + (1,) * (values.ndim - 1)) if keepdims else result[:, 0]

        return reduce

    language_ops.get_numpy_reduce_op = numpy_reduce_op
    simulator.activation = activation
    simulator.nc_matmul, simulator.tensor_copy, simulator.tensor_tensor_arith = matmul, tensor_copy, tensor_tensor
    simulator.sendrecv = sendrecv
    SimulatorTensorView.get_data, SimulatorTensorView.set_data = get_data, set_data
    try:
        yield
    finally:
        language_ops.get_numpy_reduce_op = original_reduce_op
        simulator.activation = original_activation
        simulator.nc_matmul, simulator.tensor_copy = original_matmul, original_copy
        simulator.tensor_tensor_arith = original_tensor_tensor
        simulator.sendrecv = original_sendrecv
        SimulatorTensorView.get_data, SimulatorTensorView.set_data = original_get, original_set


def _simulate_kernel_fp32(
    kernel: object, call: tuple[tuple[object, ...], dict[str, object]], grouped: bool = True
) -> object:
    """Run one rewritten kernel, optionally matching grouped reference reductions."""
    with _SIMULATION_LOCK, _regular_tensor_views(), _grouped_matmul_accumulation() if grouped else nullcontext():
        return nki.simulate(kernel)(*call[0], **call[1])


def _simulate_source_fp32(
    source: str, func_name: str, inputs: dict[str, np.ndarray], grouped: bool = False
) -> ArrayResult:
    """Execute native FP32 simulation or the explicit reference-grouping fallback."""
    output_dtypes = _output_dtypes(source, func_name)
    namespace: dict = {}
    exec(compile(_fp32_source(source), f"<batch-case-{func_name}>", "exec"), namespace)  # noqa: S102
    kernel = namespace[func_name][1 + bool(_LNC2.search(source.replace(" ", "")))]
    result = _simulate_kernel_fp32(kernel, ((), cast(dict[str, object], inputs)), grouped)
    values = [np.asarray(value) for value in (result if isinstance(result, tuple) else (result,))]
    if len(values) != len(output_dtypes):
        raise ValueError(f"generated kernel returned {len(values)} outputs for {len(output_dtypes)} ABI dtypes")
    for index, (value, dtype) in enumerate(zip(values, output_dtypes, strict=True)):
        if dtype in _FP_DTYPES_NON_FP32:
            values[index] = value.astype(np.float32 if dtype == "tfloat32" else np.dtype(dtype)).astype(np.float32)
    return values[0] if len(values) == 1 else tuple(values)


@cache
def _custom_validator(source: str) -> Any:
    """Compile a caller-supplied NumPy validator once per worker."""
    namespace: dict[str, Any] = {"np": np}
    exec(compile(source, "<simulation-validator>", "exec"), namespace)  # noqa: S102
    return next(value for name, value in namespace.items() if name != "np" and callable(value))


def _assert_outputs(actual: ArrayResult, case: _SerializedCase) -> None:
    """Compare one simulated result with its reference outputs."""
    _index, label, _source, _name, inputs, expected, validation = case
    if validation is not None:
        _custom_validator(validation[0])(actual, expected, inputs, validation[1])
        return
    actuals, expecteds = (value if isinstance(value, tuple) else (value,) for value in (actual, expected))
    assert len(actuals) == len(expecteds), f"{label}: returned {len(actuals)} outputs, expected {len(expecteds)}"
    for pair in zip(actuals, expecteds, strict=True):
        np.testing.assert_allclose(*pair, atol=_WORKER_ATOL, rtol=_WORKER_RTOL, err_msg=label)


def _simulate_case(position: int) -> _FailurePayload | None:
    """Simulate one globally initialized case and return its failure."""
    case_index, label, source, func_name, inputs, _expected, _validation = case = _WORKER_CASES[position]
    try:
        try:
            _assert_outputs(_simulate_source_fp32(source, func_name, inputs), case)
        except AssertionError:
            _assert_outputs(_simulate_source_fp32(source, func_name, inputs, grouped=True), case)
    except Exception as error:
        return {
            "case_index": case_index,
            "label": label,
            "exception_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    return None


def _worker_result(
    cases: list[_SerializedCase], atol: float, rtol: float, worker_count: int
) -> dict[str, int | _FailurePayload | None]:
    """Simulate one assigned partition and return compact result metadata."""
    global _WORKER_ATOL, _WORKER_CASES, _WORKER_RTOL
    input_bytes = max((sum(value.nbytes for value in case[4].values()) for case in cases), default=0)
    for inputs in {id(case[4]): case[4] for case in cases}.values():
        inputs.update({n: v.astype("f4") for n, v in inputs.items() if v.dtype.kind == "f" and v.dtype != "f4"})
    cases.sort(key=lambda c: sum(16 ** (s.find("nisa") // 4) for s in c[2].splitlines() if "nisa." in s), reverse=True)
    _WORKER_CASES, _WORKER_ATOL, _WORKER_RTOL = cases, atol, rtol
    failures: list[_FailurePayload | None] = []
    if cases:
        active_workers = min(len(cases), worker_count, multiprocessing.cpu_count(), 16 if input_bytes > 1 << 30 else 32)
        if active_workers == 1:
            failures = [_simulate_case(position) for position in range(len(cases))]
        else:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=active_workers, mp_context=context) as executor:
                failures = list(executor.map(_simulate_case, range(len(cases)), chunksize=1))
    failure = min((item for item in failures if item is not None), key=lambda item: item["case_index"], default=None)
    return {"assigned": len(cases), "completed": len(cases) if failure is None else 0, "failure": failure}


def _run_worker(request_path: Path, result_path: Path) -> None:
    """Run one remote request and atomically write its result metadata."""
    with request_path.open("rb") as request:
        cases, atol, rtol, worker_count = cast(tuple[list[_SerializedCase], float, float, int], pickle.load(request))
    result, temporary_path = _worker_result(cases, atol, rtol, worker_count), result_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(result_path)


if __name__ == "__main__":
    _script, marker, request, result = sys.argv
    if marker != "--worker":
        raise ValueError(f"expected --worker, got {marker!r}")
    _run_worker(Path(request), Path(result))
