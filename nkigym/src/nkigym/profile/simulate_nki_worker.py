"""Standalone remote worker for batched fp32 NKI simulation."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import multiprocessing
import pickle
import re
import threading
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import nki
import numpy as np

_FP_DTYPES_NON_FP32 = (
    "bfloat16",
    "float16",
    "float8_e4m3",
    "float8_e4m3fn",
    "float8_e4m3fn_x4",
    "float8_e5m2",
    "float8_e5m2_x4",
    "float4_e2m1fn_x4",
    "tfloat32",
)
ArrayResult = np.ndarray | tuple[np.ndarray, ...]
_SerializedCase = tuple[int, str, str, str, dict[str, np.ndarray], ArrayResult]
_FailurePayload = dict[str, int | str]
_Provenance = dict[tuple[object, ...], dict[str, Any]]
_ProvenanceStore = dict[tuple[object, ...], _Provenance]
_WORKER_CASES: list[_SerializedCase] = []
_WORKER_ATOL = 0.0
_WORKER_RTOL = 0.0
_SIMULATION_LOCK = threading.Lock()


def _fp32_source(source: str) -> str:
    """Rewrite reduced-precision NKI language dtypes to fp32."""
    preserved = set(re.findall(r"\b(sbuf_semantic_\w+)\s*=", source))
    pairs = re.findall(r"nisa\.(?:dma_transpose|nc_matmul)\((?:src|stationary)=(\w+).*?(?:dst|moving)=(\w+)", source)
    preserved.update(right for left, right in pairs if left in preserved)
    for name in preserved:
        source = re.sub(rf"(\b{re.escape(name)}\s*=.*?\bdtype=)nl\.", rf"\1nkigym_nl.", source)
    for dtype in _FP_DTYPES_NON_FP32:
        source = re.sub(rf"\bnl\.{re.escape(dtype)}\b", "nl.float32", source)
    return _coalesce_dma_copies(source.replace("nkigym_nl.", "nl."))


def _node_names(node: ast.AST) -> set[str]:
    """Return names referenced below one Python AST node."""
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _coefficient(node: ast.AST, variable: str) -> int | None:
    """Return one variable's coefficient in a generated affine expression."""
    if variable not in _node_names(node):
        return 0
    if isinstance(node, ast.Name):
        return 1 if node.id == variable else None
    if not isinstance(node, ast.BinOp):
        return None
    if isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = _coefficient(node.left, variable), _coefficient(node.right, variable)
        if left is not None and right is not None:
            return left + right if isinstance(node.op, ast.Add) else left - right
    elif isinstance(node.op, ast.Mult):
        constant, expression = (
            (node.left, node.right) if isinstance(node.left, ast.Constant) else (node.right, node.left)
        )
        if isinstance(constant, ast.Constant) and isinstance(constant.value, int):
            coefficient = _coefficient(expression, variable)
            return None if coefficient is None else int(constant.value) * coefficient
    return None


class _ZeroLoop(ast.NodeTransformer):
    """Replace one collapsed loop variable with zero."""

    def __init__(self, name: str) -> None:
        """Store the loop variable removed by coalescing."""
        self.name = name

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Replace the collapsed loop variable."""
        return ast.copy_location(ast.Constant(value=0), node) if node.id == self.name else node


def _coalesced_operand(node: ast.expr, loop: tuple[str, int]) -> ast.expr | None:
    """Collapse one proven contiguous slice across one loop."""
    result, (name, extent) = copy.deepcopy(node), loop
    candidates = [item for item in ast.walk(result) if isinstance(item, ast.Slice) and name in _node_names(item)]
    if len(candidates) != 1:
        return None
    target, upper = candidates[0], candidates[0].upper
    if target.lower is None or target.step is not None or not isinstance(upper, ast.BinOp):
        return None
    if not isinstance(upper.op, ast.Add) or ast.dump(upper.left) != ast.dump(target.lower):
        return None
    if not isinstance(upper.right, ast.Constant) or not isinstance(upper.right.value, int) or upper.right.value <= 0:
        return None
    lower, expected = target.lower, int(upper.right.value)
    if _coefficient(lower, name) != expected or name in _node_names(result) - _node_names(target):
        return None
    expected *= extent
    target.lower = cast(ast.expr, _ZeroLoop(name).visit(lower))
    target.upper = ast.BinOp(left=copy.deepcopy(target.lower), op=ast.Add(), right=ast.Constant(value=expected))
    return result


def _loop_info(node: ast.For) -> tuple[str, int] | None:
    """Return one simple positive constant-range loop."""
    iterator = node.iter
    if not isinstance(node.target, ast.Name) or node.orelse or len(node.body) != 1:
        return None
    if not isinstance(iterator, ast.Call) or not isinstance(iterator.func, ast.Name) or iterator.func.id != "range":
        return None
    argument = iterator.args[0] if len(iterator.args) == 1 else None
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, int) or argument.value <= 0:
        return None
    return node.target.id, int(argument.value)


def _dma_copy(node: ast.stmt) -> ast.Call | None:
    """Return one direct NKI DMA-copy call."""
    call = node.value if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) else None
    function = None if call is None else call.func
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        return call if function.value.id == "nisa" and function.attr == "dma_copy" else None
    return None


class _CoalesceDMACopies(ast.NodeTransformer):
    """Collapse contiguous DMA-only loop nests for CPU simulation."""

    def visit_For(self, node: ast.For) -> ast.AST:
        """Replace one proven contiguous DMA loop nest."""
        node = cast(ast.For, self.generic_visit(node))
        loop, call = _loop_info(node), _dma_copy(node.body[0])
        if loop is None or call is None:
            return node
        bindings = {keyword.arg: keyword for keyword in call.keywords}
        if any(
            loop[0] in _node_names(keyword.value) for name, keyword in bindings.items() if name not in {"src", "dst"}
        ):
            return node
        source = _coalesced_operand(bindings["src"].value, loop) if "src" in bindings else None
        destination = _coalesced_operand(bindings["dst"].value, loop) if "dst" in bindings else None
        if source is None or destination is None:
            return node
        bindings["src"].value, bindings["dst"].value = source, destination
        return ast.copy_location(node.body[0], node)


def _coalesce_dma_copies(source: str) -> str:
    """Coalesce proven contiguous DMA loop nests in standalone simulator source."""
    tree = _CoalesceDMACopies().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _view_key(view: Any) -> tuple[object, ...]:
    """Return the simulator's identity for one PSUM view."""
    storage = view.tensor.__array_interface__["data"][0]
    if view._is_identity():
        return (view.tensor_id, storage)
    return (view.tensor_id, storage, view.offset, tuple(tuple(pattern) for pattern in view.pattern))


def _outer_fma(left: np.ndarray, right: np.ndarray, addend: np.ndarray) -> np.ndarray:
    """Evaluate one float32 outer-product FMA with one final rounding."""
    product = left.astype(np.float64)[:, None] * right.astype(np.float64)[None, :]
    return np.asarray(product + addend.astype(np.float64), dtype=np.float32)


def _mkl_gemv_result(stationary: list[np.ndarray], moving: list[np.ndarray]) -> np.ndarray:
    """Match the eight-term reduction tree used by MKL AVX-512 GEMV."""
    left, right = (np.concatenate(parts, axis=0) for parts in (stationary, moving))
    result = np.zeros((left.shape[1], right.shape[1]), dtype=np.float32)
    for start in range(0, left.shape[0], 8):
        tile_left, tile_right = left[start : start + 8], right[start : start + 8]
        products = np.asarray(tile_left[:, :, None] * tile_right[:, None, :], dtype=np.float32)
        pairs = [_outer_fma(tile_left[i], tile_right[i], products[j]) for i, j in ((5, 7), (0, 2), (1, 3))]
        carried = _outer_fma(tile_left[4], tile_right[4], _outer_fma(tile_left[6], tile_right[6], result))
        result = np.add(np.add(pairs[1], pairs[2]), np.add(carried, pairs[0]), dtype=np.float32)
    return result


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
    """Group three hardware contraction tiles per FP32 reference matmul."""
    from nki._backends import simulator
    from nki._backends.simulator.dtypes import to_numpy_dtype
    from nki._backends.simulator.matmul import _flatten_to_2d
    from nki._backends.simulator.state import get_current_context
    from nki._backends.simulator.tensor_view import SimulatorTensorView
    from nki.language import _ops as language_ops

    original_matmul, original_copy, original_tensor_tensor, original_get, original_set, original_reduce_op = (
        simulator.nc_matmul,
        simulator.tensor_copy,
        simulator.tensor_tensor_arith,
        SimulatorTensorView.get_data,
        SimulatorTensorView.set_data,
        language_ops.get_numpy_reduce_op,
    )
    pending: _ProvenanceStore = {}
    symbolic: _ProvenanceStore = {}
    materializing: set[tuple[object, ...]] = set()
    position_cache: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}

    def allocation(items: _ProvenanceStore, view: Any) -> _Provenance:
        """Return provenance views belonging to one allocation."""
        return items.get(_view_key(view)[:2], {})

    def view_indices(view: Any) -> np.ndarray:
        """Return flattened storage indices in view iteration order."""
        indices = np.arange(view.tensor.size, dtype=np.int64) if view._is_identity() else view._get_indices()
        return np.asarray(indices, dtype=np.int64).reshape(-1)

    def view_span(view: Any) -> tuple[int, int]:
        """Return inclusive storage bounds for one regular tensor view."""
        deltas = [int(step) * (int(count) - 1) for step, count in view._get_pattern()]
        offset = int(view.offset)
        return offset + sum(min(0, delta) for delta in deltas), offset + sum(max(0, delta) for delta in deltas)

    def remap_positions(view: Any, absolute: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map storage indices to positions in one target view."""
        key = _view_key(view)
        if key not in position_cache:
            indices = view_indices(view)
            order = np.argsort(indices, kind="stable")
            position_cache[key] = indices[order], order
        indices, order = position_cache[key]
        locations = np.searchsorted(indices, absolute, side="right") - 1
        safe = np.maximum(locations, 0)
        selected = (locations >= 0) & (indices[safe] == absolute)
        return selected, order[safe]

    def spans_overlap(left: Any, right: Any) -> bool:
        """Return whether two views in one allocation can overlap."""
        left_span, right_span = view_span(left), view_span(right)
        return left_span[0] <= right_span[1] and right_span[0] <= left_span[1]

    def grouped_result(stationary: list[np.ndarray], moving: list[np.ndarray]) -> np.ndarray:
        """Evaluate one logical contraction using the reference grouping."""
        result, count = np.zeros((stationary[0].shape[1], moving[0].shape[1]), dtype=np.float32), len(stationary)
        widths = stationary[0].shape[1], moving[0].shape[1]
        if count == 32 and min(widths) == 1 and max(widths) % 16 == 0:
            return _mkl_gemv_result(stationary, moving)
        group = {4: 4, 8: 3, 64: 1 + 2 * (widths[0] >= 128), 128: 2}.get(count, 1)
        for start in range(0, len(stationary), group):
            left, right = (np.concatenate(parts[start : start + group], axis=0) for parts in (stationary, moving))
            result += np.matmul(left.T, right)
        return result

    def flush(view: Any) -> None:
        """Write every pending subview overlapping one read or write."""
        for key, item in tuple(allocation(pending, view).items()):
            if not spans_overlap(view, item["view"]) or not item["dirty"]:
                continue
            result = grouped_result(item["stationary"], item["moving"])
            if item["base"] is not None:
                result = item["base"] + result
            view = item["view"]
            materializing.add(key)
            try:
                original_set(view, result.reshape(view.view_shape).astype(to_numpy_dtype(view.dtype)))
            finally:
                materializing.remove(key)
            item["dirty"] = False

    def flush_symbolic(view: Any) -> None:
        """Materialize every symbolic RFactor result in one allocation."""
        for key, item in tuple(allocation(symbolic, view).items()):
            if not spans_overlap(view, item["view"]) or not item["dirty"]:
                continue
            data = original_get(item["view"]).copy().reshape(-1)
            grouped: dict[bytes, list[dict[str, Any]]] = {}
            for panel in item["panels"]:
                grouped.setdefault(panel["positions"].tobytes(), []).append(panel)
            for panels in grouped.values():
                stationary = [array for panel in panels for array in panel["stationary"]]
                moving = [array for panel in panels for array in panel["moving"]]
                result = grouped_result(stationary, moving)
                bases = [panel["base"] for panel in panels if panel["base"] is not None]
                if bases:
                    result += sum(bases[1:], start=bases[0].copy())
                data[panels[0]["positions"]] = result.reshape(-1)
            materializing.add(key)
            try:
                original_set(item["view"], data.reshape(item["view"].view_shape))
            finally:
                materializing.remove(key)
            item["dirty"] = False

    def slice_panel(panel: dict[str, Any], columns: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
        """Slice one symbolic matmul panel to complete result columns."""
        base = None if panel["base"] is None else panel["base"][:, columns]
        moving = [array[:, columns] for array in panel["moving"]]
        return panel | {"base": base, "moving": moving, "positions": positions}

    def take_symbolic(view: Any) -> list[dict[str, Any]]:
        """Remove and remap symbolic subviews contained by ``view``."""
        matches = [(key, item) for key, item in allocation(symbolic, view).items() if spans_overlap(item["view"], view)]
        if not matches:
            return []
        panels: list[dict[str, Any]] = []
        for key, item in matches:
            item_indices = view_indices(item["view"])
            remaining = []
            for panel in item["panels"]:
                absolute = item_indices[panel["positions"]]
                selected, remapped = remap_positions(view, absolute)
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

    def set_data(view: Any, value: object) -> None:
        """Preserve pending subviews before an explicit write resets them."""
        if (key := _view_key(view)) not in materializing:
            candidates = [
                (pending_key, item)
                for pending_key, item in allocation(pending, view).items()
                if spans_overlap(view, item["view"])
            ]
            if candidates:
                flush(view)
                overwritten = set(view_indices(view))
                for pending_key, item in candidates:
                    if not overwritten.isdisjoint(view_indices(item["view"])):
                        del allocation(pending, view)[pending_key]
            allocation(symbolic, view).pop(key, None)
        original_set(view, value)

    def tensor_copy(dst: Any, src: Any, engine: object, name: object) -> None:
        """Copy values and retain matmul provenance across an RFactor drain."""
        panels = []
        matches = [item for item in allocation(pending, src).values() if spans_overlap(item["view"], src)]
        for item in matches:
            indices = view_indices(item["view"])
            selected, remapped = remap_positions(src, indices)
            selected = selected.reshape(item["view"].view_shape)
            columns = np.flatnonzero(selected[0])
            if (
                columns.size
                and np.all(selected[:, columns])
                and np.count_nonzero(selected) == selected.shape[0] * columns.size
            ):
                panel = {"base": item["base"], "moving": item["moving"], "stationary": list(item["stationary"])}
                panels.append(slice_panel(panel, columns, remapped[selected.reshape(-1)]))
        original_copy(dst, src, engine, name)
        if panels:
            key = _view_key(dst)
            symbolic.setdefault(key[:2], {})[key] = {"dirty": True, "panels": panels, "view": dst}

    def tensor_tensor(dst: Any, data1: Any, data2: Any, op: object, engine: object, name: object) -> None:
        """Preserve RFactor provenance across its in-place SBUF add."""
        destination = _view_key(dst)
        panels = []
        if str(op) == "add" and destination == _view_key(data1):
            right = take_symbolic(data2)
            if right:
                panels.extend(take_symbolic(data1))
                panels.extend(right)
        original_tensor_tensor(dst, data1, data2, op, engine, name)
        if panels:
            symbolic.setdefault(destination[:2], {})[destination] = {"dirty": True, "panels": panels, "view": dst}

    def matmul(
        dst: Any,
        stationary: Any,
        moving: Any,
        is_transpose: object,
        row_pos: object,
        col_pos: object,
        perf_mode: object,
        accumulate: object,
        name: object,
    ) -> None:
        """Record one contraction tile or delegate unsupported matmul modes."""
        if is_transpose or stationary.ti_state is not None or moving.ti_state is not None:
            original_matmul(dst, stationary, moving, is_transpose, row_pos, col_pos, perf_mode, accumulate, name)
            return
        left = _flatten_to_2d(original_get(stationary).astype(np.float32), perf_mode)
        right = _flatten_to_2d(original_get(moving).astype(np.float32), perf_mode)
        key, written = _view_key(dst), get_current_context().psum_written
        bucket = pending.setdefault(key[:2], {})
        should_accumulate = accumulate if accumulate is not None else key in written
        if accumulate is False or key not in bucket:
            bucket[key] = {
                "base": original_get(dst).copy() if should_accumulate else None,
                "dirty": False,
                "moving": [],
                "stationary": [],
                "view": dst,
            }
        bucket[key]["stationary"].append(left.copy())
        bucket[key]["moving"].append(right.copy())
        bucket[key]["dirty"] = True
        written[key] = True

    def numpy_reduce_op(operation: object) -> Any:
        """Use Torch's additive reduction order for complete free axes."""
        fallback = original_reduce_op(operation)
        if str(operation) != "add":
            return fallback

        def reduce(values: np.ndarray, axis: tuple[int, ...], keepdims: bool) -> np.ndarray:
            width = values.size // values.shape[0]
            if axis != tuple(range(1, values.ndim)) or width % 32:
                return cast(np.ndarray, fallback(values, axis=axis, keepdims=keepdims))
            result = _torch_add_reduce(values.reshape(values.shape[0], width))
            return result.reshape((values.shape[0],) + (1,) * (values.ndim - 1)) if keepdims else result[:, 0]

        return reduce

    with _SIMULATION_LOCK:
        language_ops.get_numpy_reduce_op = numpy_reduce_op
        simulator.nc_matmul, simulator.tensor_copy, simulator.tensor_tensor_arith = matmul, tensor_copy, tensor_tensor
        SimulatorTensorView.get_data, SimulatorTensorView.set_data = get_data, set_data
        try:
            yield
        finally:
            language_ops.get_numpy_reduce_op = original_reduce_op
            simulator.nc_matmul, simulator.tensor_copy, simulator.tensor_tensor_arith = (
                original_matmul,
                original_copy,
                original_tensor_tensor,
            )
            SimulatorTensorView.get_data, SimulatorTensorView.set_data = original_get, original_set


def _simulate_kernel_fp32(kernel: object, call: tuple[tuple[object, ...], dict[str, object]]) -> object:
    """Run one rewritten kernel with grouped tiled matmul reduction."""
    with _grouped_matmul_accumulation():
        return nki.simulate(kernel)(*call[0], **call[1])


def _simulate_source_fp32(source: str, func_name: str, inputs: dict[str, np.ndarray], grouped: bool) -> ArrayResult:
    """Execute one standalone rendered kernel through the fp32 simulator."""
    namespace: dict = {}
    exec(compile(_fp32_source(source), f"<batch-case-{func_name}>", "exec"), namespace)  # noqa: S102
    kernel = namespace.get(func_name)
    if kernel is None:
        raise AttributeError(f"rendered kernel has no function {func_name!r}")
    result = (
        _simulate_kernel_fp32(kernel, ((), cast(dict[str, object], inputs)))
        if grouped
        else nki.simulate(kernel)(**inputs)
    )
    return tuple(np.asarray(value) for value in result) if isinstance(result, tuple) else np.asarray(result)


def _assert_outputs(label: str, actual: ArrayResult, expected: ArrayResult) -> None:
    """Compare one simulated result with its reference outputs."""
    actuals = actual if isinstance(actual, tuple) else (actual,)
    expecteds = expected if isinstance(expected, tuple) else (expected,)
    assert len(actuals) == len(expecteds), f"{label}: returned {len(actuals)} outputs, expected {len(expecteds)}"
    for pair in zip(actuals, expecteds, strict=True):
        np.testing.assert_allclose(*pair, atol=_WORKER_ATOL, rtol=_WORKER_RTOL, err_msg=label)


def _simulate_case(position: int) -> _FailurePayload | None:
    """Simulate one globally initialized case and return its failure."""
    case_index, label, source, func_name, inputs, expected = _WORKER_CASES[position]
    try:
        try:
            _assert_outputs(label, _simulate_source_fp32(source, func_name, inputs, False), expected)
        except AssertionError:
            _assert_outputs(label, _simulate_source_fp32(source, func_name, inputs, True), expected)
    except Exception as error:
        return {
            "case_index": case_index,
            "label": label,
            "exception_type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    return None


def _read_worker_request(request_path: Path) -> tuple[list[_SerializedCase], float, float, int]:
    """Load one trusted controller-generated worker request."""
    with request_path.open("rb") as request_file:
        payload = cast(tuple[list[_SerializedCase], float, float, int], pickle.load(request_file))
    if not isinstance(payload, tuple) or len(payload) != 4:
        raise ValueError("malformed batch simulation request")
    cases, atol, rtol, worker_count = payload
    if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count <= 0:
        raise ValueError("batch simulation worker count must be positive")
    return cases, atol, rtol, worker_count


def _worker_result(
    cases: list[_SerializedCase], atol: float, rtol: float, worker_count: int
) -> dict[str, int | _FailurePayload | None]:
    """Simulate one assigned partition and return compact result metadata."""
    global _WORKER_ATOL, _WORKER_CASES, _WORKER_RTOL
    input_bytes = max((sum(value.nbytes for value in case[4].values()) for case in cases), default=0)
    for inputs in {id(case[4]): case[4] for case in cases}.values():
        for name, value in inputs.items():
            if value.dtype.kind == "f":
                inputs[name] = value.astype(np.float32)
    cases.sort(key=lambda case: len(case[2]), reverse=True)
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
    cases, atol, rtol, worker_count = _read_worker_request(request_path)
    result, temporary_path = _worker_result(cases, atol, rtol, worker_count), result_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(result_path)


def _parse_args() -> argparse.Namespace:
    """Parse the private remote-worker command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    return parser.parse_args()


def _main() -> int:
    """Run the private remote-worker entry point."""
    args = _parse_args()
    if not args.worker:
        raise ValueError("simulate_nki_worker.py is only executable in worker mode")
    _run_worker(args.request, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
