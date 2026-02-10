# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
import textwrap
from collections.abc import Callable
from typing import Any

_NDARRAY_SHAPE = re.compile(r"(\w+)\s*=\s*\w+\.ndarray\(shape=\((\d+),\s*(\d+)\)")
_NC_MATMUL = re.compile(
    r"nc_matmul\(\w+(?:\[.*?\])?,\s*"
    r"(\w+)(?:\[(\d+):(\d+),\s*(\d+):(\d+)\])?,\s*"
    r"(\w+)(?:\[(\d+):(\d+),\s*(\d+):(\d+)\])?"
)


def _unwrap_kernel(kernel: Callable) -> Callable:
    """Unwrap @nki.jit or similar decorators to get the raw Python function.

    Args:
        kernel: The kernel function, possibly wrapped by @nki.jit.

    Returns:
        The underlying unwrapped function.
    """
    func = kernel
    if hasattr(func, "func"):
        func = func.func
    elif hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def _resolve_operand_shape(
    name: str, slice_groups: tuple[str | None, str | None, str | None, str | None], shapes: dict[str, tuple[int, int]]
) -> tuple[int, int] | None:
    """Resolve the effective shape of an nc_matmul operand.

    If a 2D slice subscript is present (e.g., tensor[0:128, 128:256]),
    computes shape from (end - start) per dimension. Otherwise falls back
    to the declared ndarray shape.

    Args:
        name: The buffer variable name.
        slice_groups: Four regex groups (row_start, row_end, col_start, col_end),
            each None when no slice is present.
        shapes: Mapping of variable names to declared (rows, cols) shapes.

    Returns:
        A (rows, cols) tuple, or None if the shape cannot be resolved.
    """
    row_start, row_end, col_start, col_end = slice_groups
    if row_start is not None:
        return (int(row_end) - int(row_start), int(col_end) - int(col_start))
    return shapes.get(name)


def compute_mac_count(kernel: Callable, kernel_kwargs: dict[str, Any]) -> int:
    """Compute total MAC count by statically analyzing kernel source for nc_matmul calls.

    Parses the kernel source to find nl.ndarray shape declarations and
    nc_matmul(dst, stationary, moving) calls. For each call with
    stationary[K, M] and moving[K, N], the MAC count is K * M * N.

    Args:
        kernel: The NKI kernel function (may be @nki.jit-wrapped).
        kernel_kwargs: All kernel arguments (unused, kept for API compat).

    Returns:
        Total MAC count across all nc_matmul calls. Returns 0 if no
        nc_matmul calls are found or operand shapes cannot be resolved.
    """
    func = _unwrap_kernel(kernel)
    source = textwrap.dedent(inspect.getsource(func))

    shapes: dict[str, tuple[int, int]] = {}
    for m in _NDARRAY_SHAPE.finditer(source):
        shapes[m.group(1)] = (int(m.group(2)), int(m.group(3)))

    total = 0
    for m in _NC_MATMUL.finditer(source):
        stat = _resolve_operand_shape(m.group(1), (m.group(2), m.group(3), m.group(4), m.group(5)), shapes)
        mov = _resolve_operand_shape(m.group(6), (m.group(7), m.group(8), m.group(9), m.group(10)), shapes)
        if stat and mov:
            total += stat[0] * stat[1] * mov[1]

    return total
