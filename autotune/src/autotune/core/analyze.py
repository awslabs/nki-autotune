# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
import textwrap
from collections.abc import Callable
from typing import Any

_NDARRAY_SHAPE = re.compile(r"(\w+)\s*=\s*\w+\.ndarray\(shape=\((\d+),\s*(\d+)\)")
_NC_MATMUL = re.compile(r"nc_matmul\(\w+,\s*(\w+),\s*(\w+)")


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
        stat, mov = shapes.get(m.group(1)), shapes.get(m.group(2))
        if stat and mov:
            total += stat[0] * stat[1] * mov[1]

    return total
