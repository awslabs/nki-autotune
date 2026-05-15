"""Wrap ``nki.simulate`` so the kernel and its inputs run end-to-end in fp32."""

import inspect
import re
import textwrap
from collections.abc import Callable

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


def simulate_fp32(kernel: Callable) -> Callable:
    """Wrap a NKI kernel so ``nki.simulate`` runs in ``nl.float32`` end-to-end.

    Rewrites every reduced-precision floating dtype (bfloat16, float16,
    float8_*, float4_*, tfloat32) referenced as ``nl.<dtype>`` in the
    kernel source to ``nl.float32``, then re-execs the rewritten source in
    a copy of the kernel's original module globals. The returned wrapper
    casts each numpy input tensor to ``np.float32`` before invoking
    ``nki.simulate`` so the simulator sees fp32 end-to-end.
    """
    func = kernel.func if hasattr(kernel, "func") else kernel
    source = textwrap.dedent(inspect.getsource(func))
    for dtype in _FP_DTYPES_NON_FP32:
        source = re.sub(rf"\bnl\.{re.escape(dtype)}\b", "nl.float32", source)
    ns: dict = dict(func.__globals__)
    exec(source, ns)  # noqa: S102
    sim = nki.simulate(ns[func.__name__])

    def wrapper(*args, **kwargs):
        """Cast numpy inputs to fp32 and forward to the simulated kernel."""
        cast_args = tuple(a.astype(np.float32) if isinstance(a, np.ndarray) else a for a in args)
        cast_kwargs = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in kwargs.items()}
        return sim(*cast_args, **cast_kwargs)

    return wrapper
