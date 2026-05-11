"""Local fp32 CPU-sim verification for nkigym-rendered kernels.

Two helpers, both local (no remote round-trip, no autotune dependency):

* :func:`_verify` runs ``nki.simulate`` on the rendered kernel source
  at fp32 and compares element-wise against ``f_nkigym(**inputs)``.
  ``f_nkigym`` is executed directly — its ``NKIOp`` operations have
  pure-numpy ``__call__`` implementations (see
  ``nkigym.ops.base.NKIOp.__call__`` and per-op ``_run``), so the
  math-function itself serves as the golden reference.

* :func:`_verify_fns` compares two live callables at fp32. Used on the
  numpy-input branch of ``nkigym_compile`` to confirm synthesis
  produced a math-correct ``f_nkigym`` before spending cycles tuning
  it.

Both raise ``AssertionError`` on divergence outside ``atol=rtol=5e-3``.
"""

import re
from collections.abc import Callable

import nki
import numpy as np

_ATOL = 5e-3
_RTOL = 5e-3
_SIM_SEED = 0

_NL_DTYPE_NAME_RE = re.compile(r"^(?:b?float\d+(?:_e\d+m\d+(?:fn)?(?:_x\d+)?)?|tfloat\d+|int\d+|uint\d+|bool_?)$")
"""Regex matching NKI dtype identifiers: ``float32``, ``float16``,
``bfloat16``, ``float8_e4m3``, ``float8_e4m3fn_x4``, ``tfloat32``,
``int8``/``int16``/``int32``, ``uint8``/``uint16``/``uint32``, ``bool_``.
A whitelist of dtype names keeps the rewrite from corrupting
``nl.<op>`` activation / language-op tokens (``square``, ``rsqrt``,
``add``, ``multiply`` ...) or buffer tags (``sbuf``, ``psum``,
``hbm``, ``ndarray`` ...)."""

_NL_TOKEN_RE = re.compile(r"\bnl\.([A-Za-z_][A-Za-z0-9_]*)\b")


def _draw_fp32_inputs(input_specs: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, np.ndarray]:
    """Draw reproducible fp32 inputs shaped by ``input_specs``.

    Ignores declared dtypes — the CPU-sim contract is fp32 everywhere,
    matching the renderer's ``nl.bfloat16 -> nl.float32`` rewrite below.
    """
    rng = np.random.default_rng(_SIM_SEED)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _) in input_specs.items()}


def _rewrite_to_fp32(kernel_source: str) -> str:
    """Return ``kernel_source`` with every NKI dtype token forced to ``nl.float32``.

    The hardware path keeps the user's declared dtypes; the simulator
    runs fp32 end-to-end so accuracy is not dominated by low-precision
    rounding.

    Rewrites every ``nl.<name>`` token whose suffix matches
    :data:`_NL_DTYPE_NAME_RE`. The whitelist catches all present and
    future NKI dtypes (bf16, fp16, fp8_*, int*, uint*, ...) while
    leaving activation/language ops (``nl.square``, ``nl.rsqrt``,
    ``nl.add``, ...) and buffer tags (``nl.sbuf``, ``nl.psum``,
    ``nl.ndarray``, ...) untouched.
    """

    def replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        return "nl.float32" if _NL_DTYPE_NAME_RE.match(name) else match.group(0)

    return _NL_TOKEN_RE.sub(replace, kernel_source)


def _verify(
    kernel_source: str, f_nkigym: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> None:
    """Run ``kernel_source`` through ``nki.simulate`` and compare to ``f_nkigym``.

    Raises ``AssertionError`` if the element-wise diff exceeds
    ``atol=rtol=5e-3``.
    """
    sim_source = _rewrite_to_fp32(kernel_source)
    ns: dict = {}
    exec(sim_source, ns)
    kernel_fn = ns[f_nkigym.__name__]
    inputs = _draw_fp32_inputs(input_specs)
    actual = nki.simulate(kernel_fn)(**inputs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = f_nkigym(**inputs)
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    max_abs = float(np.abs(actual - expected).max())
    max_rel = float((np.abs(actual - expected) / (np.abs(expected) + _ATOL)).max())
    if not np.allclose(actual, expected, atol=_ATOL, rtol=_RTOL):
        raise AssertionError(
            f"kernel vs f_nkigym: max_abs={max_abs:.3e} max_rel={max_rel:.3e} (atol={_ATOL}, rtol={_RTOL})"
        )


def _verify_fns(
    f_nkigym: Callable[..., np.ndarray],
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    """Check that synthesised ``f_nkigym`` agrees with ``f_numpy`` at fp32.

    Raises ``AssertionError`` on divergence.
    """
    inputs = _draw_fp32_inputs(input_specs)
    nk = f_nkigym(**inputs)
    np_ = f_numpy(**inputs)
    if isinstance(nk, tuple):
        nk = nk[0]
    if isinstance(np_, tuple):
        np_ = np_[0]
    nk = np.asarray(nk)
    np_ = np.asarray(np_)
    max_abs = float(np.abs(nk - np_).max())
    max_rel = float((np.abs(nk - np_) / (np.abs(np_) + _ATOL)).max())
    if not np.allclose(nk, np_, atol=_ATOL, rtol=_RTOL):
        raise AssertionError(
            f"f_nkigym vs f_numpy: max_abs={max_abs:.3e} max_rel={max_rel:.3e} (atol={_ATOL}, rtol={_RTOL})"
        )
