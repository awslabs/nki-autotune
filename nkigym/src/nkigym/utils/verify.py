"""CPU-sim verification: rendered source vs ``f_nkigym`` at fp32.

Both sides run fp32 end-to-end — the rendered source has every
``nl.bfloat16`` / ``nl.float16`` dtype rewritten to ``nl.float32``
before ``exec``, and the nkigym CPU-sim path already materialises
bf16 buffers as fp32 (see ``nkigym.ops.alloc._DTYPE_MAP``).

``verify`` reports the ``np.testing.assert_allclose`` worst-element
margin — ``|diff| / (atol + rtol * |desired|)`` at the worst entry —
so values close to but below the threshold remain legible.
"""

from collections.abc import Callable

import nki
import numpy as np


def verify(
    source: str,
    f_nkigym: Callable[..., np.ndarray],
    input_specs: dict[str, dict],
    seed: int = 0,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> None:
    """Run ``source`` through ``nki.simulate`` and compare to ``f_nkigym``.

    Args:
        source: Rendered NKI kernel source (result of
            ``nkigym.codegen.render.render``).
        f_nkigym: The ``@nkigym_kernel``-decorated function whose
            rendered form is ``source``. Used as the fp32 golden.
        input_specs: ``{name: {"shape": (...), "dtype": str}}`` per
            kernel parameter.
        seed: Seed for the fp32 input draw.
        atol: Absolute tolerance for ``np.testing.assert_allclose``.
        rtol: Relative tolerance for ``np.testing.assert_allclose``.

    Raises:
        AssertionError: On any element-wise divergence outside
            ``atol + rtol * |desired|``.
    """
    rng = np.random.default_rng(seed)
    inputs = {name: rng.standard_normal(spec["shape"]).astype(np.float32) for name, spec in input_specs.items()}
    sim_source = source.replace("nl.bfloat16", "nl.float32").replace("nl.float16", "nl.float32")
    ns: dict = {}
    exec(sim_source, ns)
    kernel_fn = ns[f_nkigym.__name__]
    actual = nki.simulate(kernel_fn)(**inputs)
    if isinstance(actual, tuple):
        actual = actual[0]
    actual = np.asarray(actual).astype(np.float64)
    expected = np.asarray(f_nkigym(**inputs)).astype(np.float64)

    abs_diff = np.abs(actual - expected)
    threshold = atol + rtol * np.abs(expected)
    ratio = abs_diff / threshold
    worst_idx = int(np.argmax(ratio))
    worst_diff = float(abs_diff.flat[worst_idx])
    worst_thresh = float(threshold.flat[worst_idx])
    worst_margin = float(ratio.flat[worst_idx])
    summary = (
        f"worst_margin={worst_margin:.4f} "
        f"(|diff|={worst_diff:.2e}, atol+rtol*|desired|={worst_thresh:.2e}, "
        f"atol={atol}, rtol={rtol})"
    )
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=summary)
    print(f"[verify] PASS: {summary}")
