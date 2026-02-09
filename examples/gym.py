"""NKI Gym tiling, reuse, and operand merge demo.

Demonstrates the full transform pipeline on a tiled matmul:

1. Generate a tiled function from a high-level NKI op.
2. Apply data reuse and operand merging interleaved, one atomic
   step at a time, saving each intermediate IR and verifying
   numerical correctness on the CPU numpy simulator.
"""

import os
import random
from pathlib import Path

import numpy as np

import nkigym
from nkigym.tiling import generate_tiled_function
from nkigym.transforms import DataReuseTransform, OperandMergeTransform
from nkigym.utils import get_source

CACHE_ROOT = "/fsx/weittang/gym_cache"


def matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Perform NKI matrix multiplication.

    Args:
        lhs: Left-hand side tensor of shape [K, M] (partition x free).
        rhs: Right-hand side tensor of shape [K, N] (partition x free).

    Returns:
        Output tensor of shape [M, N].
    """
    return nkigym.nc_matmul(lhs, rhs)


def main() -> None:
    """Run the tiling and interleaved transform demo."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    cache_path = Path(CACHE_ROOT)

    k, m, n = 256, 256, 512
    input_shapes: dict[str, tuple[int, int]] = {"lhs": (k, m), "rhs": (k, n)}

    lhs = np.random.randn(k, m)
    rhs = np.random.randn(k, n)
    expected = matmul(lhs, rhs)

    func = generate_tiled_function(matmul, input_shapes)
    np.testing.assert_allclose(func(lhs, rhs), expected)
    (cache_path / "nkigym_user.py").write_text(f'"""{matmul.__name__} (user source)"""\n' + get_source(matmul))
    (cache_path / "nkigym_matmul_0.py").write_text(f'"""step 0: tiled function (baseline)"""\n' + get_source(func))

    reuse = DataReuseTransform()
    merge = OperandMergeTransform()
    max_steps = 1

    for step in range(1, max_steps + 1):
        reuse_pairs = reuse.analyze(func)
        merge_opps = merge.analyze(func)

        candidates = [(reuse, p, f"data reuse ({p})") for p in reuse_pairs] + [
            (merge, o, f"operand merge ({o.description})") for o in merge_opps
        ]

        if not candidates:
            break

        transform, option, desc = random.choice(candidates)
        func = transform.transform(func, option)

        result = func(lhs, rhs)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

        filename = f"nkigym_matmul_{step}.py"
        (cache_path / filename).write_text(f'"""step {step}: {desc}"""\n' + get_source(func))


if __name__ == "__main__":
    main()
