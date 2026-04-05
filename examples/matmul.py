"""Matrix multiplication: numpy reference vs nkigym simulation.

Demonstrates that nkigym nc_matmul produces identical results
to numpy at float64 precision, then renders the naive NKI kernel.
"""

from pathlib import Path

import numpy as np

import nkigym

SAVE_DIR = Path("/home/ubuntu/cache/matmul_test")


def matmul_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply with numpy: a.T @ b.

    Args:
        a: Stationary tensor of shape (K, M).
        b: Moving tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    return a.T @ b


def matmul_nkigym(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply using nkigym logical ops.

    Args:
        a: Stationary tensor of shape (K, M).
        b: Moving tensor of shape (K, N).

    Returns:
        Output tensor of shape (M, N).
    """
    output = nkigym.nc_matmul(a, b)
    return output


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    K, M, N = 2048, 2048, 2048
    a = rng.standard_normal((K, M))
    b = rng.standard_normal((K, N))

    out_np = matmul_numpy(a, b)
    out_gym = matmul_nkigym(a, b)

    print(f"a: {a.shape}  b: {b.shape}")
    print(f"numpy  sample: {out_np[0, :4]}")
    print(f"nkigym sample: {out_gym[0, :4]}")
    max_diff = np.max(np.abs(out_np - out_gym))
    print(f"max |diff|: {max_diff:.2e}")
    np.testing.assert_allclose(out_gym, out_np, rtol=1e-10, atol=1e-10)
    print("PASS: matmul nkigym matches numpy")

    kernel_src = nkigym.render(matmul_nkigym, a=a, b=b)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / "matmul_kernel.py"
    out_path.write_text(kernel_src)
    print(f"\nSaved NKI kernel to {out_path}")
