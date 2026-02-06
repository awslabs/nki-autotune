"""NKI workload specification for dimension analysis.

This demonstrates how to analyze NKI functions for tiling using
the analyze_dimension function from nkigym.
"""

import os
import shutil
from pathlib import Path

import numpy as np
from nkipy.runtime import BaremetalExecutor

import nkigym
from autotune.core.compile import TensorStub, compile_kernel, create_spike_kernel, run_spike_kernel
from nkigym.lower import lower_gym_to_nki
from nkigym.tiling import generate_tiled_function
from nkigym.transforms import analyze_data_reuse, merge_reusable_tensors
from nkigym.utils import get_source

CACHE_ROOT = "/fsx/weittang/gym_cache"


def golden_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Reference matmul matching NKI nc_matmul semantics.

    Args:
        lhs: Left-hand side array of shape [K, M].
        rhs: Right-hand side array of shape [K, N].

    Returns:
        Result array of shape [M, N].
    """
    return np.matmul(lhs.T, rhs)


def matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Perform NKI matrix multiplication.

    Args:
        lhs: Left-hand side tensor of shape [K, M] (partition x free).
        rhs: Right-hand side tensor of shape [K, N] (partition x free).

    Returns:
        Output tensor of shape [M, N].
    """
    return nkigym.nc_matmul(lhs, rhs)


def run_nki_kernel(
    kernel_file: Path,
    kernel_func_name: str,
    input_tensors: dict[str, np.ndarray],
    output_shapes: dict[str, tuple[int, ...]],
    output_dir: Path,
) -> tuple[np.ndarray, ...]:
    """Run a compiled NKI kernel and return its output.

    Args:
        kernel_file: Path to the NKI kernel .py file.
        kernel_func_name: Name of the kernel function.
        input_tensors: Dict mapping input names to numpy arrays.
        output_shapes: Dict mapping output names to shapes.
        output_dir: Directory for compilation artifacts.

    Returns:
        Tuple of output numpy arrays.
    """
    dtype = next(iter(input_tensors.values())).dtype
    kernel_name = (str(kernel_file), kernel_func_name)
    output_tensors = [(name, shape, dtype) for name, shape in output_shapes.items()]

    neff_path = compile_kernel(
        kernel_name=kernel_name,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        kernel_kwargs={},
        target_instance_family="trn2",
        compiler_flags="--auto-cast=none",
        output_dir=str(output_dir),
    )

    output_stubs = [TensorStub(shape=shape, dtype=dtype, name=name) for name, shape in output_shapes.items()]

    with BaremetalExecutor(verbose=0) as spike:
        spike_kernel = create_spike_kernel(
            neff_path=neff_path,
            kernel_name=kernel_name,
            input_tensors=input_tensors,
            output_tensors=output_stubs,
            kernel_kwargs={},
        )
        _, kernel_outputs = run_spike_kernel(
            spike=spike, spike_kernel=spike_kernel, input_tensors=input_tensors, neff=neff_path, kernel_kwargs={}
        )

    return kernel_outputs


def main() -> None:
    """Run the tiling and data reuse analysis demo."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    cache_path = Path(CACHE_ROOT)

    k, m, n = 256, 256, 512
    input_shapes: dict[str, tuple[int, int]] = {"lhs": (k, m), "rhs": (k, n)}

    lhs = np.random.randn(k, m)
    rhs = np.random.randn(k, n)
    expected = golden_matmul(lhs, rhs)

    (cache_path / "nkigym_matmul.py").write_text(get_source(matmul))
    np.testing.assert_allclose(matmul(lhs, rhs), expected)
    print("matmul matches golden")

    tiled_matmul = generate_tiled_function(matmul, input_shapes)
    (cache_path / "tiled_matmul.py").write_text(get_source(tiled_matmul))
    np.testing.assert_allclose(tiled_matmul(lhs, rhs), expected)
    print("tiled_matmul matches golden")

    groups = analyze_data_reuse(tiled_matmul)
    for i, group in enumerate(groups):
        tiled_matmul = merge_reusable_tensors(tiled_matmul, group[0], group[1])
        np.testing.assert_allclose(tiled_matmul(lhs, rhs), expected)
        print(f"merged_matmul (pass {i + 1}) matches golden")
    (cache_path / "transformed_matmul.py").write_text(get_source(tiled_matmul))

    nki_source = lower_gym_to_nki(tiled_matmul)
    (cache_path / "nki_matmul.py").write_text(nki_source)

    output_dir = cache_path / "nki_tiled_matmul"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    nki_outputs = run_nki_kernel(
        kernel_file=cache_path / "nki_matmul.py",
        kernel_func_name="nki_tiled_matmul",
        input_tensors={"lhs": lhs.astype(np.float32), "rhs": rhs.astype(np.float32)},
        output_shapes={"output": (m, n)},
        output_dir=output_dir,
    )
    np.testing.assert_allclose(nki_outputs[0], expected, rtol=1e-3, atol=1e-3)
    print("NKI kernel output matches golden")


if __name__ == "__main__":
    main()
