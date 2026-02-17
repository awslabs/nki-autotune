"""NKI Gym walkthrough: golden NumPy, nkigym simulation, tiling, and codegen.

Demonstrates:
1. Numerical equivalence between pure NumPy and nkigym GymOp simulation.
2. Converting a nkigym function to the GymProgram IR.
3. Calling the GymProgram directly as a numpy simulator.
4. Tiling a GymProgram into 128x128 tiles with reduction accumulation.
5. Rendering a tiled GymProgram as Python source code.
6. Simulating the tiled program and verifying numerical correctness.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

import nkigym
from nkigym.ir import program_to_source, source_to_program
from nkigym.tiling import tile_program
from nkigym.utils import setup_logging
from nkigym.utils.source import callable_to_source, source_to_callable

logger = logging.getLogger(__name__)


def golden_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Pure NumPy matrix multiplication (golden reference).

    Neuron matmul convention: lhs is [K, M], rhs is [K, N], output is [M, N].

    Args:
        lhs: Left-hand side tensor of shape [K, M].
        rhs: Right-hand side tensor of shape [K, N].

    Returns:
        Output tensor of shape [M, N].
    """
    return lhs.T @ rhs


def nkigym_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """NKI Gym matrix multiplication (NumPy simulation mode).

    Args:
        lhs: Left-hand side tensor of shape [K, M].
        rhs: Right-hand side tensor of shape [K, N].

    Returns:
        Output tensor of shape [M, N].
    """
    return nkigym.nc_matmul(lhs, rhs)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NKI Gym walkthrough example")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache"), help="Directory for storing output logs (default: cache)"
    )
    return parser.parse_args()


def main() -> None:
    """Run the full NKI Gym pipeline: parse, simulate, tile, codegen."""
    args = parse_args()
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path = cache_dir / "gym.log"
    setup_logging(str(log_path))

    k, m, n = 256, 256, 256
    rng = np.random.default_rng(42)
    lhs = rng.standard_normal((k, m)).astype(np.float64)
    rhs = rng.standard_normal((k, n)).astype(np.float64)
    atol = 1e-10
    rtol = 1e-10

    golden = golden_matmul(lhs, rhs)
    np.testing.assert_allclose(golden, nkigym_matmul(lhs, rhs), rtol=rtol, atol=atol)

    source = callable_to_source(nkigym_matmul)
    program = source_to_program(source, {"lhs": (k, m), "rhs": (k, n)}, np.float64)
    logger.info(program)
    program_source = program_to_source(program)
    np.testing.assert_allclose(golden, source_to_callable(program_source, program.name)(lhs, rhs), rtol=rtol, atol=atol)

    tiled = tile_program(program)
    logger.info(tiled)
    tiled_source = program_to_source(tiled)
    np.testing.assert_allclose(golden, source_to_callable(tiled_source, tiled.name)(lhs, rhs), rtol=rtol, atol=atol)


if __name__ == "__main__":
    main()
