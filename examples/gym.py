"""NKI Gym search: combinatorial and graph-based schedule exploration.

Demonstrates the schedule search pipeline: define a user function
with ``nkigym.<op>(...)`` calls, and the search automatically parses
dimensions, enumerates or samples schedules, renders NKI kernels,
compiles, and benchmarks on hardware.

Supports two workloads (matmul, attention) and two search modes
(enumerate, graph).
"""

import argparse
import logging
from pathlib import Path

import numpy as np

import nkigym
from nkigym.search import graph_search, search


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply: stationary @ moving.

    Args:
        a: Stationary tensor of shape [K, M].
        b: Moving tensor of shape [K, N].

    Returns:
        Output tensor of shape [M, N].
    """
    c = nkigym.nc_matmul(a, b)
    return c


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal single-head attention: softmax(mask(Q @ K^T)) @ V.

    Follows the design guide section 1 math function definition.
    Pre-scales Q by 1/sqrt(d_k) before the matmul so that the
    scale factor is absorbed into the input tensor.

    Args:
        Q: Query tensor of shape [seq_q, d_k] (pre-scaled by 1/sqrt(d_k)).
        K: Key tensor of shape [seq_k, d_k].
        V: Value tensor of shape [seq_k, d_v].

    Returns:
        Output tensor of shape [seq_q, d_v].
    """
    Q_t = nkigym.transpose(Q)
    K_t = nkigym.transpose(K)
    S = nkigym.nc_matmul(Q_t, K_t)
    masked_S = nkigym.affine_select(S, cmp_op="greater_equal", on_false_value=-np.inf, channel_multiplier=1, step=-1)
    max_S = nkigym.tensor_reduce(masked_S, op=np.maximum)
    shifted_S = nkigym.tensor_scalar(masked_S, max_S, op0=np.subtract)
    exp_S = nkigym.activation(shifted_S, op=np.exp)
    sum_exp = nkigym.tensor_reduce(exp_S, op=np.add)
    inv_sum = nkigym.activation(sum_exp, op="reciprocal")
    exp_S_t = nkigym.transpose(exp_S)
    attn = nkigym.nc_matmul(exp_S_t, V)
    output = nkigym.tensor_scalar(attn, inv_sum, op0=np.multiply)
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NKI Gym search example")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for storing output artifacts")
    parser.add_argument("--remote-config", type=Path, default=None, help="Path to remote.json config file")
    parser.add_argument("--local", action="store_true", help="Run benchmarks locally instead of via SSH")
    parser.add_argument("--workload", choices=["matmul", "attention"], default="matmul", help="Workload to tune")
    parser.add_argument("--search-mode", choices=["enumerate", "graph"], default="enumerate", help="Search strategy")
    parser.add_argument("--num-targets", type=int, default=99999, help="Maximum number of variants to benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (0 for system entropy)")
    return parser.parse_args()


def _build_matmul_kwargs(rng: np.random.Generator) -> tuple[object, dict[str, np.ndarray]]:
    """Build matmul workload function and input kwargs.

    Args:
        rng: Numpy random generator.

    Returns:
        Tuple of (func, kernel_kwargs).
    """
    a = rng.standard_normal((2048, 2048)).astype(np.float16)
    b = rng.standard_normal((2048, 2048)).astype(np.float16)
    return matmul, {"a": a, "b": b}


def _build_attention_kwargs(rng: np.random.Generator) -> tuple[object, dict[str, np.ndarray]]:
    """Build attention workload function and input kwargs.

    Pre-scales Q by 1/sqrt(d_k) so the scale is absorbed into the input.

    Args:
        rng: Numpy random generator.

    Returns:
        Tuple of (func, kernel_kwargs).
    """
    scale = 1.0 / np.sqrt(128)
    q = (rng.standard_normal((4096, 128)) * scale).astype(np.float16)
    k = rng.standard_normal((4096, 128)).astype(np.float16)
    v = rng.standard_normal((4096, 128)).astype(np.float16)
    return attention, {"Q": q, "K": k, "V": v}


def main() -> None:
    """Run schedule search on the selected workload."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    remote_config = None
    if not args.local:
        if args.remote_config is None:
            raise SystemExit("--remote-config is required for distributed mode (or use --local)")
        remote_config = args.remote_config

    rng = np.random.default_rng(args.seed if args.seed != 0 else None)
    builders = {"matmul": _build_matmul_kwargs, "attention": _build_attention_kwargs}
    func, kernel_kwargs = builders[args.workload](rng)

    search_fn = graph_search if args.search_mode == "graph" else search
    search_fn(
        func=func,
        num_targets=args.num_targets,
        seed=args.seed,
        save_cache=args.cache_dir,
        kernel_kwargs=kernel_kwargs,
        remote_config=remote_config,
    )


if __name__ == "__main__":
    main()
