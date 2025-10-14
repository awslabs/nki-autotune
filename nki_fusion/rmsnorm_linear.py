import neuronxcc.nki.language as nl
import numpy as np

from nki_fusion.axes import generate_parallel_axes_configs


def rmsnorm_matmul_golden(lhs, rhs, epsilon: float) -> np.ndarray:
    squares = lhs**2
    sum_of_squares = np.sum(squares, axis=-1, keepdims=False)
    square_mean = sum_of_squares / lhs.shape[-1]

    rms = np.sqrt(square_mean + epsilon)
    lhs_normalized = lhs / rms[:, None]
    result = np.matmul(lhs_normalized, rhs)
    return result


def test_rmsnorm_matmul_fusion():
    seq_len = 256
    hidden_dim = 1024
    output_dim = 512
    epsilon = 1e-6
    atol = 1e-5
    rtol = 1e-5
    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_K = nl.tile_size.pmax  # 128
    input_tensors = {"lhs": np.random.randn(seq_len, hidden_dim), "rhs": np.random.randn(hidden_dim, output_dim)}
    parallel_axes = [("lhs", 0, TILE_M), ("rhs", 1, TILE_N)]
    sequential_axes = [("lhs", 1, TILE_K), ("rhs", 0, TILE_K)]
    parallel_axes_configs = generate_parallel_axes_configs(input_tensors=input_tensors, parallel_axes=parallel_axes)
    for parallel_axis_config in parallel_axes_configs:
        print(parallel_axis_config)
    # parallel_axes = [Axis("lhs", 0, TILE_M), Axis("rhs", 1, TILE_N)]
    # sequential_axes = [Axis("lhs", 1, TILE_K), Axis("rhs", 0, TILE_K)]

    # chain = FusionChain()
    # chain.execute(input_tensors=input_tensors, parallel_axes=parallel_axes, sequential_axes=sequential_axes)


if __name__ == "__main__":
    test_rmsnorm_matmul_fusion()
