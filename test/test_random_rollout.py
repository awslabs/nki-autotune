"""Combinatorial random-rollout correctness coverage."""

from __future__ import annotations

import importlib.util
import os
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from nkigym.codegen import render
from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.profile import simulate_fp32
from nkigym.transforms import public_transforms

ATOL = 5e-3
RTOL = 5e-3
ROLLOUT_STEPS = 500
SHAPES_PER_WORKLOAD = 5
TILE_SIZE = NKIMatmul.MIN_TILE_SIZE["K"]
MAX_EXTENT_TILES = 8
MAX_MATMUL_TILE_VOLUME = 60
MAX_ATTENTION_TILE_AREA = 20
EPSILON = 1e-6
RUN_SEED_ENVIRONMENT = "NKI_RANDOM_ROLLOUT_SEED"
InputSpecs = dict[str, tuple[tuple[int, ...], str]]
TRANSFORMS = public_transforms()


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    """Return the canonical SSA graph for ``lhs_T.T @ rhs``."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


@nkigym_kernel
def f_lhs_matmul(lhs, rhs):
    """Return the canonical SSA graph for ``lhs @ rhs``."""
    sbuf_lhs = NKILoad()(src=lhs)
    psum_lhs_T = NKITranspose()(data=sbuf_lhs)
    sbuf_lhs_T = NKITensorCopy()(src=psum_lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


@nkigym_kernel
def f_attention(query, key, value):
    """Return the canonical SSA graph for scaled dot-product attention."""
    sbuf_query = NKILoad()(src=query)
    sbuf_key = NKILoad()(src=key)
    psum_scores = NKIMatmul()(stationary=sbuf_query, moving=sbuf_key)
    sbuf_scores = NKITensorCopy()(src=psum_scores)
    sbuf_scaled_scores = NKITensorScalar(op0="multiply")(data=sbuf_scores, operand0=128**-0.5)
    sbuf_row_max = NKITensorReduce(op="maximum", axis=1)(data=sbuf_scaled_scores)
    sbuf_centered = NKITensorScalar(op0="subtract")(data=sbuf_scaled_scores, operand0=sbuf_row_max)
    sbuf_exp = NKIActivation(op="exp")(data=sbuf_centered)
    sbuf_row_sum = NKITensorReduce(op="add", axis=1)(data=sbuf_exp)
    sbuf_inv_sum = NKIActivation(op="reciprocal")(data=sbuf_row_sum)
    sbuf_probability = NKITensorScalar(op0="multiply")(data=sbuf_exp, operand0=sbuf_inv_sum)
    sbuf_probability_t = NKIDMATranspose()(src=sbuf_probability)
    sbuf_value = NKILoad()(src=value)
    psum_output = NKIMatmul()(stationary=sbuf_probability_t, moving=sbuf_value)
    sbuf_output = NKITensorCopy()(src=psum_output)
    hbm_output = NKIStore()(src=sbuf_output)
    return hbm_output


def _rmsnorm_matmul_kernel(k: int) -> Callable[..., np.ndarray]:
    """Build an RMSNorm+matmul graph with the scale for one K extent."""
    scale = 1.0 / k

    @nkigym_kernel
    def f_rmsnorm_matmul(lhs, rhs):
        """Return row-wise RMSNorm followed by matmul."""
        sbuf_lhs = NKILoad()(src=lhs)
        sbuf_square_sum = NKIActivationReduce(op="square", reduce_op="add")(data=sbuf_lhs)
        sbuf_rms_inverse = NKIActivation(op="rsqrt", scale=scale, bias=EPSILON)(data=sbuf_square_sum)
        sbuf_normalized = NKITensorScalar(op0="multiply")(data=sbuf_lhs, operand0=sbuf_rms_inverse)
        sbuf_normalized_T = NKIDMATranspose()(src=sbuf_normalized)
        sbuf_rhs = NKILoad()(src=rhs)
        psum_output = NKIMatmul()(stationary=sbuf_normalized_T, moving=sbuf_rhs)
        sbuf_output = NKITensorCopy()(src=psum_output)
        hbm_output = NKIStore()(src=sbuf_output)
        return hbm_output

    return f_rmsnorm_matmul


@dataclass(frozen=True)
class Workload:
    """Kernel fixture and NumPy reference for one rollout workload."""

    name: str
    input_specs: InputSpecs
    f_numpy: Callable[..., np.ndarray]
    f_nkigym: Callable[..., np.ndarray]


def f_lhs_t_rhs_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the NumPy reference for ``lhs_T.T @ rhs``."""
    return lhs_T.T @ rhs


def f_lhs_rhs_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the NumPy reference for ``lhs @ rhs``."""
    return lhs @ rhs


def f_attention_numpy(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Return the NumPy reference for scaled dot-product attention."""
    scores = query.T @ key / np.sqrt(128)
    scores -= np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities @ value


def f_rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the NumPy reference for row-wise RMSNorm followed by matmul."""
    lhs_fp32 = lhs.astype(np.float32)
    normalized = lhs_fp32 / np.sqrt(np.mean(np.square(lhs_fp32), axis=1, keepdims=True) + EPSILON)
    return normalized @ rhs.astype(np.float32)


def _lhs_t_workload(k: int, m: int, n: int) -> Workload:
    """Build one pretransposed matmul workload."""
    return Workload(
        name=f"random_matmul_lhsT_rhs_k{k}_m{m}_n{n}",
        input_specs={"lhs_T": ((k, m), "bfloat16"), "rhs": ((k, n), "bfloat16")},
        f_numpy=f_lhs_t_rhs_numpy,
        f_nkigym=f_matmul,
    )


def _lhs_workload(k: int, m: int, n: int) -> Workload:
    """Build one row-major matmul workload."""
    return Workload(
        name=f"random_matmul_lhs_rhs_k{k}_m{m}_n{n}",
        input_specs={"lhs": ((m, k), "bfloat16"), "rhs": ((k, n), "bfloat16")},
        f_numpy=f_lhs_rhs_numpy,
        f_nkigym=f_lhs_matmul,
    )


def _attention_workload(query_length: int, sequence_length: int) -> Workload:
    """Build one attention workload."""
    return Workload(
        name=f"random_attention_q{query_length}_s{sequence_length}",
        input_specs={
            "query": ((128, query_length), "bfloat16"),
            "key": ((128, sequence_length), "bfloat16"),
            "value": ((sequence_length, 128), "bfloat16"),
        },
        f_numpy=f_attention_numpy,
        f_nkigym=f_attention,
    )


def _rmsnorm_matmul_workload(k: int, m: int, n: int) -> Workload:
    """Build one RMSNorm+matmul workload."""
    return Workload(
        name=f"random_rmsnorm_matmul_k{k}_m{m}_n{n}",
        input_specs={"lhs": ((m, k), "bfloat16"), "rhs": ((k, n), "bfloat16")},
        f_numpy=f_rmsnorm_matmul_numpy,
        f_nkigym=_rmsnorm_matmul_kernel(k),
    )


def _run_seed() -> int:
    """Return a fresh run seed unless an explicit replay seed is configured."""
    configured = os.environ.get(RUN_SEED_ENVIRONMENT)
    if configured is None:
        seed = random.SystemRandom().randrange(1 << 63)
    else:
        try:
            seed = int(configured, 0)
        except ValueError as error:
            raise ValueError(f"{RUN_SEED_ENVIRONMENT} must be an integer") from error
        if seed < 0 or seed >= 1 << 63:
            raise ValueError(f"{RUN_SEED_ENVIRONMENT} must be in [0, {1 << 63})")
    return seed


def _matmul_n_tile_factors() -> tuple[int, ...]:
    """Return N factors accepted by the canonical matmul tile contract."""
    maximum = NKIMatmul.MAX_TILE_SIZE["N"]
    if maximum is None:
        factors = tuple(range(1, MAX_EXTENT_TILES + 1))
    else:
        factors = tuple(
            factor
            for factor in range(1, MAX_EXTENT_TILES + 1)
            if (factor * TILE_SIZE) % min(factor * TILE_SIZE, maximum) == 0
        )
    return factors


def _random_matmul_shapes(rng: random.Random, count: int) -> tuple[tuple[int, int, int], ...]:
    """Sample unique aligned K, M, N shapes within the existing cost envelope."""
    shapes: list[tuple[int, int, int]] = []
    n_factors = _matmul_n_tile_factors()
    while len(shapes) < count:
        factors = (rng.randint(1, MAX_EXTENT_TILES), rng.randint(1, MAX_EXTENT_TILES), rng.choice(n_factors))
        shape = (factors[0] * TILE_SIZE, factors[1] * TILE_SIZE, factors[2] * TILE_SIZE)
        if factors[0] * factors[1] * factors[2] <= MAX_MATMUL_TILE_VOLUME and shape not in shapes:
            shapes.append(shape)
    return tuple(shapes)


def _random_attention_shapes(rng: random.Random, count: int) -> tuple[tuple[int, int], ...]:
    """Sample unique aligned query and sequence lengths within the cost envelope."""
    shapes: list[tuple[int, int]] = []
    sequence_factors = _matmul_n_tile_factors()
    while len(shapes) < count:
        factors = (rng.randint(1, MAX_EXTENT_TILES), rng.choice(sequence_factors))
        shape = (factors[0] * TILE_SIZE, factors[1] * TILE_SIZE)
        if factors[0] * factors[1] <= MAX_ATTENTION_TILE_AREA and shape not in shapes:
            shapes.append(shape)
    return tuple(shapes)


def _random_workloads(rng: random.Random) -> tuple[Workload, ...]:
    """Build five randomized shapes for each of the four workload families."""
    lhs_t_shapes = _random_matmul_shapes(rng, SHAPES_PER_WORKLOAD)
    lhs_shapes = _random_matmul_shapes(rng, SHAPES_PER_WORKLOAD)
    attention_shapes = _random_attention_shapes(rng, SHAPES_PER_WORKLOAD)
    rmsnorm_shapes = _random_matmul_shapes(rng, SHAPES_PER_WORKLOAD)
    workloads = (
        *(_lhs_t_workload(k, m, n) for k, m, n in lhs_t_shapes),
        *(_lhs_workload(k, m, n) for k, m, n in lhs_shapes),
        *(_attention_workload(query_length, sequence_length) for query_length, sequence_length in attention_shapes),
        *(_rmsnorm_matmul_workload(k, m, n) for k, m, n in rmsnorm_shapes),
    )
    return workloads


def _load_source(source: str, tmp_path: Path, module_name: str) -> ModuleType:
    """Load rendered kernel source as a temporary Python module."""
    path = tmp_path / f"{module_name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load rendered kernel from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rollout(workload: Workload, seed: int) -> Iterator[tuple[str, KernelIR]]:
    """Yield the initial state and every action selected by ``seed``."""
    environment = KernelMDP(workload.f_nkigym, workload.input_specs, transforms=TRANSFORMS)
    rng = random.Random(seed)
    prefix = f"{workload.name} seed {seed}"
    state = environment.reset()
    yield (f"{prefix} step 0", state)
    for step in range(1, ROLLOUT_STEPS + 1):
        actions = environment.legal_actions(state)
        if not actions:
            raise AssertionError(f"{prefix} terminated after {step - 1} steps")
        action = rng.choice(actions)
        state = environment.step(state, action)
        label = f"{prefix} step {step}: {type(action[0]).__name__} {action[1]!r}"
        yield (label, state)


def _inputs(input_specs: InputSpecs, seed: int) -> dict[str, np.ndarray]:
    """Create replayable fp32 inputs for one rollout workload."""
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in input_specs.items()}


def _assert_states_match_numpy(
    workload: Workload, seed: int, states: Iterator[tuple[str, KernelIR]], tmp_path: Path
) -> int:
    """Render every rollout state and compare it with NumPy."""
    inputs = _inputs(workload.input_specs, seed)
    expected = workload.f_numpy(**inputs)
    generated_name = f"nki_{workload.f_nkigym.__name__}"
    count = 0
    for index, (label, state) in enumerate(states):
        source = render(state)
        module_name = f"{workload.name}_state_{index}"
        module = _load_source(source, tmp_path, module_name)
        actual = np.asarray(simulate_fp32(getattr(module, generated_name))(**inputs))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, err_msg=label)
        count = index + 1
    return count


def test_one_random_rollout_per_randomized_shape_preserves_every_generated_kernel(tmp_path: Path) -> None:
    """One 500-step rollout per newly sampled shape preserves every generated kernel."""
    run_seed = _run_seed()
    rng = random.Random(run_seed)
    print(f"{RUN_SEED_ENVIRONMENT}={run_seed}", flush=True)
    workloads = _random_workloads(rng)
    for workload in workloads:
        seed = rng.randrange(1 << 63)
        print(f"{workload.name} seed {seed}", flush=True)
        count = _assert_states_match_numpy(workload, seed, _rollout(workload, seed), tmp_path)
        assert count == ROLLOUT_STEPS + 1
