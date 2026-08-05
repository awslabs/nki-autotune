"""Unseeded random-rollout numerical regression coverage."""

from __future__ import annotations

import importlib.util
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
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import (
    BufferCompaction,
    BufferLayout,
    CancelTransposePair,
    CodeMotion,
    CommonSubexpressionElimination,
    EliminateIdentityInitializer,
    Fuse,
    FuseBroadcastActivation,
    FusePointwiseActivation,
    FusePointwiseReduction,
    InsertTransposePair,
    OnlineFusion,
    Reorder,
    RFactor,
    SoftwarePipeline,
    Split,
    TransposeThroughLoad,
    TransposeThroughMatmul,
    TransposeThroughTensorCopy,
)

ATOL = 5e-3
RTOL = 5e-3
NUM_ROLLOUTS = 20
ROLLOUT_STEPS = 500
InputSpecs = dict[str, tuple[tuple[int, ...], str]]
TRANSFORMS = [
    BufferCompaction(),
    BufferLayout(),
    CancelTransposePair(),
    CodeMotion(),
    CommonSubexpressionElimination(),
    EliminateIdentityInitializer(),
    Fuse(),
    FuseBroadcastActivation(),
    FusePointwiseActivation(),
    FusePointwiseReduction(),
    InsertTransposePair(),
    OnlineFusion(),
    RFactor(),
    Reorder(),
    SoftwarePipeline(),
    Split(),
    TransposeThroughLoad(),
    TransposeThroughMatmul(),
    TransposeThroughTensorCopy(),
]


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


LHS_T_WORKLOAD = Workload(
    name="random_matmul_lhsT_rhs",
    input_specs={"lhs_T": ((512, 128), "bfloat16"), "rhs": ((512, 128), "bfloat16")},
    f_numpy=f_lhs_t_rhs_numpy,
    f_nkigym=f_matmul,
)
LHS_WORKLOAD = Workload(
    name="random_matmul_lhs_rhs",
    input_specs={"lhs": ((128, 512), "bfloat16"), "rhs": ((512, 128), "bfloat16")},
    f_numpy=f_lhs_rhs_numpy,
    f_nkigym=f_lhs_matmul,
)
ATTENTION_WORKLOAD = Workload(
    name="random_attention",
    input_specs={"query": ((128, 128), "bfloat16"), "key": ((128, 256), "bfloat16"), "value": ((256, 128), "bfloat16")},
    f_numpy=f_attention_numpy,
    f_nkigym=f_attention,
)
WORKLOADS = (LHS_T_WORKLOAD, LHS_WORKLOAD, ATTENTION_WORKLOAD)


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


def _rollout(workload: Workload, rollout: int, seed: int) -> Iterator[tuple[str, KernelIR]]:
    """Yield the initial state and every action selected by ``seed``."""
    environment = KernelMDP(workload.f_nkigym, workload.input_specs, transforms=TRANSFORMS)
    rng = random.Random(seed)
    prefix = f"rollout {rollout} seed {seed}"
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
    workload: Workload, rollout: int, seed: int, states: Iterator[tuple[str, KernelIR]], tmp_path: Path
) -> int:
    """Render every rollout state and compare it with NumPy."""
    inputs = _inputs(workload.input_specs, seed)
    expected = workload.f_numpy(**inputs)
    generated_name = f"nki_{workload.f_nkigym.__name__}"
    count = 0
    for index, (label, state) in enumerate(states):
        source = render(state)
        module_name = f"{workload.name}_rollout_{rollout}_state_{index}"
        module = _load_source(source, tmp_path, module_name)
        actual = np.asarray(simulate_fp32(getattr(module, generated_name))(**inputs))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, err_msg=label)
        count = index + 1
    return count


def test_random_rollouts_preserve_every_generated_kernel(tmp_path: Path) -> None:
    """Twenty 500-step random rollouts preserve every generated kernel."""
    seed_source = random.SystemRandom()
    for rollout in range(NUM_ROLLOUTS):
        seed = seed_source.randrange(1 << 63)
        workload = WORKLOADS[rollout % len(WORKLOADS)]
        print(f"{workload.name} rollout {rollout} seed {seed}", flush=True)
        count = _assert_states_match_numpy(workload, rollout, seed, _rollout(workload, rollout, seed), tmp_path)
        assert count == ROLLOUT_STEPS + 1
