"""Tests for ``nkigym_compile`` and the tune-stage run_tune driver.

The synthesis stage is expensive (calls the Claude Agent SDK) and
non-deterministic, so these tests bypass it by seeding
``f_nkigym.py`` on disk and invoking the post-synthesis helpers
directly.
"""

from pathlib import Path

import numpy as np
import pytest

from nkigym.compile import _cpu_sim_check, _load_f_nkigym, _run_initial_codegen
from nkigym.tune.fuse_loops import FuseLoops
from nkigym.tune.stage import run_tune


def _rmsnorm_matmul_numpy(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Plain-numpy ``rmsnorm(lhs) @ rhs`` golden."""
    m = np.mean(np.square(lhs), axis=1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(m + 1e-6)
    return (lhs * rms_inv) @ rhs


_F_NKIGYM_SOURCE = """\
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.transpose import NKITranspose
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce


@nkigym_kernel
def f_nkigym(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    sum_sq = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    rms_inv = NKIActivation(op="rsqrt", scale=1 / 256, bias=1e-6)(data=sum_sq)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out
"""


_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}


def _seed_f_nkigym(tmp_path: Path) -> None:
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)


def _explicit_run_tune(tmp_path: Path, rewrites: list) -> None:
    """Invoke run_tune on the explicit path with stub batch kwargs."""
    run_tune(
        f_numpy=_rmsnorm_matmul_numpy,
        input_specs=_SPECS,
        cache_path=tmp_path,
        rewrites=rewrites,
        seed=0,
        load_f_nkigym=_load_f_nkigym,
        cpu_sim_check=_cpu_sim_check,
        num_kernels=0,
        hosts=[],
        venv_python="",
        neuron_platform_target="",
        collect_detailed_profile=False,
        trace_output_shape=lambda f, s: (0,),
        assert_no_cpu_sim_failures=lambda p: None,
        atol=5e-3,
        rtol=5e-3,
    )


def test_initial_codegen_runs_cpu_sim_and_writes_kernel(tmp_path: Path) -> None:
    """initial_codegen produces kernel.py and auto-validates it against the numpy golden."""
    _seed_f_nkigym(tmp_path)
    _run_initial_codegen(_rmsnorm_matmul_numpy, _SPECS, tmp_path)
    kernel_path = tmp_path / "kernel.py"
    assert kernel_path.exists()
    kernel_source = kernel_path.read_text()
    assert "@nki.jit" in kernel_source
    assert "def f_nkigym(lhs, rhs):" in kernel_source


def test_cpu_sim_mismatch_raises(tmp_path: Path) -> None:
    """If the rendered kernel doesn't match the numpy golden, the check raises."""
    _seed_f_nkigym(tmp_path)

    def bogus_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return lhs @ rhs[:, : rhs.shape[1]] + 1.0

    with pytest.raises(AssertionError, match="CPU-sim mismatch"):
        _run_initial_codegen(bogus_golden, _SPECS, tmp_path)


def test_run_tune_explicit_empty_rewrites_writes_kernel_tuned(tmp_path: Path) -> None:
    """Explicit path with rewrites=[] writes kernel_tuned.py and CPU-sim succeeds."""
    _seed_f_nkigym(tmp_path)
    _explicit_run_tune(tmp_path, rewrites=[])
    assert (tmp_path / "kernel_tuned.py").exists()


def test_run_tune_explicit_applies_fuse_loops_d0(tmp_path: Path) -> None:
    """Applying FuseLoops on activation_reduce↔activation d0 still matches numpy."""
    _seed_f_nkigym(tmp_path)
    rewrites = [FuseLoops(path=(), boundary=(2, 3), dim_id="d0")]
    _explicit_run_tune(tmp_path, rewrites=rewrites)
    kernel_source = (tmp_path / "kernel_tuned.py").read_text()
    assert "nisa.activation_reduce(" in kernel_source
    assert "nisa.activation(" in kernel_source


def test_run_tune_explicit_rejects_illegal_rewrite(tmp_path: Path) -> None:
    """A rewrite that fails is_legal raises ValueError before producing any artifact."""
    _seed_f_nkigym(tmp_path)
    bogus = FuseLoops(path=(), boundary=(99, 100), dim_id="d0")
    with pytest.raises(ValueError, match="illegal"):
        _explicit_run_tune(tmp_path, rewrites=[bogus])
    assert not (tmp_path / "kernel_tuned.py").exists()


def test_run_tune_explicit_applies_reorder_loops_inside_tensor_scalar(tmp_path: Path) -> None:
    """An explicit ReorderLoops on tensor_scalar's inner chain still matches numpy."""
    from nkigym.tune.reorder_loops import ReorderLoops

    _seed_f_nkigym(tmp_path)
    rewrites = [ReorderLoops(path=(4, 0), outer_dim="d0", inner_dim="d1")]
    _explicit_run_tune(tmp_path, rewrites=rewrites)
    assert (tmp_path / "kernel_tuned.py").exists()


def test_run_tune_explicit_compose_reorder_then_fuse(tmp_path: Path) -> None:
    """Reorder exposes a new fusion boundary; composed pipeline still CPU-sim-passes."""
    from nkigym.tune.reorder_loops import ReorderLoops

    _seed_f_nkigym(tmp_path)
    rewrites = [
        ReorderLoops(path=(4, 0), outer_dim="d0", inner_dim="d1"),
        FuseLoops(path=(), boundary=(3, 4), dim_id="d0"),
    ]
    _explicit_run_tune(tmp_path, rewrites=rewrites)
    assert (tmp_path / "kernel_tuned.py").exists()


def test_nkigym_compile_batch_path_mocks_remote_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch path calls remote_profile with one KernelJob per sampled kernel."""
    import json

    from nkigym import nkigym_compile
    from nkigym.tune import stage as stage_mod

    _seed_f_nkigym(tmp_path)

    captured: dict = {}

    def _fake_remote_profile(**kwargs):
        captured.update(kwargs)
        """Write a minimal results.json mimicking remote_profile's schema."""
        results = {
            "metadata": {"num_kernels": len(kwargs["kernels"]), "wallclock_s": 0.0, "hosts": []},
            "metrics": {},
            "kernels": [
                {"kernel_name": name, "cpu_sim_passed": True, "hardware_output": ""} for name in kwargs["kernels"]
            ],
        }
        (tmp_path / "results.json").write_text(json.dumps(results))

    """Also stub _run_synthesis so the seeded f_nkigym.py isn't overwritten."""

    def _noop_synthesis(f_numpy, input_specs, cache_path):
        pass

    monkeypatch.setattr(stage_mod, "remote_profile", _fake_remote_profile)
    monkeypatch.setattr("nkigym.compile._run_synthesis", _noop_synthesis)

    nkigym_compile(
        f_numpy=_rmsnorm_matmul_numpy,
        input_specs=_SPECS,
        cache_dir=tmp_path,
        num_kernels=3,
        hosts=["gym-1"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        seed=0,
    )

    names = sorted(captured["kernels"].keys())
    assert names == ["kernel_tuned_0000.py", "kernel_tuned_0001.py", "kernel_tuned_0002.py"]
    for name, job in captured["kernels"].items():
        assert job.func_name == "f_nkigym"
        assert job.input_specs == _SPECS
        assert job.nkigym_func_name == "f_nkigym"
        assert (tmp_path / name).exists()
    assert captured["neuron_platform_target"] == "trn2"
    assert captured["venv_python"] == "/home/ubuntu/venvs/kernel-env/bin/python"
    assert captured["hosts"] == ["gym-1"]


def test_nkigym_compile_batch_raises_on_cpu_sim_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single failing kernel in results.json surfaces as AssertionError."""
    import json

    from nkigym import nkigym_compile
    from nkigym.tune import stage as stage_mod

    _seed_f_nkigym(tmp_path)

    def _fake_remote_profile(**kwargs):
        results = {
            "metadata": {"num_kernels": len(kwargs["kernels"]), "wallclock_s": 0.0, "hosts": []},
            "metrics": {},
            "kernels": [
                {"kernel_name": name, "cpu_sim_passed": (idx != 1), "hardware_output": ""}
                for idx, name in enumerate(sorted(kwargs["kernels"]))
            ],
        }
        (tmp_path / "results.json").write_text(json.dumps(results))

    def _noop_synthesis(f_numpy, input_specs, cache_path):
        pass

    monkeypatch.setattr(stage_mod, "remote_profile", _fake_remote_profile)
    monkeypatch.setattr("nkigym.compile._run_synthesis", _noop_synthesis)

    with pytest.raises(AssertionError, match="kernel_tuned_0001.py"):
        nkigym_compile(
            f_numpy=_rmsnorm_matmul_numpy,
            input_specs=_SPECS,
            cache_dir=tmp_path,
            num_kernels=3,
            hosts=["gym-1"],
            venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
            neuron_platform_target="trn2",
            seed=0,
        )
