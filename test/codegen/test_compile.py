"""Tests for ``nkigym_compile`` — staged synthesis → codegen driver."""

from pathlib import Path

import numpy as np
import pytest

from nkigym import nkigym_compile
from nkigym.tune.fuse_loops import FuseLoops


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
from nkigym.ops.activation_reduce import NKIActivationReduce


@nkigym_kernel
def f_nkigym(lhs, rhs):
    lhs_sbuf = NKILoad()(data=lhs)
    rhs_sbuf = NKILoad()(data=rhs)
    rms_inv = NKIActivationReduce(op="square", reduce_op="add", post_op="rsqrt", scale=1 / 256, bias=1e-6)(
        data=lhs_sbuf
    )
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    lhs_T = NKITranspose()(data=lhs_rms)
    prod = NKIMatmul()(stationary=lhs_T, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out
"""


def test_initial_codegen_runs_cpu_sim_and_writes_kernel(tmp_path: Path) -> None:
    """initial_codegen produces kernel.py and auto-validates it against the numpy golden."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["initial_codegen"])
    kernel_path = tmp_path / "kernel.py"
    assert kernel_path.exists()
    kernel_source = kernel_path.read_text()
    assert "@nki.jit" in kernel_source
    assert "def f_nkigym(lhs, rhs):" in kernel_source


def test_initial_codegen_without_synthesis_artifact_raises(tmp_path: Path) -> None:
    """Running initial_codegen without f_nkigym.py on disk raises ValueError."""
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    with pytest.raises(ValueError, match="initial_codegen requires"):
        nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["initial_codegen"])


def test_unknown_stage_raises(tmp_path: Path) -> None:
    """An unrecognised stage name is rejected before any work runs."""
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    with pytest.raises(ValueError, match="Unknown stage"):
        nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["mystery_pass"])


def test_cpu_sim_mismatch_raises(tmp_path: Path) -> None:
    """If the rendered kernel doesn't match the numpy golden, the check raises."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}

    def bogus_golden(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Deliberately wrong golden to force a mismatch."""
        return lhs @ rhs[:, : rhs.shape[1]] + 1.0

    with pytest.raises(AssertionError, match="CPU-sim mismatch"):
        nkigym_compile(bogus_golden, specs, tmp_path, stages=["initial_codegen"])


def test_tune_stage_with_empty_rewrites_writes_kernel_tuned_file(tmp_path: Path) -> None:
    """`tune` with `rewrites=[]` writes kernel_tuned.py and CPU-sim succeeds."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], rewrites=[])
    assert (tmp_path / "kernel_tuned.py").exists()


def test_tune_stage_applies_fuse_loops_d0(tmp_path: Path) -> None:
    """Applying FuseLoops on activation_reduce↔tensor_scalar d0 still matches numpy."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    rewrites = [FuseLoops(path=(), boundary=(2, 3), dim_id="d0")]
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], rewrites=rewrites)
    kernel_source = (tmp_path / "kernel_tuned.py").read_text()
    assert "nisa.activation_reduce(" in kernel_source
    assert "nisa.tensor_scalar(" in kernel_source


def test_tune_stage_rejects_illegal_rewrite(tmp_path: Path) -> None:
    """A rewrite that fails is_legal raises ValueError before producing any artifact."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    bogus = FuseLoops(path=(), boundary=(99, 100), dim_id="d0")
    with pytest.raises(ValueError, match="illegal"):
        nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], rewrites=[bogus])
    assert not (tmp_path / "kernel_tuned.py").exists()


def test_tune_stage_random_draw_is_reproducible(tmp_path: Path) -> None:
    """Running tune twice with rewrites=None + same seed produces identical output."""
    (tmp_path / "f_nkigym.py").write_text(_F_NKIGYM_SOURCE)
    specs = {"lhs": ((128, 256), "bfloat16"), "rhs": ((256, 512), "bfloat16")}
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], seed=42)
    first = (tmp_path / "kernel_tuned.py").read_text()
    nkigym_compile(_rmsnorm_matmul_numpy, specs, tmp_path, stages=["tune"], seed=42)
    second = (tmp_path / "kernel_tuned.py").read_text()
    assert first == second
