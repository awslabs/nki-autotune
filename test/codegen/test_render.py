"""End-to-end render + CPU-sim numerics gate for the BlockNode IR refactor."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir

import numpy as np

from nkigym.codegen import render
from nkigym.synthesis.simulate_nki import simulate_fp32


def _load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("dumped_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_canonical_matmul_emits_expected_structure():
    """The rendered canonical kernel has the expected top-level shape."""
    ir = build_canonical_ir()
    src = render(ir)
    assert "@nki.jit" in src
    assert "def nki_f_matmul(lhs_T, rhs):" in src
    assert "psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)" in src
    assert "nisa.memset" in src
    assert "nisa.nc_matmul" in src
    assert "return hbm_out" in src.strip().splitlines()[-1]


def test_render_canonical_matmul_passes_numerics():
    """The rendered canonical kernel passes fp32 simulation against numpy."""
    ir = build_canonical_ir()
    src = render(ir)
    cache_dir = Path("/tmp/blocknode_render_test_canonical")
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True)
    kernel_path = cache_dir / "kernel.py"
    kernel_path.write_text(src)
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    module = _load_module_from_path(str(kernel_path))
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
