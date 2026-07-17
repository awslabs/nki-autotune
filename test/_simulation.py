"""Shared helpers for rendered-kernel simulation tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np

from nkigym.codegen import render
from nkigym.ir import KernelIR
from nkigym.synthesis.simulate_nki import simulate_fp32


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


def assert_matmul_source_simulates(source: str, tmp_path: Path, module_name: str) -> None:
    """Assert that one rendered matmul kernel matches NumPy in the fp32 simulator."""
    module = _load_source(source, tmp_path, module_name)
    kernels = [getattr(module, name) for name in dir(module) if name.startswith("nki_f")]
    if len(kernels) != 1:
        raise AssertionError(f"expected one rendered kernel, found {len(kernels)}")

    rng = np.random.default_rng(0)
    inputs = {
        "lhs_T": rng.standard_normal((2048, 2048)).astype(np.float32),
        "rhs": rng.standard_normal((2048, 2048)).astype(np.float32),
    }
    actual = np.asarray(simulate_fp32(kernels[0])(**inputs))
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)


def assert_matmul_ir_simulates(ir: KernelIR, tmp_path: Path, module_name: str) -> None:
    """Render a matmul IR and assert that it matches NumPy in the fp32 simulator."""
    assert_matmul_source_simulates(render(ir), tmp_path, module_name)
