"""End-to-end render + CPU-sim numerics gate for the BlockNode IR refactor."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir
from test.transforms._ladder_compare import assert_matches_render_ordered

import numpy as np

from nkigym.codegen import render
from nkigym.synthesis.simulate_nki import simulate_fp32


def _load_module_from_path(path: str):
    spec = importlib.util.spec_from_file_location("dumped_kernel", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load generated module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_canonical_matmul_emits_expected_structure():
    """The rendered canonical kernel has the expected top-level shape."""
    ir = build_canonical_ir()
    src = render(ir)
    assert "@nki.jit" in src
    assert "def nki_f_matmul(lhs_T, rhs):" in src
    assert "psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]" in src
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


_KERNEL_0_REFERENCE = """
@nki.jit
def kernel_0(lhs_T, rhs):
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(src=lhs_T[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0:0 + 2048])
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(src=rhs[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0:0 + 2048])
    psum_prod = [nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0][0:128, i_d1_0, 0:0 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(stationary=sbuf_lhs_T[0][0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128], moving=sbuf_rhs[0][0:128, i_d0_0, i_d2_0 * 512:i_d2_0 * 512 + 512], dst=psum_prod[0][0:128, i_d1_0, i_d2_0 * 512:i_d2_0 * 512 + 512])
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0][0:128, i_d1_0, 0:0 + 2048], dst=sbuf_prod[0][0:128, i_d1_0, 0:0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(src=sbuf_prod[0][0:128, i_d1_0, 0:0 + 2048], dst=hbm_out[i_d1_0 * 128:i_d1_0 * 128 + 128, 0:0 + 2048])
    return hbm_out
"""


def test_render_canonical_decls_interleaved_before_first_use():
    """Each buffer decl is emitted immediately before the first loop that uses it
    (the kernel_0 interleaving), not clustered at the top of the function."""
    ir = build_canonical_ir()
    assert_matches_render_ordered(render(ir), _KERNEL_0_REFERENCE)
