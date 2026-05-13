"""Unit tests for v2 place_buffers — N-D LCA-based shape derivation."""

import numpy as np

from nkigym.codegen.place_buffers import place_buffers
from nkigym.ir.build import build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul_small(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Single-tile matmul fixture: K=M=128, N=512 — one tile per dim."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(128, 128), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(128, 512), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(128, 512), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(128, 512), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


@nkigym_kernel
def _matmul_large(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Multi-tile matmul fixture: K=M=N=2048."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_SMALL_SPECS = {"lhs_T": {"shape": (128, 128), "dtype": "bfloat16"}, "rhs": {"shape": (128, 512), "dtype": "bfloat16"}}


_LARGE_SPECS = {
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}


def test_place_buffers_emits_3d_shape_for_2d_tensor_single_tile() -> None:
    """Single-tile config: shape collapses but num_P_tiles=1 stays explicit."""
    module = build_initial_ir(_matmul_small, _SMALL_SPECS)
    place_buffers(module)
    """lhs_T_sbuf is 2D (K=128, M=128): K tile=128, so num_K_tiles=1.
       M tile=128, so num_M_tiles=1. Shape = (128, 1, 128)."""
    lhs_T_sbuf = module.tensors["lhs_T_sbuf"]
    assert lhs_T_sbuf.shape == (128, 1, 128)


def test_place_buffers_emits_3d_shape_for_multi_tile() -> None:
    """Multi-tile config: shape = (P_tile, num_P_tiles, F_tile * num_F_tiles)."""
    module = build_initial_ir(_matmul_large, _LARGE_SPECS)
    place_buffers(module)
    """K=2048, tile=128 → num_K_tiles=16; M=2048, tile=128 → num_M_tiles=16.
       lhs_T_sbuf shape = (P_tile=128, num_K_tiles=16, M_tile * num_M_tiles = 2048)."""
    lhs_T_sbuf = module.tensors["lhs_T_sbuf"]
    assert lhs_T_sbuf.shape == (128, 16, 2048)


def test_place_buffers_leaves_param_tensors_alone() -> None:
    """Param tensors (HBM) have shape from input_specs; place_buffers mustn't touch."""
    module = build_initial_ir(_matmul_large, _LARGE_SPECS)
    original_lhs_shape = module.tensors["lhs_T"].shape
    original_rhs_shape = module.tensors["rhs"].shape
    place_buffers(module)
    assert module.tensors["lhs_T"].shape == original_lhs_shape
    assert module.tensors["rhs"].shape == original_rhs_shape


def test_place_buffers_leaves_return_hbm_alone() -> None:
    """Return HBM tensor shape set by canonical build from alloc — not touched."""
    module = build_initial_ir(_matmul_large, _LARGE_SPECS)
    hbm_out_before = module.tensors["hbm_out"].shape
    place_buffers(module)
    """HBM return tensors keep their declared 2D shape (output of the kernel)."""
    assert module.tensors["hbm_out"].shape == hbm_out_before


def test_place_buffers_psum_shape() -> None:
    """PSUM tensor gets 3D shape matching SBUF convention."""
    module = build_initial_ir(_matmul_large, _LARGE_SPECS)
    place_buffers(module)
    """psum_acc dims (M=2048, N=2048); M tile=128, N tile=512.
       num_M_tiles=16, num_N_tiles=4. Shape = (128, 16, 4 * 512) = (128, 16, 2048)."""
    psum_acc = module.tensors["psum_acc"]
    assert psum_acc.shape == (128, 16, 2048)


def test_place_buffers_sbuf_product_shape() -> None:
    """sbuf_prod (M, N) bf16 — same shape as psum_acc."""
    module = build_initial_ir(_matmul_large, _LARGE_SPECS)
    place_buffers(module)
    sbuf_prod = module.tensors["sbuf_prod"]
    assert sbuf_prod.shape == (128, 16, 2048)


def test_writer_driven_buffer_shape() -> None:
    """With NKILoad writing full-F and NKIMatmul reading per-M-tile, the
    SBUF buffer shape is (P_tile, num_P_slots, F_full_extent) — driven by
    the producer's write pattern, not the consumer's tile."""

    @nkigym_kernel
    def _k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        lhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
        NKILoad()(src=lhs, dst=lhs_sbuf)
        NKILoad()(src=rhs, dst=rhs_sbuf)
        NKIMemset(value=0.0)(dst=psum_acc)
        NKIMatmul()(stationary=lhs_sbuf, moving=rhs_sbuf, dst=psum_acc)
        NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    specs = {"lhs": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    km = build_initial_ir(_k, specs)
    place_buffers(km)
    """lhs_sbuf: writer is NKILoad with tile (P=128, F=2048-full).
    num_P_slots = 16 (covers 2048/128); F_tile * num_F_tiles = 2048 * 1."""
    assert km.tensors["lhs_sbuf"].shape == (128, 16, 2048)
    """rhs_sbuf: same story."""
    assert km.tensors["rhs_sbuf"].shape == (128, 16, 2048)
    """psum_acc: RMW writer is NKIMatmul with tile (M=128, N=512).
    num_P_slots = 16 (2048/128); F_tile * num_F_tiles = 512 * 4 = 2048."""
    assert km.tensors["psum_acc"].shape == (128, 16, 2048)
