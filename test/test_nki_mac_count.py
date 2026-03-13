"""Tests for MAC counting via NKIKernel.mac_count property."""

import numpy as np

import nkigym
from nkigym.codegen import codegen
from nkigym.ir.tensor import TensorRef
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.matmul import NKIMatmul


def matmul_tanh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """256x256 matmul + tanh test workload."""
    c = nkigym.nc_matmul(a, b)
    return nkigym.activation(c, op=np.tanh)


_SHAPE = (256, 256)
_KWARGS = {"a": np.zeros(_SHAPE, dtype=np.float32), "b": np.zeros(_SHAPE, dtype=np.float32)}


class TestMacCount:
    """MAC counting from NKIKernel property."""

    def test_matmul_mac_count(self) -> None:
        """256x256 matmul: K=256, M=256, N=256 => 256*256*256 MACs."""
        kernel = codegen(matmul_tanh, _KWARGS)
        assert kernel.mac_count == 256 * 256 * 256

    def test_activation_zero_macs(self) -> None:
        """Activation ops contribute 0 MACs."""
        kernel = codegen(matmul_tanh, _KWARGS)
        expected_matmul_macs = 256 * 256 * 256
        assert kernel.mac_count == expected_matmul_macs


class TestNKIMatmulMacCount:
    """Instance-level mac_count tests for NKIMatmul."""

    def test_128x64x32(self) -> None:
        """K=128, M=64, N=32 => 128*64*32 MACs."""
        stat = TensorRef("s", (128, 64), ((0, 128), (0, 64)))
        mov = TensorRef("m", (128, 32), ((0, 128), (0, 32)))
        dst = TensorRef("d", (64, 32), ((0, 64), (0, 32)))
        stmt = NKIMatmul(dst=dst, stationary=stat, moving=mov)
        assert stmt.mac_count() == 128 * 64 * 32

    def test_alloc_zero_macs(self) -> None:
        """Non-compute ops return 0 MACs."""
        stmt = NKIAlloc(dst="t", shape=(128, 128), dtype="nl.float32", buffer="nisa.sbuf")
        assert stmt.mac_count() == 0
