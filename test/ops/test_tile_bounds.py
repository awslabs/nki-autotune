"""Regression tests for MIN_TILE_SIZE / MAX_TILE_SIZE class attrs on NKIOp."""

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import NKIOp
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose


def test_nkiop_has_min_and_max_tile_size_dicts():
    """Base class exposes empty defaults so subclasses only override what they need."""
    assert hasattr(NKIOp, "MIN_TILE_SIZE")
    assert hasattr(NKIOp, "MAX_TILE_SIZE")
    assert NKIOp.MIN_TILE_SIZE == {}
    assert NKIOp.MAX_TILE_SIZE == {}


def test_matmul_bounds():
    assert NKIMatmul.MIN_TILE_SIZE == {"K": 128, "M": 128, "N": 128}
    assert NKIMatmul.MAX_TILE_SIZE == {"K": 128, "M": 128, "N": 512}


def test_transpose_bounds():
    assert NKITranspose.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKITranspose.MAX_TILE_SIZE == {"P": 128, "F": 128}


def test_dma_transpose_bounds():
    assert NKIDMATranspose.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIDMATranspose.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_load_bounds():
    assert NKILoad.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKILoad.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_store_bounds():
    assert NKIStore.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIStore.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_memset_bounds():
    assert NKIMemset.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIMemset.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_tensor_copy_bounds():
    assert NKITensorCopy.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKITensorCopy.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_tensor_reduce_bounds():
    assert NKITensorReduce.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKITensorReduce.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_activation_bounds():
    assert NKIActivation.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIActivation.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_activation_reduce_bounds():
    assert NKIActivationReduce.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIActivationReduce.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_tensor_scalar_bounds():
    assert NKITensorScalar.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKITensorScalar.MAX_TILE_SIZE == {"P": 128, "F": None}


def test_alloc_bounds():
    """NKIAlloc is just another ISA op: MIN 128 per axis, unbounded max on both."""
    assert NKIAlloc.MIN_TILE_SIZE == {"P": 128, "F": 128}
    assert NKIAlloc.MAX_TILE_SIZE == {"P": None, "F": None}
