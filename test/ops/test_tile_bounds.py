"""Regression tests for MIN_TILE_SIZE / MAX_TILE_SIZE class attrs on NKIOp."""

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
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


def test_operation_tile_bounds():
    """Every operation declares the expected ISA tile interval."""
    pf_min = {"P": 128, "F": 128}
    cases = {
        NKIMatmul: ({"K": 128, "M": 128, "N": 128}, {"K": 128, "M": 128, "N": 512}),
        NKITranspose: (pf_min, {"P": 128, "F": 128}),
        NKIDMATranspose: (pf_min, {"P": 128, "F": None}),
        NKILoad: (pf_min, {"P": 128, "F": None}),
        NKIStore: (pf_min, {"P": 128, "F": None}),
        NKIMemset: (pf_min, {"P": 128, "F": None}),
        NKITensorCopy: (pf_min, {"P": 128, "F": None}),
        NKITensorReduce: (pf_min, {"P": 128, "F": None}),
        NKIActivation: (pf_min, {"P": 128, "F": None}),
        NKIActivationReduce: (pf_min, {"P": 128, "F": None}),
        NKITensorScalar: (pf_min, {"P": 128, "F": None}),
    }
    for op_cls, (minimum, maximum) in cases.items():
        assert op_cls.MIN_TILE_SIZE == minimum, op_cls.__name__
        assert op_cls.MAX_TILE_SIZE == maximum, op_cls.__name__
