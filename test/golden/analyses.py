"""Shared analysis and op-call fixtures for schedule transform tests."""

import numpy as np

from nkigym.codegen.analysis import _Analysis, _OpCall
from nkigym.ops.activation import NKIActivation
from nkigym.ops.add import NKIAdd
from nkigym.ops.matmul import NKIMatmul

MATMUL_256_ANALYSIS = _Analysis(
    var_dims={"a": ("d0", "d1"), "b": ("d0", "d3"), "c": ("d1", "d3")},
    var_shapes={"a": (256, 256), "b": (256, 256), "c": (256, 256)},
    parallel_dims=["d1", "d3"],
    reduction_dims=["d0"],
    tile_counts={"d1": 2, "d3": 1},
    reduction_tile_counts={"d0": 2},
    dim_tile_sizes={"d0": 128, "d1": 128, "d3": 256},
    return_var="c",
)
MATMUL_256_OP_CALLS = [_OpCall(NKIMatmul, ("a", "b"), (), "c")]
MATMUL_256_PARAMS = ("a", "b")


MATMUL_RECT_ANALYSIS = _Analysis(
    var_dims={"a": ("d0", "d1"), "b": ("d0", "d3"), "c": ("d1", "d3")},
    var_shapes={"a": (512, 256), "b": (512, 1024), "c": (256, 1024)},
    parallel_dims=["d1", "d3"],
    reduction_dims=["d0"],
    tile_counts={"d1": 2, "d3": 2},
    reduction_tile_counts={"d0": 4},
    dim_tile_sizes={"d0": 128, "d1": 128, "d3": 512},
    return_var="c",
)
MATMUL_RECT_OP_CALLS = [_OpCall(NKIMatmul, ("a", "b"), (), "c")]
MATMUL_RECT_PARAMS = ("a", "b")


MATMUL_TANH_ANALYSIS = _Analysis(
    var_dims={"a": ("d0", "d1"), "b": ("d0", "d3"), "c": ("d1", "d3"), "result": ("d1", "d3")},
    var_shapes={"a": (256, 256), "b": (256, 256), "c": (256, 256), "result": (256, 256)},
    parallel_dims=["d1", "d3"],
    reduction_dims=["d0"],
    tile_counts={"d1": 2, "d3": 1},
    reduction_tile_counts={"d0": 2},
    dim_tile_sizes={"d0": 128, "d1": 128, "d3": 256},
    return_var="result",
)
MATMUL_TANH_OP_CALLS = [
    _OpCall(NKIMatmul, ("a", "b"), (), "c"),
    _OpCall(NKIActivation, ("c",), (("op", np.tanh),), "result"),
]
MATMUL_TANH_PARAMS = ("a", "b")


ADD_ONLY_ANALYSIS = _Analysis(
    var_dims={"x": ("d0", "d1"), "y": ("d0", "d1"), "z": ("d0", "d1")},
    var_shapes={"x": (256, 256), "y": (256, 256), "z": (256, 256)},
    parallel_dims=["d0", "d1"],
    reduction_dims=[],
    tile_counts={"d0": 2, "d1": 2},
    reduction_tile_counts={},
    dim_tile_sizes={"d0": 128, "d1": 128},
    return_var="z",
)
ADD_ONLY_OP_CALLS = [_OpCall(NKIAdd, ("x", "y"), (), "z")]
ADD_ONLY_PARAMS = ("x", "y")


MATMUL_ADD_ANALYSIS = _Analysis(
    var_dims={"a": ("d0", "d1"), "b": ("d0", "d3"), "bias": ("d1", "d3"), "c": ("d1", "d3"), "result": ("d1", "d3")},
    var_shapes={"a": (256, 256), "b": (256, 256), "bias": (256, 256), "c": (256, 256), "result": (256, 256)},
    parallel_dims=["d1", "d3"],
    reduction_dims=["d0"],
    tile_counts={"d1": 2, "d3": 1},
    reduction_tile_counts={"d0": 2},
    dim_tile_sizes={"d0": 128, "d1": 128, "d3": 256},
    return_var="result",
)
MATMUL_ADD_OP_CALLS = [_OpCall(NKIMatmul, ("a", "b"), (), "c"), _OpCall(NKIAdd, ("c", "bias"), (), "result")]
MATMUL_ADD_PARAMS = ("a", "b", "bias")
