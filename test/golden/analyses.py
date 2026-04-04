"""Shared analysis and op-call fixtures for schedule transform tests."""

import numpy as np

from nkigym.codegen.analysis import _Analysis, _OpCall
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_1d import NKIActivation1D
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.add import NKIAdd
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.tensor_scalar_const import NKITensorScalarConst
from nkigym.ops.transpose import NKITranspose

RMSNORM_MATMUL_OP_CALLS = [
    _OpCall(NKIActivationReduce, ("a",), (("op", "square"), ("reduce_op", np.add)), "sum_sq"),
    _OpCall(
        NKITensorScalarConst,
        ("sum_sq",),
        (("op0", np.multiply), ("operand0", 0.00048828125), ("op1", np.add), ("operand1", 1e-06)),
        "scaled",
    ),
    _OpCall(NKIActivation1D, ("scaled",), (("op", "rsqrt"),), "rsqrt_val"),
    _OpCall(NKITensorScalar, ("a", "rsqrt_val"), (("op0", np.multiply),), "a_normed"),
    _OpCall(NKITranspose, ("a_normed",), (), "a_t"),
    _OpCall(NKIMatmul, ("a_t", "b"), (), "result"),
]

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


RMSNORM_MATMUL_ANALYSIS = _Analysis(
    var_dims={
        "a": ("d0", "d1"),
        "b": ("d1", "d3"),
        "sum_sq": ("d0",),
        "scaled": ("d0",),
        "rsqrt_val": ("d0",),
        "a_normed": ("d0", "d1"),
        "a_t": ("d1", "d0"),
        "result": ("d0", "d3"),
    },
    var_shapes={
        "a": (256, 256),
        "b": (256, 256),
        "sum_sq": (256,),
        "scaled": (256,),
        "rsqrt_val": (256,),
        "a_normed": (256, 256),
        "a_t": (256, 256),
        "result": (256, 256),
    },
    parallel_dims=["d0", "d3"],
    reduction_dims=["d1"],
    tile_counts={"d0": 2, "d3": 1},
    reduction_tile_counts={"d1": 2},
    dim_tile_sizes={"d0": 128, "d1": 128, "d3": 256},
    return_var="result",
)
RMSNORM_MATMUL_PARAMS = ("a", "b")


ATTENTION_PARAMS = ("Q", "K", "V")

ATTENTION_ANALYSIS = _Analysis(
    var_dims={
        "Q": ("d0", "d1"),
        "K": ("d2", "d1"),
        "V": ("d2", "d5"),
        "Q_t": ("d1", "d0"),
        "K_t": ("d1", "d2"),
        "S": ("d0", "d2"),
        "scaled_S": ("d0", "d2"),
        "masked_S": ("d0", "d2"),
        "max_S": ("d0",),
        "shifted_S": ("d0", "d2"),
        "exp_S": ("d0", "d2"),
        "sum_exp": ("d0",),
        "inv_sum": ("d0",),
        "exp_S_t": ("d2", "d0"),
        "attn": ("d0", "d5"),
        "output": ("d0", "d5"),
    },
    var_shapes={
        "Q": (4096, 128),
        "K": (4096, 128),
        "V": (4096, 128),
        "Q_t": (128, 4096),
        "K_t": (128, 4096),
        "S": (4096, 4096),
        "scaled_S": (4096, 4096),
        "masked_S": (4096, 4096),
        "max_S": (4096,),
        "shifted_S": (4096, 4096),
        "exp_S": (4096, 4096),
        "sum_exp": (4096,),
        "inv_sum": (4096,),
        "exp_S_t": (4096, 4096),
        "attn": (4096, 128),
        "output": (4096, 128),
    },
    parallel_dims=["d0", "d5"],
    reduction_dims=["d1", "d2"],
    tile_counts={"d0": 32, "d5": 1},
    reduction_tile_counts={"d1": 1, "d2": 8},
    dim_tile_sizes={"d0": 128, "d1": 128, "d2": 512, "d5": 128},
    return_var="output",
)
