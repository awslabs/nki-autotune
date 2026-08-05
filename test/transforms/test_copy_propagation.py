"""Tests for contract-driven on-chip copy propagation."""

from __future__ import annotations

from pathlib import Path
from test._simulation import _load_source

import numpy as np

from nkigym.codegen import render
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation import NKIActivation
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar_reduce import NKITensorScalarReduce
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import CopyPropagation


@nkigym_kernel
def f_copy_consumer(lhs, rhs):
    """Materialize a matmul copy before an identity activation."""
    sbuf_lhs = NKILoad()(src=lhs)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_product = NKIMatmul()(stationary=sbuf_lhs, moving=sbuf_rhs)
    sbuf_product = NKITensorCopy()(src=psum_product)
    sbuf_output = NKIActivation(op="copy")(data=sbuf_product)
    hbm_output = NKIStore()(src=sbuf_output)
    return hbm_output


@nkigym_kernel
def f_dtype_converting_copy(data, scale):
    """Copy BF16 scale data into the FP32 storage required by a consumer."""
    sbuf_data = NKILoad()(src=data)
    sbuf_scale = NKILoad()(src=scale)
    sbuf_scale_fp32 = NKITensorCopy()(src=sbuf_scale)
    reduced = NKITensorScalarReduce(op0="multiply", reduce_op="add")(data=sbuf_data, operand0=sbuf_scale_fp32)
    output = NKIStore()(src=reduced)
    return output


def _build() -> KernelIR:
    """Build the canonical copy-consumer fixture."""
    specs = {"lhs": ((128, 128), "bfloat16"), "rhs": ((128, 128), "bfloat16")}
    return build_initial_ir(f_copy_consumer, specs)


def test_copy_propagation_preserves_matmul_result(tmp_path: Path) -> None:
    """A sole PSUM copy consumer reads the source directly without changing the result."""
    ir = _build()
    transform = CopyPropagation()
    options = transform.analyze(ir)
    assert len(options) == 1
    transformed = transform.apply(ir, options[0])
    source = render(transformed)
    assert "nisa.tensor_copy(" not in source
    assert "sbuf_product" not in source
    assert "nisa.activation(data=psum_product" in source
    assert transform.analyze(transformed) == []

    module = _load_source(render(transformed), tmp_path, "copy_propagation")
    rng = np.random.default_rng(43)
    lhs = rng.standard_normal((128, 128), dtype=np.float32)
    rhs = rng.standard_normal((128, 128), dtype=np.float32)
    actual = np.asarray(simulate_fp32(module.nki_f_copy_consumer)(lhs=lhs, rhs=rhs))
    expected = lhs.T @ rhs
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_copy_propagation_preserves_required_storage_conversion() -> None:
    """A BF16-to-FP32 copy cannot be removed from a required-FP32 operand."""
    specs = {"data": ((128, 128), "bfloat16"), "scale": ((128,), "bfloat16")}
    ir = build_initial_ir(f_dtype_converting_copy, specs)
    assert ir.buffer("sbuf_scale").physical_dtype() == "bfloat16"
    assert ir.buffer("sbuf_scale_fp32").physical_dtype() == "float32"
    assert CopyPropagation().analyze(ir) == []
