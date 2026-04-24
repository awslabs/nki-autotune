"""RMSNorm + matmul: build KernelIR by hand, render, profile on remote hosts.

Hardcodes the strict-IR-driven champion from ``nkigym/design_rmsnorm_matmul.md``
(~73 % MFU, no in-place reuse, no online fusion).

Math: ``output = RMSNorm(a) @ b = (a / sqrt(mean(a^2) + eps)) @ b``.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/rmsnorm_matmul.py
"""

import shutil
from pathlib import Path

from autotune.runner.api import remote_profile
from autotune.runner.types import KernelJob
from nkigym.codegen import gadgets as _gadgets
from nkigym.codegen import render_ir
from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.types import DimInfo, DimRole, TensorInfo

NKIGYM_SOURCE = """
import numpy as np
from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.tensor_scalar import NKITensorScalar
from nkigym.ops.transpose import NKITranspose

EPS = 1e-6


def rmsnorm_matmul_nkigym(a, b):
    k = a.shape[1]
    sq, sum_sq = NKIActivationReduce()(data=a, op="square", reduce_op="add")
    scaled = NKITensorScalar()(data=sum_sq, op0="multiply", operand0=1.0 / k, op1="add", operand1=EPS)
    rsqrt_val = NKIActivation()(data=scaled, op="rsqrt")
    a_normed = NKITensorScalar()(data=a, op0="multiply", operand0=rsqrt_val)
    a_t = NKITranspose()(data=a_normed)
    result = NKIMatmul()(stationary=a_t, moving=b)
    return result
"""


def build_rmsnorm_matmul_ir() -> KernelIR:
    """Hand-built ``RMSNorm(a) @ b`` IR matching ``design_rmsnorm_matmul.md``."""
    K = 2048
    EPS = 1e-6
    dimensions = {
        "d0": DimInfo(dim_size=2048, logical_tile_size=128, physical_tile_size=128, role=DimRole.ACCUMULATION),
        "d1": DimInfo(dim_size=2048, logical_tile_size=128, physical_tile_size=128, role=DimRole.PARALLEL),
        "d2": DimInfo(dim_size=2048, logical_tile_size=512, physical_tile_size=512, role=DimRole.PARALLEL),
    }
    logical_tensors = {
        "a": TensorInfo(dim_ids=("d1", "d0"), shape=(2048, 2048), dtype="bfloat16"),
        "b": TensorInfo(dim_ids=("d0", "d2"), shape=(2048, 2048), dtype="bfloat16"),
        "output": TensorInfo(dim_ids=("d1", "d2"), shape=(2048, 2048), dtype="bfloat16"),
    }
    physical_buffers = {
        "sbuf_a": PhysicalBuffer(tile=(128, 128), dim_ids=("d1", "d0"), p_axis="d1", f_axis="d0", dtype="bfloat16"),
        "sbuf_b": PhysicalBuffer(tile=(128, 512), dim_ids=("d0", "d2"), p_axis="d0", f_axis="d2", dtype="bfloat16"),
        "sbuf_sum_sq": PhysicalBuffer(tile=(128, 1), dim_ids=("d1",), p_axis="d1", f_axis=None, dtype="float32"),
        "sbuf_rsqrt_val": PhysicalBuffer(tile=(128, 1), dim_ids=("d1",), p_axis="d1", f_axis=None, dtype="float32"),
        "sbuf_a_normed": PhysicalBuffer(
            tile=(128, 128), dim_ids=("d1", "d0"), p_axis="d1", f_axis="d0", dtype="bfloat16"
        ),
        "sbuf_a_t": PhysicalBuffer(tile=(128, 128), dim_ids=("d0", "d1"), p_axis="d0", f_axis="d1", dtype="bfloat16"),
        "sbuf_output": PhysicalBuffer(
            tile=(128, 512), dim_ids=("d1", "d2"), p_axis="d1", f_axis="d2", dtype="bfloat16"
        ),
        "output_hbm": PhysicalBuffer(
            tile=(2048, 2048), dim_ids=("d1", "d2"), p_axis="d1", f_axis="d2", dtype="bfloat16"
        ),
    }
    ops = [
        Op(kind="NKILoad", inputs={"data": "a"}, outputs=["sbuf_a"]),
        Op(kind="NKILoad", inputs={"data": "b"}, outputs=["sbuf_b"]),
        Op(
            kind="NKIActivationReduce",
            inputs={"data": "sbuf_a"},
            outputs=["sbuf_sum_sq"],
            kwargs={"op": "square", "reduce_op": "add"},
        ),
        Op(
            kind="NKITensorScalar",
            inputs={"data": "sbuf_sum_sq"},
            outputs=["sbuf_rsqrt_val"],
            kwargs={"op0": "multiply", "operand0": 1.0 / K, "op1": "add", "operand1": EPS},
        ),
        Op(kind="NKIActivation", inputs={"data": "sbuf_rsqrt_val"}, outputs=["sbuf_rsqrt_val"], kwargs={"op": "rsqrt"}),
        Op(
            kind="NKITensorScalar",
            inputs={"data": "sbuf_a", "operand0": "sbuf_rsqrt_val"},
            outputs=["sbuf_a_normed"],
            kwargs={"op0": "multiply", "operand0": "sbuf_rsqrt_val"},
        ),
        Op(
            kind="NKITranspose", inputs={"data": "sbuf_a_normed"}, outputs=["sbuf_a_t"], attrs={"mode": "dma_transpose"}
        ),
        Op(
            kind="NKIMatmul",
            inputs={"stationary": "sbuf_a_t", "moving": "sbuf_b"},
            outputs=["sbuf_output"],
            axis_map={"K": "d0", "M": "d1", "N": "d2"},
            blocking_dims={"d0"},
        ),
        Op(kind="NKIStore", inputs={"data": "sbuf_output"}, outputs=["output_hbm"]),
    ]
    edges = [
        (0, 2, "sbuf_a", "data"),
        (2, 3, "sbuf_sum_sq", "data"),
        (3, 4, "sbuf_rsqrt_val", "data"),
        (0, 5, "sbuf_a", "data"),
        (4, 5, "sbuf_rsqrt_val", "operand0"),
        (5, 6, "sbuf_a_normed", "data"),
        (6, 7, "sbuf_a_t", "stationary"),
        (1, 7, "sbuf_b", "moving"),
        (7, 8, "sbuf_output", "data"),
    ]
    return KernelIR(
        func_name="rmsnorm_matmul_nkigym",
        param_names=["a", "b"],
        return_name="output",
        dimensions=dimensions,
        logical_tensors=logical_tensors,
        physical_buffers=physical_buffers,
        ops=ops,
        edges=edges,
        dim_order=["d1", "d2", "d0"],
        ltiles_per_block={"d0": 8, "d1": 4, "d2": 1},
        buffer_scopes={
            "sbuf_a": BufferScope.MIDDLE,
            "sbuf_b": BufferScope.INNER,
            "sbuf_a_normed": BufferScope.MIDDLE,
            "sbuf_a_t": BufferScope.MIDDLE,
        },
        num_buffers={
            "sbuf_b": NumBuffers(num_p_buffers=4, num_f_buffers=2),
            "sbuf_output": NumBuffers(num_p_buffers=None, num_f_buffers=4),
        },
        emission_depth={"sbuf_b": 1, "sbuf_output": 0},
    )


def inline_gadgets(kernel_src: str) -> str:
    """Prepend the gadgets module so the emitted source ships as one file."""
    gadgets_src = Path(_gadgets.__file__).read_text()
    kernel_src = kernel_src.replace(
        "from nkigym.codegen.gadgets import (\n"
        "    activation_block,\n"
        "    activation_reduce_block,\n"
        "    allocate_buffers,\n"
        "    load_block,\n"
        "    matmul_block,\n"
        "    memset_buffers,\n"
        "    store_block,\n"
        "    tensor_scalar_block,\n"
        "    transpose_block,\n"
        ")",
        "",
    )
    return gadgets_src + "\n\n" + kernel_src


if __name__ == "__main__":
    M, K, N = 2048, 2048, 2048
    HOSTS = ["gym-2"]
    ATOL, RTOL = 1e-2, 1e-2

    CACHE_ROOT = Path("/home/ubuntu/cache/rmsnorm_matmul")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    ir = build_rmsnorm_matmul_ir()
    source = render_ir(ir)
    (CACHE_ROOT / "rendered.py").write_text(source)
    payload = inline_gadgets(source)
    (CACHE_ROOT / "rendered_inlined.py").write_text(payload)

    job = KernelJob(
        source=payload,
        func_name=ir.func_name,
        output_shape=(M, N),
        input_specs={"a": ((M, K), "bfloat16"), "b": ((K, N), "bfloat16")},
        nkigym_source=NKIGYM_SOURCE,
        nkigym_func_name=ir.func_name,
        mac_count=M * K * N,
        atol=ATOL,
        rtol=RTOL,
        neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
    )

    output = remote_profile(kernels={"rmsnorm_matmul.py": job}, hosts=HOSTS, cache_dir=str(CACHE_ROOT))
    for r in output.results:
        print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')}  min_ms={r.min_ms}  MFU={r.mfu}")
