"""Matrix multiplication: build KernelIR by hand, render, profile on remote hosts.

Hardcodes the champion matmul IR from ``nkigym/design.md`` (the 90.9 % MFU
``lhsT @ rhs`` configuration). The IR goes through ``render_ir`` to produce
a standalone NKI source file, then ``remote_profile`` benchmarks it on a
Trainium host.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul.py
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
from nkigym.ops.matmul import NKIMatmul


def matmul_lhsT_rhs_nkigym(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output
"""


def build_matmul_ir() -> KernelIR:
    """Hand-built ``lhs_T @ rhs`` IR matching ``design.md``."""
    dimensions = {
        "d0": DimInfo(dim_size=2048, logical_tile_size=128, physical_tile_size=128, role=DimRole.ACCUMULATION),
        "d1": DimInfo(dim_size=2048, logical_tile_size=128, physical_tile_size=128, role=DimRole.PARALLEL),
        "d2": DimInfo(dim_size=2048, logical_tile_size=512, physical_tile_size=512, role=DimRole.PARALLEL),
    }
    logical_tensors = {
        "lhs_T": TensorInfo(dim_ids=("d0", "d1"), shape=(2048, 2048), dtype="bfloat16"),
        "rhs": TensorInfo(dim_ids=("d0", "d2"), shape=(2048, 2048), dtype="bfloat16"),
        "output": TensorInfo(dim_ids=("d1", "d2"), shape=(2048, 2048), dtype="bfloat16"),
    }
    physical_buffers = {
        "sbuf_lhs_T": PhysicalBuffer(tile=(128, 128), dim_ids=("d0", "d1"), p_axis="d0", f_axis="d1", dtype="bfloat16"),
        "sbuf_rhs": PhysicalBuffer(tile=(128, 512), dim_ids=("d0", "d2"), p_axis="d0", f_axis="d2", dtype="bfloat16"),
        "sbuf_output": PhysicalBuffer(
            tile=(128, 512), dim_ids=("d1", "d2"), p_axis="d1", f_axis="d2", dtype="bfloat16"
        ),
        "output_hbm": PhysicalBuffer(
            tile=(2048, 2048), dim_ids=("d1", "d2"), p_axis="d1", f_axis="d2", dtype="bfloat16"
        ),
    }
    ops = [
        Op(kind="NKILoad", inputs={"data": "lhs_T"}, outputs=["sbuf_lhs_T"]),
        Op(kind="NKILoad", inputs={"data": "rhs"}, outputs=["sbuf_rhs"]),
        Op(
            kind="NKIMatmul",
            inputs={"stationary": "sbuf_lhs_T", "moving": "sbuf_rhs"},
            outputs=["sbuf_output"],
            axis_map={"K": "d0", "M": "d1", "N": "d2"},
            blocking_dims={"d0"},
        ),
        Op(kind="NKIStore", inputs={"data": "sbuf_output"}, outputs=["output_hbm"]),
    ]
    edges = [(0, 2, "sbuf_lhs_T", "stationary"), (1, 2, "sbuf_rhs", "moving"), (2, 3, "sbuf_output", "data")]
    return KernelIR(
        func_name="matmul_lhsT_rhs_nkigym",
        param_names=["lhs_T", "rhs"],
        return_name="output",
        dimensions=dimensions,
        logical_tensors=logical_tensors,
        physical_buffers=physical_buffers,
        ops=ops,
        edges=edges,
        dim_order=["d2", "d0", "d1"],
        ltiles_per_block={"d0": 8, "d1": 4, "d2": 1},
        buffer_scopes={"sbuf_lhs_T": BufferScope.INNER, "sbuf_rhs": BufferScope.INNER},
        num_buffers={
            "sbuf_lhs_T": NumBuffers(num_p_buffers=2, num_f_buffers=4),
            "sbuf_rhs": NumBuffers(num_p_buffers=2, num_f_buffers=None),
            "sbuf_output": NumBuffers(num_p_buffers=None, num_f_buffers=4),
        },
        emission_depth={"sbuf_lhs_T": 1, "sbuf_rhs": 1, "sbuf_output": 0},
    )


def inline_gadgets(kernel_src: str) -> str:
    """Prepend the gadgets module so the emitted source is self-contained
    for shipping to remote workers."""
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
    K, M, N = 2048, 2048, 2048
    HOSTS = ["gym-2"]
    ATOL, RTOL = 1e-2, 1e-2

    CACHE_ROOT = Path("/home/ubuntu/cache/matmul")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    ir = build_matmul_ir()
    source = render_ir(ir)
    (CACHE_ROOT / "rendered.py").write_text(source)
    payload = inline_gadgets(source)
    (CACHE_ROOT / "rendered_inlined.py").write_text(payload)

    job = KernelJob(
        source=payload,
        func_name=ir.func_name,
        output_shape=(M, N),
        input_specs={"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")},
        nkigym_source=NKIGYM_SOURCE,
        nkigym_func_name=ir.func_name,
        mac_count=K * M * N,
        atol=ATOL,
        rtol=RTOL,
        neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
    )

    output = remote_profile(kernels={"matmul.py": job}, hosts=HOSTS, cache_dir=str(CACHE_ROOT))
    for r in output.results:
        print(f"{r.kernel_name}: sim={r.cpu_sim.get('passed')}  min_ms={r.min_ms}  MFU={r.mfu}")
