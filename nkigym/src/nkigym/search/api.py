"""``remote_run``: render a :class:`KernelIR` to NKI source and profile it.

Callers build a ``KernelIR`` (via ``build_ir`` for the baseline, or
hand-constructed for tuned variants), optionally tweak its knobs, and
pass it here. The IR is rendered, shipped to a Trainium host, and
benchmarked; results mirror the on-disk cache layout.
"""

import ast
import importlib
import inspect
import pkgutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

import nkigym.ops as ops_pkg
from autotune.runner.api import remote_profile
from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob, ProfileConfig
from nkigym.codegen import gadgets as _gadgets
from nkigym.codegen import render_ir
from nkigym.kernel_ir.ir import KernelIR
from nkigym.ops.base import NKIOp
from nkigym.search.mac import compute_mac_count


def remote_run(
    ir: KernelIR,
    func: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    hosts: list[str],
    cache_dir: str,
    atol: float,
    rtol: float,
    kernel_name: str = "kernel.py",
    neuronx_cc_args: tuple[str, ...] = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false"),
    config: ProfileConfig = ProfileConfig(),
) -> ProfileOutput:
    """Render ``ir`` and profile against the math function ``func``.

    Args:
        ir: The KernelIR to render.
        func: The nkigym math function — shipped to remote workers for
            the fp32 golden.
        input_specs: ``{param: (shape, dtype)}`` for every parameter.
        hosts: SSH hostnames for remote profiling.
        cache_dir: Directory for kernel sources + results.
        atol: Absolute tolerance for CPU-sim correctness.
        rtol: Relative tolerance for CPU-sim correctness.
        kernel_name: Filename for the rendered kernel.
        neuronx_cc_args: Extra compiler flags.
        config: Infra settings.

    Returns:
        ProfileOutput with timing and correctness.
    """
    source = render_ir(ir)
    payload = _inline_gadgets(source)
    mac_count = compute_mac_count(func, input_specs)
    nkigym_source = _func_source_with_imports(func)
    output_shape = tuple(ir.logical_tensors[ir.return_name].shape)
    job = KernelJob(
        source=payload,
        func_name=ir.func_name,
        output_shape=output_shape,
        input_specs=input_specs,
        nkigym_source=nkigym_source,
        nkigym_func_name=func.__name__,
        mac_count=mac_count,
        atol=atol,
        rtol=rtol,
        neuronx_cc_args=neuronx_cc_args,
    )
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    _dump_ir(cache_path, kernel_name, ir)
    return remote_profile(kernels={kernel_name: job}, hosts=hosts, cache_dir=cache_dir, config=config)


def _dump_ir(cache_path: Path, kernel_name: str, ir: KernelIR) -> None:
    """Write the KernelIR fields alongside the rendered kernel."""
    stem = Path(kernel_name).stem
    ir_dir = cache_path / stem
    ir_dir.mkdir(parents=True, exist_ok=True)
    (ir_dir / "ir.md").write_text(_format_ir(ir))


def _format_ir(ir: KernelIR) -> str:
    """Human-readable dump of every KernelIR field."""
    lines: list[str] = [
        f"# KernelIR: {ir.func_name}",
        "",
        f"- param_names: {ir.param_names}",
        f"- return_name: {ir.return_name}",
        f"- dim_order: {ir.dim_order}",
        f"- ltiles_per_block: {ir.ltiles_per_block}",
        "",
        "## dimensions",
    ]
    for name, info in ir.dimensions.items():
        lines.append(f"- {name}: {info}")
    lines.extend(["", "## logical_tensors"])
    for name, t in ir.logical_tensors.items():
        lines.append(f"- {name}: {t}")
    lines.extend(["", "## physical_buffers"])
    for name, buf in ir.physical_buffers.items():
        lines.append(f"- {name}: {buf}")
    lines.extend(["", "## buffer_scopes"])
    for name, scope in ir.buffer_scopes.items():
        lines.append(f"- {name}: {scope.value}")
    lines.extend(["", "## num_buffers"])
    for name, nb in ir.num_buffers.items():
        lines.append(f"- {name}: {nb}")
    lines.extend(["", "## emission_depth"])
    for name, depth in ir.emission_depth.items():
        lines.append(f"- {name}: {depth}")
    lines.extend(["", "## ops"])
    for i, op in enumerate(ir.ops):
        lines.append(f"{i}. {op}")
    lines.extend(["", "## edges"])
    for producer, consumer in ir.edges:
        lines.append(f"- {producer} → {consumer}")
    return "\n".join(lines) + "\n"


def _inline_gadgets(kernel_src: str) -> str:
    """Prepend gadgets.py so the emitted source is self-contained for workers."""
    gadgets_src = Path(_gadgets.__file__).read_text()
    gadgets_import_block = (
        "from nkigym.codegen.gadgets import (\n"
        "    allocate_buffers,\n"
        "    load_block,\n"
        "    matmul_block,\n"
        "    memset_buffers,\n"
        "    store_block,\n"
        "    transpose_block,\n"
        ")"
    )
    return gadgets_src + "\n\n" + kernel_src.replace(gadgets_import_block, "")


def _func_source_with_imports(func: Callable[..., np.ndarray]) -> str:
    """Return *func*'s source prefixed with a minimal nkigym-safe preamble."""
    module = inspect.getmodule(func)
    if module is None:
        raise ValueError(f"Cannot resolve module for {func!r}")
    module_src = inspect.getsource(module)
    tree = ast.parse(module_src)
    const_segments: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            seg = ast.get_source_segment(module_src, node)
            if seg is not None:
                const_segments.append(seg)
    preamble = "\n".join(["import numpy as np", _nkiop_imports_line(), *const_segments])
    return preamble + "\n\n" + inspect.getsource(func)


def _nkiop_imports_line() -> str:
    """Emit one ``from nkigym.ops.X import NKIY`` per ``NKIOp`` subclass."""
    lines: list[str] = []
    for _finder, name, _is_pkg in pkgutil.iter_modules(ops_pkg.__path__):
        mod = importlib.import_module(f"nkigym.ops.{name}")
        for attr, value in vars(mod).items():
            if isinstance(value, type) and issubclass(value, NKIOp) and value is not NKIOp:
                lines.append(f"from nkigym.ops.{name} import {attr}")
    return "\n".join(sorted(set(lines)))
