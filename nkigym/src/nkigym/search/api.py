"""remote_search: sample unique kernel variants and profile them on remote Trainium hosts."""

import ast
import importlib
import inspect
import pkgutil
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

import nkigym.ops as ops_pkg
from autotune.runner.api import remote_profile
from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob, ProfileConfig
from nkigym.kernel_ir import build_naive_ir
from nkigym.ops.base import NKIOp
from nkigym.search.mac import compute_mac_count
from nkigym.search.sampler import sample_variants


def _func_source_with_imports(func: Callable[..., np.ndarray]) -> str:
    """Return the function's source prefixed with a minimal nkigym-safe preamble.

    The worker ``exec``s this string to materialize the nkigym
    math function as the golden reference. The preamble pulls in
    ``numpy``, every ``NKIOp`` subclass, and module-level ``Assign``
    constants from the function's own module — enough for typical
    math functions, without importing the module itself (which may
    trigger ``__main__`` side effects or pull coordinator-only
    dependencies like ``graphviz``).
    """
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
    """Emit a single ``from nkigym.ops.X import NKIY`` line per ``NKIOp`` subclass.

    Walking ``nkigym.ops`` gives us every op the user might call;
    shipping the full set is cheap and keeps the preamble independent
    of which specific ops the function references.
    """
    lines: list[str] = []
    for _finder, name, _is_pkg in pkgutil.iter_modules(ops_pkg.__path__):
        mod = importlib.import_module(f"nkigym.ops.{name}")
        for attr, value in vars(mod).items():
            if isinstance(value, type) and issubclass(value, NKIOp) and value is not NKIOp:
                lines.append(f"from nkigym.ops.{name} import {attr}")
    return "\n".join(sorted(set(lines)))


def remote_search(
    func: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    hosts: list[str],
    cache_dir: str,
    num_variants: int,
    atol: float,
    rtol: float,
    seed: int | None = None,
    config: ProfileConfig = ProfileConfig(),
) -> ProfileOutput:
    """Sample ``num_variants`` unique variants from ``func`` and profile them.

    Builds a seed ``KernelIR`` for ``func`` internally, rejection-
    samples variants, wraps each in a ``KernelJob`` and delegates to
    ``remote_profile``. The nkigym math function itself is shipped
    to every remote worker and executed there as the fp32 golden
    reference — no separate numpy reference is required. Users who
    want to sanity-check the nkigym function against a hand-written
    numpy version should do so outside this call. MACs are computed
    from the seed IR on the coordinator (identical across variants)
    and plumbed through to the MFU calculation on workers.

    Args:
        func: Math function using NKIOp classes. Dimension analysis,
            seed op ir, and the fp32 golden reference are all
            derived from this.
        input_specs: ``{param_name: (shape, dtype)}`` used by every
            variant.
        hosts: SSH hostnames for remote profiling.
        cache_dir: Directory to cache kernel sources + results.
        num_variants: Number of unique variants to sample.
        atol: Absolute tolerance for CPU-sim correctness.
        rtol: Relative tolerance for CPU-sim correctness.
        seed: RNG seed for sampling. ``None`` = nondeterministic.
        config: Infra settings (SSH timeout, venv path, etc.).

    Returns:
        ProfileOutput with per-variant timing and correctness.
    """
    rng = random.Random(seed)
    naive_ir, seed_ir = build_naive_ir(func, input_specs)
    mac_count = compute_mac_count(func, input_specs)
    nkigym_source = _func_source_with_imports(func)
    cache_path = Path(cache_dir)
    variants = sample_variants(naive_ir, seed_ir, num_variants, rng, cache_dir=cache_path)
    output_shape = tuple(naive_ir.logical_tensors[naive_ir.return_name].shape)
    kernels: dict[str, KernelJob] = {
        f"{name}.py": KernelJob(
            source=source,
            func_name=naive_ir.func_name,
            output_shape=output_shape,
            input_specs=input_specs,
            nkigym_source=nkigym_source,
            nkigym_func_name=func.__name__,
            mac_count=mac_count,
            atol=atol,
            rtol=rtol,
        )
        for name, _ir, source in variants
    }
    return remote_profile(kernels=kernels, hosts=hosts, cache_dir=cache_dir, config=config)
