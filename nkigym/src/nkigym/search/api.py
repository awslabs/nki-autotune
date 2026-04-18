"""remote_search: sample unique kernel variants and profile them on remote Trainium hosts."""

import json
import random
import shutil
from pathlib import Path

from autotune.runner.api import remote_profile
from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob, ProfileConfig
from nkigym.kernel_ir import KernelIR
from nkigym.search.sampler import sample_variants

_WARMUP = 10
_ITERS = 100


def remote_search(
    initial_kernel: KernelIR,
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    golden_source: str,
    golden_func_name: str,
    hosts: list[str],
    cache_dir: str,
    num_variants: int,
    atol: float,
    rtol: float,
    seed: int | None = None,
    config: ProfileConfig = ProfileConfig(),
) -> ProfileOutput:
    """Sample ``num_variants`` unique variants from ``initial_kernel`` and profile them.

    Rejection-samples ``(group_dim_orders, tensor_placements)`` via
    ``sample_variants``, dumping each sampled IR and rendered source
    to ``<cache_dir>/kernels/<name>/`` as it is drawn, then wraps
    every variant in a ``KernelJob`` and delegates to
    ``remote_profile``. After profiling, the per-variant neuron
    compiler log and auxiliary ``nki/`` / ``neff/`` directories
    written by ``remote_profile`` are consolidated under the
    matching ``<cache_dir>/kernels/<name>/`` subdirectory.
    Benchmark warmup and iters are fixed at ``10`` and ``100``.

    Args:
        initial_kernel: Seed IR. Its fusion_groups, ltiles_per_block,
            and buffer_degrees carry through to every sampled variant
            unchanged; only dim_order and tensor_placements are drawn.
        input_specs: ``{param_name: (shape, dtype)}`` used by every
            variant.
        golden_source: Source code of the reference numpy function.
        golden_func_name: Reference function's name.
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
    cache_path = Path(cache_dir)
    variants = sample_variants(initial_kernel, num_variants, rng, cache_dir=cache_path)
    kernels: dict[str, KernelJob] = {
        f"{name}.py": KernelJob(
            source=source,
            input_specs=input_specs,
            golden_source=golden_source,
            golden_func_name=golden_func_name,
            atol=atol,
            rtol=rtol,
        )
        for name, _ir, source in variants
    }
    output = remote_profile(
        kernels=kernels, hosts=hosts, cache_dir=cache_dir, warmup=_WARMUP, iters=_ITERS, config=config
    )
    _consolidate_cache(cache_path, [name for name, _, _ in variants])
    return output


def _consolidate_cache(cache_dir: Path, variant_names: list[str]) -> None:
    """Fold ``<cache>/nki/`` and ``<cache>/neff/`` into ``<cache>/kernels/<name>/``.

    ``remote_profile`` writes kernel sources under ``<cache>/nki/``
    and the neuron compiler log under
    ``<cache>/neff/<name>/log-neuron-cc.txt``. This helper moves
    everything into the per-variant ``<cache>/kernels/<name>/``
    directory the sampler already populated, removes the now-empty
    ``nki/`` and ``neff/`` directories, and rewrites
    ``results.json`` entries to point at the new paths.
    """
    nki_dir = cache_dir / "nki"
    neff_dir = cache_dir / "neff"
    kernels_dir = cache_dir / "kernels"
    for name in variant_names:
        variant_dir = kernels_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        neff_src = neff_dir / name
        if neff_src.exists():
            for entry in neff_src.iterdir():
                shutil.move(str(entry), str(variant_dir / entry.name))
            neff_src.rmdir()
    if nki_dir.exists():
        shutil.rmtree(nki_dir)
    if neff_dir.exists():
        shutil.rmtree(neff_dir)

    results_path = cache_dir / "results.json"
    if results_path.exists():
        data = json.loads(results_path.read_text())
        for entry in data.get("kernels", []):
            kname = entry.get("kernel_name", "")
            stem = Path(kname).stem
            entry["kernel_path"] = f"kernels/{stem}/{stem}.py"
            entry.pop("nki_path", None)
        results_path.write_text(json.dumps(data, indent=2))
