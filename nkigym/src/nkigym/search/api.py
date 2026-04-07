"""Search API: expand transform graph and benchmark variants remotely.

``remote_search`` wraps ``remote_profile`` with transform graph
expansion — it renders the base IR, grows the graph to
``num_variants`` nodes, and submits every variant for profiling.
"""

import random

from autotune.runner.api import remote_profile
from autotune.runner.output import ProfileOutput
from autotune.runner.types import KernelJob
from nkigym.codegen.render import KernelIR, render_ir
from nkigym.search.graph import TransformGraph
from nkigym.transforms import Transform
from nkigym.transforms.loop_fusion import LoopFusion

_DEFAULT_TRANSFORMS: list[Transform] = [LoopFusion()]


def remote_search(
    initial_kernel: KernelIR,
    golden_source: str,
    golden_func_name: str,
    hosts: list[str],
    cache_dir: str,
    num_variants: int,
    atol: float,
    rtol: float,
    warmup: int,
    iters: int,
    transforms: list[Transform] | None = None,
    seed: int = 42,
) -> ProfileOutput:
    """Expand transform graph and benchmark all variants.

    Args:
        initial_kernel: Base KernelIR from ``build_ir``.
        golden_source: Source code of the golden numpy function.
        golden_func_name: Name of the golden function.
        hosts: SSH hostnames for remote workers.
        cache_dir: Directory to cache results.
        num_variants: Target number of kernel variants to explore.
        atol: Absolute tolerance for correctness checking.
        rtol: Relative tolerance for correctness checking.
        warmup: Warmup iterations before timing.
        iters: Benchmark iterations.
        transforms: Transforms to apply. Defaults to ``[LoopFusion()]``.
        seed: RNG seed for reproducible exploration.

    Returns:
        ProfileOutput with timing and correctness for all variants.
    """
    active_transforms = transforms if transforms is not None else list(_DEFAULT_TRANSFORMS)

    graph = TransformGraph(base_ir=initial_kernel, render_fn=render_ir, transforms=active_transforms)
    rng = random.Random(seed)
    graph.expand(num_variants=num_variants, rng=rng)

    kernels: dict[str, KernelJob] = {}
    for i, node in enumerate(graph.nodes):
        name = f"{initial_kernel.func_name}_v{i}"
        kernels[name] = KernelJob(
            source=node.source,
            input_specs=initial_kernel.input_specs,
            golden_source=golden_source,
            golden_func_name=golden_func_name,
            atol=atol,
            rtol=rtol,
        )

    return remote_profile(kernels=kernels, hosts=hosts, cache_dir=cache_dir, warmup=warmup, iters=iters)
