"""Driver for the ``"tune"`` stage of ``nkigym_compile``.

Loads the synthesised ``f_nkigym``, builds the canonical
:class:`LoopForest`, applies a list of :class:`KernelRewrite`s (explicit
or randomly drawn), renders the transformed kernel, and runs a CPU-sim
accuracy check against the numpy golden.
"""

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.loop_forest import build_canonical_forest
from nkigym.codegen.render import render
from nkigym.tune import KernelRewrite
from nkigym.tune.fuse_loops import enumerate_fusion_atoms


def run_tune(
    f_numpy: Callable[..., np.ndarray],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    cache_path: Path,
    rewrites: list[KernelRewrite] | None,
    seed: int,
    load_f_nkigym: Callable[[Path], Callable[..., np.ndarray]],
    cpu_sim_check: Callable[[str, str, Callable[..., np.ndarray], dict[str, tuple[tuple[int, ...], str]]], None],
) -> None:
    """Apply structural rewrites, render, write ``kernel_tuned.py``, CPU-sim check.

    Two paths:

    * **Explicit** (``rewrites`` is a list): apply each rewrite in order,
      checking legality before each ``apply`` because the caller's
      boundary indices may become stale after prior applies.
    * **Random** (``rewrites`` is ``None``): re-enumerate legal atoms
      between applies and pick the first one whose coin-flip survives,
      using a seeded RNG. Terminates when no legal atom survives the
      draw.

    Args:
        f_numpy: Plain-numpy reference used as the CPU-sim golden.
        input_specs: Per-param ``(shape, dtype)``.
        cache_path: Directory holding ``f_nkigym.py``; receives
            ``kernel_tuned.py``.
        rewrites: Explicit list or ``None`` for the random path.
        seed: Seeds the random draw.
        load_f_nkigym: Helper that imports ``f_nkigym.py`` and returns
            the ``f_nkigym`` callable. Injected so the stage driver
            stays independent of ``nkigym.compile``'s internal helpers.
        cpu_sim_check: Helper that runs the rendered kernel through
            ``nki.simulate`` at fp32 and compares to ``f_numpy``.
    """
    f_nkigym_path = cache_path / "f_nkigym.py"
    if not f_nkigym_path.exists():
        raise ValueError(
            f"tune requires {f_nkigym_path!s} — run the 'synthesis' stage first "
            f"or place the file manually before invoking this stage."
        )
    f_nkigym = load_f_nkigym(f_nkigym_path)
    op_graph = parse_and_resolve(f_nkigym, input_specs)
    forest = build_canonical_forest(op_graph)

    if rewrites is None:
        rng = random.Random(seed)
        while True:
            atoms = enumerate_fusion_atoms(forest)
            candidates = [a for a in atoms if rng.random() < 0.5]
            if not candidates:
                break
            chosen = candidates[0]
            op_graph, forest = chosen.apply(op_graph, forest)
    else:
        for r in rewrites:
            if not r.is_legal(op_graph, forest):
                raise ValueError(f"{r!r} illegal on current state")
            op_graph, forest = r.apply(op_graph, forest)

    kernel_source = render(op_graph, forest=forest)
    (cache_path / "kernel_tuned.py").write_text(kernel_source)
    cpu_sim_check(kernel_source, f_nkigym.__name__, f_numpy, input_specs)
