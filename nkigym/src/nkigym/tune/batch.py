"""Frontier-expansion sampler for the tune stage batch path.

Two pure functions — no I/O, no rendering, no dependency on
``autotune`` — so the sampler can be unit-tested in isolation:

* :func:`enumerate_pool` explores the reachable rewrite graph via
  randomized frontier expansion, deduping states by
  :func:`hash_forest`.
* :func:`sample_pool` draws ``num_kernels`` states uniformly
  without replacement from the enumerated pool.

See ``docs/superpowers/specs/2026-05-05-unified-tune-stage-design.md``
for the algorithm rationale.
"""

import random
import warnings

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import LoopForest, hash_forest
from nkigym.tune import KernelRewrite
from nkigym.tune.fuse_loops import enumerate_fusion_atoms
from nkigym.tune.reorder_loops import enumerate_reorder_atoms


def enumerate_pool(
    op_graph: OpGraph, forest: LoopForest, max_pool_size: int, rng: random.Random
) -> dict[int, tuple[OpGraph, LoopForest]]:
    """Enumerate the reachable rewrite graph via randomized frontier expansion.

    Maintains a frontier: pool nodes that still have un-tried outgoing
    atoms. Each iteration picks one frontier node uniformly, pops one
    of its unexplored atoms uniformly, applies it, and adds the
    destination to the pool when the hash is new. Terminates when
    ``len(pool) >= max_pool_size`` OR the frontier is empty.

    The per-node atom list is snapshot at pool-insertion time and
    never re-enumerated — safe because atom enumerators are pure
    functions of the forest and frontier nodes are immutable
    ``(op_graph, forest)`` tuples.

    Args:
        op_graph: Starting ``OpGraph``.
        forest: Starting ``LoopForest``.
        max_pool_size: Stop when the pool reaches this size.
        rng: Seeded ``random.Random`` — drives frontier-node pick and
            atom pick.

    Returns:
        Dict keyed by ``hash_forest``; values are ``(op_graph, forest)``
        tuples. Always contains the starting state.
    """
    h0 = hash_forest(forest)
    pool: dict[int, tuple[OpGraph, LoopForest]] = {h0: (op_graph, forest)}
    frontier: dict[int, list[KernelRewrite]] = {h0: enumerate_fusion_atoms(forest) + enumerate_reorder_atoms(forest)}
    if not frontier[h0]:
        del frontier[h0]

    while frontier and len(pool) < max_pool_size:
        frontier_keys = list(frontier)
        h = rng.choice(frontier_keys)
        atoms = frontier[h]
        j = rng.randrange(len(atoms))
        atoms[j], atoms[-1] = atoms[-1], atoms[j]
        atom = atoms.pop()
        if not atoms:
            del frontier[h]

        src_og, src_f = pool[h]
        new_og, new_f = atom.apply(src_og, src_f)
        h_new = hash_forest(new_f)
        if h_new in pool:
            continue
        pool[h_new] = (new_og, new_f)
        new_atoms = enumerate_fusion_atoms(new_f) + enumerate_reorder_atoms(new_f)
        if new_atoms:
            frontier[h_new] = new_atoms

    return pool


def sample_pool(
    pool: dict[int, tuple[OpGraph, LoopForest]], num_kernels: int, rng: random.Random
) -> list[tuple[OpGraph, LoopForest]]:
    """Sample ``num_kernels`` distinct states uniformly from ``pool``.

    When ``len(pool) < num_kernels``, emits a :class:`UserWarning`
    and returns every pool value. Pool values are returned in
    rng-draw order — callers that care about order should sort or
    otherwise post-process.

    Args:
        pool: Output of :func:`enumerate_pool`.
        num_kernels: Requested number of samples.
        rng: Seeded ``random.Random``.

    Returns:
        List of ``(op_graph, forest)`` pairs; length
        ``min(num_kernels, len(pool))``.
    """
    values = list(pool.values())
    if len(values) < num_kernels:
        warnings.warn(f"pool size {len(values)} < num_kernels {num_kernels}; returning all", UserWarning, stacklevel=2)
        result = values
    else:
        result = rng.sample(values, num_kernels)
    return result
