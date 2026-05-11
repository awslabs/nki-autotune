"""Frontier-expansion sampler for the tune stage — 7-atom enumeration.

Two pure functions — no I/O, no rendering:

* :func:`enumerate_pool` explores the reachable rewrite graph via randomized
  frontier expansion, deduping states by :func:`hash_state`.
* :func:`sample_pool` draws ``num_kernels`` states uniformly without
  replacement from the enumerated pool.

Atoms enumerated by default:

- Domain: :class:`Split`, :class:`Reorder`.
- Placement: :class:`ComputeAt`, :class:`ReverseComputeAt`.
- Annotation: :class:`Annotate` (``buffer_degree`` +
  ``software_pipeline_depth`` keys).

Off the default sampler (atom available for explicit use, but not
enumerated during random frontier expansion):

- :class:`RFactor` — ``_apply_rmw``'s drain-tree rebuild has a
  KeyError when block-local iter vars fall outside the forest-visible
  chain. Re-enable once the fix lands.
- :class:`Fuse` — renderer's ``_resolve_iv_name`` recursively
  decomposes fused components, producing divergent
  ``((i // a) // b) // c`` expressions when the fuse target is
  itself a component of a prior fuse. Re-enable once the recursion is
  bounded to a single level of ``//``/``%`` decomposition.

``HoistInvariant`` was removed — its legality is a strict subset of
:class:`ComputeAt` under the iter-var IR. ``MultiBuffer`` and
``SoftwarePipeline`` were consolidated into :class:`Annotate` with keyed
dispatch.

See ``docs/superpowers/specs/2026-05-10-iter-var-refactor-design.md``.
"""

import random
import warnings

from nkigym.codegen.dep_cache import subtree_signature
from nkigym.codegen.ir import KernelModule, validate_dataflow_ordering
from nkigym.tune import AtomLegalityError, KernelRewrite
from nkigym.tune.annotate import enumerate_annotate_atoms
from nkigym.tune.compute_at import enumerate_compute_at_atoms
from nkigym.tune.reorder import enumerate_reorder_atoms
from nkigym.tune.reverse_compute_at import enumerate_reverse_compute_at_atoms
from nkigym.tune.split import enumerate_split_atoms


def hash_state(module: KernelModule) -> int:
    """Structural hash of the tune-stage state.

    Folds the body's structural signature with every tensor's
    ``buffer_degree`` and ``module.fused_iter_var_map`` so that body-editing
    atoms, :class:`Annotate` buffer-degree mutations, and :class:`Fuse`
    iter-var registrations all register as distinct states.

    Args:
        module: The :class:`KernelModule` to hash.

    Returns:
        An ``int`` suitable for dict-key deduplication of pool states.
    """
    body_key = tuple(subtree_signature(c) for c in module.body)
    tensor_key = tuple(sorted((t.name, tuple(sorted(t.buffer_degree.items()))) for t in module.tensors.values()))
    fuse_key = tuple(sorted(module.fused_iter_var_map.items()))
    return hash((body_key, tensor_key, fuse_key))


def _enumerate_atoms(module: KernelModule) -> list[KernelRewrite]:
    """Return every default-sampler atom currently applicable to ``module``.

    Five atoms: Split, Reorder, ComputeAt, ReverseComputeAt, Annotate.
    RFactor and Fuse are excluded pending fixes documented in the
    module docstring.

    Args:
        module: The :class:`KernelModule` to enumerate atoms on.

    Returns:
        Flat list of every legal atom across the eligible enumerators.
    """
    atoms: list[KernelRewrite] = []
    atoms.extend(enumerate_split_atoms(module))
    atoms.extend(enumerate_reorder_atoms(module))
    atoms.extend(enumerate_compute_at_atoms(module))
    atoms.extend(enumerate_reverse_compute_at_atoms(module))
    atoms.extend(enumerate_annotate_atoms(module))
    return atoms


def enumerate_pool(module: KernelModule, max_pool_size: int, rng: random.Random) -> dict[int, KernelModule]:
    """Enumerate the reachable rewrite graph via randomized frontier expansion.

    Frontier = pool nodes with un-tried outgoing atoms. Each iteration picks
    one frontier node uniformly, pops one of its unexplored atoms uniformly,
    applies it, adds the destination to the pool if new. Terminates when
    ``len(pool) >= max_pool_size`` OR the frontier is empty.

    Atom lists are snapshot at pool-insertion time — safe because enumerators
    are pure functions of ``module`` and pool values are immutable
    :class:`KernelModule` references.

    After each apply, :func:`validate_dataflow_ordering` vets the result —
    compositions of structural atoms (``ComputeAt``/``Reorder``/...) can
    produce a forest where a leaf reads a tensor before its producer emits
    in linearization order. Such states are skipped (not added to the pool).

    Args:
        module: Starting :class:`KernelModule`.
        max_pool_size: Stop when pool reaches this size.
        rng: Seeded ``random.Random`` driving frontier node + atom picks.

    Returns:
        Dict keyed by :func:`hash_state`; values are :class:`KernelModule`
        references. Always contains the starting state.
    """
    h0 = hash_state(module)
    pool: dict[int, KernelModule] = {h0: module}
    frontier: dict[int, list[KernelRewrite]] = {h0: _enumerate_atoms(module)}
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

        src_module = pool[h]
        try:
            new_module = atom.apply(src_module)
        except AtomLegalityError:
            continue
        if not validate_dataflow_ordering(new_module):
            continue
        h_new = hash_state(new_module)
        if h_new in pool:
            continue
        pool[h_new] = new_module
        new_atoms = _enumerate_atoms(new_module)
        if new_atoms:
            frontier[h_new] = new_atoms

    return pool


def sample_pool(pool: dict[int, KernelModule], num_kernels: int, rng: random.Random) -> list[KernelModule]:
    """Sample ``num_kernels`` distinct modules uniformly from ``pool``.

    Emits :class:`UserWarning` and returns every pool value when the pool
    is smaller than requested.

    Args:
        pool: Output of :func:`enumerate_pool`.
        num_kernels: Requested number of samples.
        rng: Seeded ``random.Random``.

    Returns:
        List of :class:`KernelModule` references; length
        ``min(num_kernels, len(pool))``.
    """
    values = list(pool.values())
    result: list[KernelModule]
    if len(values) < num_kernels:
        warnings.warn(f"pool size {len(values)} < num_kernels {num_kernels}; returning all", UserWarning, stacklevel=2)
        result = values
    else:
        result = rng.sample(values, num_kernels)
    return result
