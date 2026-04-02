"""Graph-based variant exploration using random transform sampling.

Instead of exhaustive enumeration, maintains a graph of kernel variants
connected by transforms.  Each growth step picks a random node, applies
a random transform, and adds the result if valid and unique.
"""

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from nkigym.codegen.analysis import _Analysis, _OpCall
from nkigym.codegen.passes import _PassAssignment
from nkigym.schedule.enumerate import _divisors
from nkigym.schedule.render import render_schedule
from nkigym.schedule.types import DimSchedule, Schedule, _total_tiles, validate

logger = logging.getLogger(__name__)

_TRANSFORM_NAMES: tuple[str, ...] = ("reorder_loop", "change_placement", "change_blocking")


@dataclass
class VariantNode:
    """A single kernel variant in the transform graph.

    Attributes:
        kernel_source: Rendered NKI kernel source string.
        parent_idx: Index of the parent node, or None for seeds.
        transform_name: Name of the transform that produced this node.
        schedule_hash: Hash of the schedule for deduplication.
        schedule: The schedule descriptor that produced this kernel.
    """

    kernel_source: str
    parent_idx: int | None
    transform_name: str
    schedule_hash: int
    schedule: Schedule


def _schedule_hash(schedule: Schedule) -> int:
    """Compute a deterministic hash of a schedule descriptor.

    Args:
        schedule: Schedule to hash.

    Returns:
        Integer hash suitable for deduplication.
    """
    raw = repr(schedule).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _swap_two(items: tuple[tuple[str, int], ...], idx_a: int, idx_b: int) -> tuple[tuple[str, int], ...]:
    """Swap two items in a loop order tuple.

    Args:
        items: Original loop order.
        idx_a: First swap index.
        idx_b: Second swap index.

    Returns:
        New loop order with items at idx_a and idx_b swapped.
    """
    lst = list(items)
    lst[idx_a], lst[idx_b] = lst[idx_b], lst[idx_a]
    return tuple(lst)


@dataclass
class TransformGraph:
    """Graph of kernel variants connected by transform edges.

    Nodes are kernel source strings; edges are transform operations.
    The graph grows by randomly picking existing nodes and applying
    random transforms to produce new unique variants.

    Attributes:
        analysis: Dimension analysis from the workload.
        op_calls: Parsed operation calls.
        params: Input parameter names.
        func_name: Kernel function name.
        pa: Pass assignment from analysis.
    """

    analysis: _Analysis
    op_calls: list[_OpCall]
    params: tuple[str, ...]
    func_name: str
    pa: _PassAssignment
    _nodes: list[VariantNode] = field(default_factory=list)
    _seen_hashes: set[int] = field(default_factory=set)

    def add_seed(self, kernel_source: str, schedule: Schedule) -> int:
        """Add a root kernel variant (no parent).

        Args:
            kernel_source: Rendered NKI source string.
            schedule: The schedule descriptor that produced this kernel.

        Returns:
            Node index of the added seed.
        """
        h = _schedule_hash(schedule)
        node = VariantNode(
            kernel_source=kernel_source, parent_idx=None, transform_name="seed", schedule_hash=h, schedule=schedule
        )
        idx = len(self._nodes)
        self._nodes.append(node)
        self._seen_hashes.add(h)
        return idx

    def random_grow(self, target_size: int, rng: np.random.Generator, max_attempts: int = 1000) -> int:
        """Randomly grow the graph to target_size nodes.

        Each step picks a random existing node, applies a random
        transform to its schedule, renders the result, and adds it
        if valid and unique.

        Args:
            target_size: Desired total number of nodes.
            rng: Numpy random generator for reproducibility.
            max_attempts: Maximum failed attempts before giving up.

        Returns:
            Number of nodes added during this call.
        """
        added = 0
        attempts = 0
        while len(self._nodes) < target_size and attempts < max_attempts:
            attempts += 1
            parent_idx = int(rng.integers(0, len(self._nodes)))
            parent = self._nodes[parent_idx]
            transform_name = _TRANSFORM_NAMES[int(rng.integers(0, len(_TRANSFORM_NAMES)))]
            new_schedule = _apply_transform(parent.schedule, transform_name, self.analysis, self.params, self.pa, rng)
            if new_schedule is None:
                continue
            if not validate(self.analysis, new_schedule, self.params):
                continue
            h = _schedule_hash(new_schedule)
            if h in self._seen_hashes:
                continue
            try:
                kernel_source = render_schedule(
                    self.analysis, new_schedule, self.op_calls, self.params, self.func_name, self.pa
                )
            except Exception:
                logger.debug("Render failed for transform %s", transform_name)
                continue
            node = VariantNode(
                kernel_source=kernel_source,
                parent_idx=parent_idx,
                transform_name=transform_name,
                schedule_hash=h,
                schedule=new_schedule,
            )
            self._nodes.append(node)
            self._seen_hashes.add(h)
            added += 1
            attempts = 0
        return added

    def variants(self) -> list[str]:
        """Return all unique kernel source strings.

        Returns:
            List of rendered NKI kernel source strings.
        """
        return [n.kernel_source for n in self._nodes]

    def schedules(self) -> list[Schedule]:
        """Return all schedules in node order.

        Returns:
            List of schedule descriptors.
        """
        return [n.schedule for n in self._nodes]

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)


def _apply_transform(
    schedule: Schedule,
    transform_name: str,
    analysis: _Analysis,
    params: tuple[str, ...],
    pa: _PassAssignment,
    rng: np.random.Generator,
) -> Schedule | None:
    """Apply a named transform to a schedule, returning the new schedule or None.

    Args:
        schedule: Input schedule to transform.
        transform_name: One of the _TRANSFORM_NAMES.
        analysis: Dimension analysis.
        params: Input parameter names.
        pa: Pass assignment.
        rng: Random generator.

    Returns:
        New schedule, or None if the transform cannot be applied.
    """
    if transform_name == "reorder_loop":
        return _transform_reorder(schedule, rng)
    if transform_name == "change_placement":
        return _transform_placement(schedule, analysis, params, rng)
    if transform_name == "change_blocking":
        return _transform_blocking(schedule, analysis, rng)
    return None


def _transform_reorder(schedule: Schedule, rng: np.random.Generator) -> Schedule | None:
    """Swap two adjacent items in the loop order.

    Args:
        schedule: Input schedule.
        rng: Random generator.

    Returns:
        New schedule with swapped loop order, or None if too few items.
    """
    lo = schedule.loop_order
    n = len(lo)
    if n < 2:
        return None
    idx = int(rng.integers(0, n - 1))
    new_lo = _swap_two(lo, idx, idx + 1)
    return Schedule(loop_order=new_lo, dim_schedules=schedule.dim_schedules, op_placements=schedule.op_placements)


def _transform_placement(
    schedule: Schedule, analysis: _Analysis, params: tuple[str, ...], rng: np.random.Generator
) -> Schedule:
    """Randomly change one parameter's op placement level.

    Args:
        schedule: Input schedule.
        analysis: Dimension analysis.
        params: Input parameter names.
        rng: Random generator.

    Returns:
        New schedule with modified placement.
    """
    from nkigym.schedule.types import _var_dim_ids

    idx = int(rng.integers(0, len(params)))
    param = params[idx]
    n_deps = len(_var_dim_ids(analysis, param)) if param in analysis.var_dims else 0
    new_level = int(rng.integers(0, n_deps + 1))
    placements = list(schedule.op_placements)
    placements[idx] = new_level
    return Schedule(
        loop_order=schedule.loop_order, dim_schedules=schedule.dim_schedules, op_placements=tuple(placements)
    )


def _transform_blocking(schedule: Schedule, analysis: _Analysis, rng: np.random.Generator) -> Schedule:
    """Randomly change one dimension's tiles_per_block.

    Args:
        schedule: Input schedule.
        analysis: Dimension analysis.
        rng: Random generator.

    Returns:
        New schedule with modified blocking.
    """
    ds_list = list(schedule.dim_schedules)
    idx = int(rng.integers(0, len(ds_list)))
    old_ds = ds_list[idx]
    total = _total_tiles(old_ds.dim_id, analysis)
    divs = _divisors(total)
    new_tpb = divs[int(rng.integers(0, len(divs)))]
    ds_list[idx] = DimSchedule(old_ds.dim_id, old_ds.tile_size, new_tpb)
    return Schedule(loop_order=schedule.loop_order, dim_schedules=tuple(ds_list), op_placements=schedule.op_placements)
