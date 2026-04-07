"""Transform graph: explore kernel variants by applying transforms.

Nodes are (KernelIR, rendered source) pairs.  Edges record which
transform produced each child from which parent.  Duplicate sources
are rejected so different transform paths that converge collapse
into one node.
"""

import random
from collections.abc import Callable
from typing import NamedTuple

from nkigym.codegen.render import KernelIR
from nkigym.transforms import Transform


class SearchNode(NamedTuple):
    """A node in the transform graph."""

    ir: KernelIR
    source: str


class SearchEdge(NamedTuple):
    """An edge in the transform graph."""

    transform_name: str
    parent_idx: int
    child_idx: int


class TransformGraph:
    """Directed graph of kernel variants connected by transforms.

    Starts with one node (the base kernel) and grows by randomly
    picking a node, picking an applicable transform, and applying it.

    Attributes:
        nodes: All discovered kernel variants.
        edges: Transform applications connecting nodes.
    """

    def __init__(self, base_ir: KernelIR, render_fn: Callable[[KernelIR], str], transforms: list[Transform]) -> None:
        """Initialize with base IR as node 0.

        Args:
            base_ir: Initial kernel IR from ``build_ir``.
            render_fn: Function to render KernelIR to source string.
            transforms: List of transforms to explore.
        """
        base_source = render_fn(base_ir)
        self.nodes: list[SearchNode] = [SearchNode(ir=base_ir, source=base_source)]
        self.edges: list[SearchEdge] = []
        self._render_fn = render_fn
        self._transforms = transforms
        self._seen_sources: set[str] = {base_source}

    def expand(self, num_variants: int, rng: random.Random) -> None:
        """Grow the graph to at most num_variants nodes.

        Randomly picks an unexplored (node, transform) pair, applies
        it, and adds any new unique children.  Stops when the target
        count is reached or no applicable transforms remain.

        Args:
            num_variants: Target number of nodes.
            rng: Random number generator for reproducibility.
        """
        frontier: list[tuple[int, Transform]] = []
        for t in self._transforms:
            frontier.append((0, t))

        while len(self.nodes) < num_variants and frontier:
            idx = rng.randrange(len(frontier))
            parent_idx, transform = frontier.pop(idx)

            parent_ir = self.nodes[parent_idx].ir
            for child_ir in transform.candidates(parent_ir):
                if len(self.nodes) >= num_variants:
                    break
                child_source = self._render_fn(child_ir)
                if child_source in self._seen_sources:
                    continue
                self._seen_sources.add(child_source)
                child_idx = len(self.nodes)
                self.nodes.append(SearchNode(ir=child_ir, source=child_source))
                self.edges.append(SearchEdge(transform_name=transform.NAME, parent_idx=parent_idx, child_idx=child_idx))
                for t in self._transforms:
                    frontier.append((child_idx, t))
