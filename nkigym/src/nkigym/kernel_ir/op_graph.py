"""OpGraph: computation DAG tracking producer-consumer dependencies.

Usage::

    from nkigym.kernel_ir.op_graph import build_op_graph

    graph = build_op_graph(my_math_func, input_specs)
    print(graph)
"""

import heapq
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import graphviz
import numpy as np

from nkigym.kernel_ir.parse import find_ops
from nkigym.kernel_ir.trace import trace_scalar_kwargs
from nkigym.ops.base import NKIOp


@dataclass
class OpGraph:
    """Computation DAG.

    Attributes:
        op_classes: ``op_idx -> NKIOp subclass``. Single source
            of truth for all per-op attributes (NAME, BLOCKING_AXES,
            PSUM_DTYPE, INPUT_LOCS, ISA_LOC, format_isa_call).
        edges: ``(producer, consumer, tensor, role)`` tuples.
            Only inter-op tensors — kernel inputs with no
            producer op are absent.
        op_tensors: Per-op ``(inputs, outputs)``.
            ``inputs`` maps ``role -> tensor_name`` (including
            kernel inputs with no producer). ``outputs`` lists
            output tensor names.
        op_all_kwargs: Per-op ``{kwarg_name: source_string}``
            for all kwargs (tensors and scalars). Used by
            ``format_isa_call`` for scalar parameters.
    """

    op_classes: list[type[NKIOp]]
    edges: list[tuple[int, int, str, str]]
    op_tensors: list[tuple[dict[str, str], list[str]]]
    op_all_kwargs: list[dict[str, str]]

    def __post_init__(self) -> None:
        """Build the tensor→producer-op-index cache."""
        self._producer_cache: dict[str, int] = {
            name: op_idx for op_idx, (_inputs, outputs) in enumerate(self.op_tensors) for name in outputs
        }

    def __repr__(self) -> str:
        """Return summary string with node and edge counts."""
        return f"OpGraph({len(self.op_classes)} nodes, {len(self.edges)} edges)"

    def op_tensor_names(self, op_idx: int) -> list[str]:
        """Return every tensor name touched by ``op_idx`` (inputs + outputs)."""
        inputs, outputs = self.op_tensors[op_idx]
        return [*inputs.values(), *outputs]

    def ops_touching(self, tensor_name: str) -> list[int]:
        """Return every op index that reads or writes ``tensor_name``."""
        return [op_idx for op_idx in range(len(self.op_tensors)) if tensor_name in self.op_tensor_names(op_idx)]

    def producer_op(self, tensor_name: str) -> int | None:
        """Return the op index that produces *tensor_name*, or None if it is a kernel input."""
        return self._producer_cache.get(tensor_name)

    def producer_isa_loc(self, tensor_name: str) -> str | None:
        """Return the ISA_LOC of the op producing *tensor_name*, or None if it is a kernel input."""
        producer = self.producer_op(tensor_name)
        loc = self.op_classes[producer].ISA_LOC if producer is not None else None
        return loc

    def toposort_groups(self, fusion_groups: list[list[int]]) -> list[int]:
        """Topologically sort fusion groups by the group-level DAG.

        Each edge in ``self.edges`` is lifted to group level; ties
        broken by the minimum ``op_idx`` in each group.
        """
        num_groups = len(fusion_groups)
        op_to_group: dict[int, int] = {}
        for gi, group in enumerate(fusion_groups):
            for op_idx in group:
                op_to_group[op_idx] = gi

        adjacency: dict[int, list[int]] = {gi: [] for gi in range(num_groups)}
        in_degree: dict[int, int] = dict.fromkeys(range(num_groups), 0)
        seen_edges: set[tuple[int, int]] = set()
        for producer, consumer, _tensor, _role in self.edges:
            gp = op_to_group[producer]
            gc = op_to_group[consumer]
            if gp != gc and (gp, gc) not in seen_edges:
                seen_edges.add((gp, gc))
                adjacency[gp].append(gc)
                in_degree[gc] += 1

        heap: list[tuple[int, int]] = []
        for gi in range(num_groups):
            if in_degree[gi] == 0:
                heapq.heappush(heap, (min(fusion_groups[gi]), gi))
        order: list[int] = []

        while heap:
            _priority, gi = heapq.heappop(heap)
            order.append(gi)
            for neighbor in adjacency[gi]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(heap, (min(fusion_groups[neighbor]), neighbor))

        if len(order) != num_groups:
            raise ValueError("Cycle detected in group-level DAG")

        return order

    def render(self, path: str | Path) -> Path:
        """Render the DAG to a PNG file via Graphviz.

        Args:
            path: Output file path (without extension).

        Returns:
            Path to the rendered PNG.
        """
        dot = graphviz.Digraph(format="png")
        dot.attr(rankdir="TB", dpi="150")
        dot.attr("node", shape="box", style="rounded")

        for i, op_cls in enumerate(self.op_classes):
            dot.node(str(i), f"[{i}] {op_cls.NAME}")

        for producer, consumer, tensor, role in self.edges:
            dot.edge(str(producer), str(consumer), label=f"{tensor} ({role})")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        dot.render(str(out), cleanup=True)
        return out.with_suffix(".png")


def build_op_graph(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> OpGraph:
    """Build an OpGraph from a math function.

    Parses the function's NKIOp calls and constructs the DAG
    from producer-consumer tensor relationships. The function is
    also traced once against dummy numpy inputs so non-tensor
    scalar kwargs are captured as concrete literals (e.g.
    ``1.0 / k`` where ``k`` is a local of the math function)
    rather than as AST source strings.

    Args:
        func: Math function using NKIOp classes.
        input_specs: ``{param_name: (shape, dtype)}`` matching the
            math function's signature; used for scalar-kwarg
            tracing.

    Returns:
        The computation DAG.
    """
    ops, _ = find_ops(func)

    traced = trace_scalar_kwargs(func, input_specs)
    if len(traced) != len(ops):
        raise ValueError(f"Traced {len(traced)} op calls but AST found {len(ops)}")

    op_classes_list: list[type[NKIOp]] = []
    edges: list[tuple[int, int, str, str]] = []
    op_tensors: list[tuple[dict[str, str], list[str]]] = []
    op_all_kwargs: list[dict[str, str]] = []
    tensor_producers: dict[str, int] = {}

    for i, (op_cls, name_kwargs, output_names, all_kwargs) in enumerate(ops):
        op_classes_list.append(op_cls)
        kwargs = dict(all_kwargs)
        for kw_name, literal in traced[i].items():
            kwargs[kw_name] = literal
        op_all_kwargs.append(kwargs)

        inputs: dict[str, str] = {}
        for role, var_name in name_kwargs.items():
            if isinstance(var_name, str):
                inputs[role] = var_name
                if var_name in tensor_producers:
                    edges.append((tensor_producers[var_name], i, var_name, role))

        op_tensors.append((inputs, list(output_names)))

        for oname in output_names:
            tensor_producers[oname] = i

    return OpGraph(op_classes=op_classes_list, edges=edges, op_tensors=op_tensors, op_all_kwargs=op_all_kwargs)
