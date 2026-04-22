"""KernelGraph: ordered list of FusionGroup nodes + group-level edges.

The sole structural dataclass. Each node is a ``FusionGroup``
(wrapping one or more ``NKIOp`` instances); ``edges`` lifts
tensor producer/consumer relationships to the group level so
downstream passes can toposort, check cross-group tensors, and
diagram the kernel without walking the ops.

Per-op resolved data (tensor wiring, kwargs, axis maps, tile
sizes, blocking dims) lives on ``KernelContext``; per-group
codegen state (``dim_order`` / ``buffer_degrees`` /
``tensor_placements``) lives on each ``FusionGroup``.
"""

import heapq
from dataclasses import dataclass, field, replace
from pathlib import Path

import graphviz

from nkigym.kernel_ir.context.context import KernelContext, TensorInfo
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.ops.base import NKIOp
from nkigym.ops.dma import NKILoad, NKIStore


@dataclass
class KernelGraph:
    """Kernel graph as a list of ``FusionGroup`` nodes with group-level edges.

    Attributes:
        groups: Ordered list of ``FusionGroup`` nodes.
        edges: ``(producer_group_idx, consumer_group_idx,
            tensor_name, role)`` tuples. Recomputed by
            ``rebuild_edges`` whenever a rewrite changes the
            group structure.
    """

    groups: list[FusionGroup]
    edges: list[tuple[int, int, str, str]] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return full group + edge detail for debugging."""
        lines = [f"KernelGraph({len(self.groups)} groups, {len(self.edges)} edges)"]
        for gi, group in enumerate(self.groups):
            lines.append(f"  group {gi}:")
            lines.extend(group.summary_lines(indent="    "))
        if self.edges:
            lines.append("  edges:")
            for gp, gc, tensor, role in self.edges:
                lines.append(f"    g{gp} -> g{gc}: {tensor} ({role})")
        return "\n".join(lines)

    def op_index_of(self, op: NKIOp) -> tuple[int, int]:
        """Return ``(group_idx, local_idx)`` for ``op``. Raises if absent."""
        for gi, group in enumerate(self.groups):
            for li, candidate in enumerate(group.ops):
                if candidate is op:
                    return gi, li
        raise ValueError(f"op {op!r} not in graph")

    def group_of(self, op: NKIOp) -> int:
        """Return the group index containing ``op``."""
        gi, _ = self.op_index_of(op)
        return gi

    def toposort_groups(self) -> list[int]:
        """Topologically sort groups by the group-level DAG."""
        num_groups = len(self.groups)
        adjacency: dict[int, list[int]] = {gi: [] for gi in range(num_groups)}
        in_degree: dict[int, int] = dict.fromkeys(range(num_groups), 0)
        seen_edges: set[tuple[int, int]] = set()
        for gp, gc, _tensor, _role in self.edges:
            if gp == gc or (gp, gc) in seen_edges:
                continue
            seen_edges.add((gp, gc))
            adjacency[gp].append(gc)
            in_degree[gc] += 1

        heap: list[tuple[int, int]] = []
        for gi in range(num_groups):
            if in_degree[gi] == 0:
                heapq.heappush(heap, (gi, gi))
        order: list[int] = []

        while heap:
            _priority, gi = heapq.heappop(heap)
            order.append(gi)
            for neighbor in adjacency[gi]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(heap, (neighbor, neighbor))

        if len(order) != num_groups:
            raise ValueError("Cycle detected in group-level DAG")

        return order

    def render(self, path: str | Path) -> Path:
        """Render the group DAG to a PNG via Graphviz."""
        dot = graphviz.Digraph(format="png")
        dot.attr(rankdir="TB", dpi="150")
        dot.attr("node", shape="box", style="rounded")

        for gi, group in enumerate(self.groups):
            op_names = ", ".join(op.NAME for op in group.ops)
            dot.node(str(gi), f"[g{gi}] {op_names}")

        for gp, gc, tensor, role in self.edges:
            dot.edge(str(gp), str(gc), label=f"{tensor} ({role})")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        dot.render(str(out), cleanup=True)
        return out.with_suffix(".png")


def insert_dma_nodes(context: KernelContext, graph: KernelGraph) -> tuple[KernelContext, KernelGraph]:
    """Return ``(context, graph)`` with explicit ``NKILoad`` / ``NKIStore`` nodes.

    Each kernel input gets one ``NKILoad`` inserted right before
    its first consumer group (lazy DFS ordering); the return
    tensor gets one ``NKIStore`` appended at the end. Every
    inserted op gets an alias tensor (``<name>_sbuf`` for loads,
    ``<name>_hbm`` for stores) and the original consumer inputs
    are rewired to the Load's output. ``KernelContext`` is
    extended with the new tensors and per-op metadata for the
    inserted ops.
    """
    load_out_of: dict[str, str] = {p: f"{p}_sbuf" for p in context.param_names}
    store_in = context.return_name
    store_out = f"{store_in}_hbm"

    op_inputs = {op: dict(v) for op, v in context.op_inputs.items()}
    op_outputs = {op: list(v) for op, v in context.op_outputs.items()}
    op_kwargs = {op: dict(v) for op, v in context.op_kwargs.items()}
    op_axis_map = {op: dict(v) for op, v in context.op_axis_map.items()}
    op_tile_sizes = {op: dict(v) for op, v in context.op_tile_sizes.items()}
    op_blocking_dims = {op: set(v) for op, v in context.op_blocking_dims.items()}

    load_ops: dict[str, NKIOp] = {}
    for p in context.param_names:
        load_op = NKILoad()
        load_ops[p] = load_op
        op_inputs[load_op] = {"data": p}
        op_outputs[load_op] = [load_out_of[p]]
        op_kwargs[load_op] = {"data": p}
        op_axis_map[load_op] = {}
        op_tile_sizes[load_op] = {}
        op_blocking_dims[load_op] = set()

    store_op = NKIStore()
    op_inputs[store_op] = {"data": store_in}
    op_outputs[store_op] = [store_out]
    op_kwargs[store_op] = {"data": store_in}
    op_axis_map[store_op] = {}
    op_tile_sizes[store_op] = {}
    op_blocking_dims[store_op] = set()

    params = set(context.param_names)
    for op in list(op_inputs):
        if op is store_op or op in load_ops.values():
            continue
        rewired = {
            role: load_out_of.get(name, name) if name in params else name for role, name in op_inputs[op].items()
        }
        op_inputs[op] = rewired

    new_tensors = _extend_tensors_with_dma(
        context.logical_tensors, context.param_names, load_out_of, store_in, store_out
    )

    new_context = replace(
        context,
        logical_tensors=new_tensors,
        op_inputs=op_inputs,
        op_outputs=op_outputs,
        op_kwargs=op_kwargs,
        op_axis_map=op_axis_map,
        op_tile_sizes=op_tile_sizes,
        op_blocking_dims=op_blocking_dims,
    )

    new_groups = _build_groups_with_dma(graph, load_ops, store_op, new_context)
    new_graph = KernelGraph(groups=new_groups)
    rebuild_edges(new_graph, new_context)
    return new_context, new_graph


def _extend_tensors_with_dma(
    tensors: dict[str, TensorInfo], param_names: list[str], load_out_of: dict[str, str], store_in: str, store_out: str
) -> dict[str, TensorInfo]:
    """Alias load / store output tensors to their source's (dim_ids, shape, dtype)."""
    result: dict[str, TensorInfo] = dict(tensors)
    for p in param_names:
        source = tensors[p]
        result[load_out_of[p]] = TensorInfo(source.dim_ids, source.shape, source.dtype)
    source = tensors[store_in]
    result[store_out] = TensorInfo(source.dim_ids, source.shape, source.dtype)
    return result


def _build_groups_with_dma(
    graph: KernelGraph, load_ops: dict[str, NKIOp], store_op: NKIOp, context: KernelContext
) -> list[FusionGroup]:
    """Insert singleton Load groups lazily before first-consuming group; append Store group at end.

    After rewiring, consumer inputs reference the Load output
    tensor name (``<param>_sbuf``), not the raw param. Map each
    Load output back to its Load op and match on that.
    """
    output_to_load: dict[str, NKIOp] = {}
    for load_op in load_ops.values():
        outputs = context.op_outputs.get(load_op, [])
        if outputs:
            output_to_load[outputs[0]] = load_op
    loaded: set[NKIOp] = set()
    new_groups: list[FusionGroup] = []
    for group in graph.groups:
        needed: list[NKIOp] = []
        for op in group.ops:
            for name in context.op_inputs.get(op, {}).values():
                load_op = output_to_load.get(name)
                if load_op is not None and load_op not in loaded:
                    needed.append(load_op)
                    loaded.add(load_op)
        for load_op in needed:
            new_groups.append(FusionGroup(ops=[load_op]))
        new_groups.append(FusionGroup(ops=list(group.ops)))
    new_groups.append(FusionGroup(ops=[store_op]))
    return new_groups


def rebuild_edges(graph: KernelGraph, context: KernelContext) -> None:
    """Recompute ``graph.edges`` from tensor producer/consumer relationships.

    For each tensor-name edge A -> B where group ``gi`` produces
    the tensor (via any of its ops' outputs) and group ``gj !=
    gi`` consumes it (via any of its ops' inputs), emit an edge
    ``(gi, gj, tensor, role)``.
    """
    producer_of: dict[str, int] = {}
    for gi, group in enumerate(graph.groups):
        for op in group.ops:
            for name in context.op_outputs.get(op, []):
                producer_of[name] = gi

    edges: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int, str, str]] = set()
    for gi, group in enumerate(graph.groups):
        for op in group.ops:
            for role, name in context.op_inputs.get(op, {}).items():
                producer = producer_of.get(name)
                if producer is None or producer == gi:
                    continue
                key = (producer, gi, name, role)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(key)
    graph.edges = edges
