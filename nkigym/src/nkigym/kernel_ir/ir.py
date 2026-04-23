"""KernelIR: flat top-level kernel IR.

Replaces the previous (KernelIR, KernelIR, KernelIR) three-class
sandwich with a single dataclass carrying:

* kernel-wide globals (``func_name``, ``param_names``, ``return_name``,
  ``dimensions``, ``logical_tensors``, ``ltiles_per_block``);
* per-op resolved data keyed by ``NKIOp`` instance (``op_inputs``,
  ``op_outputs``, ``op_kwargs``, ``op_axis_map``, ``op_tile_sizes``,
  ``op_blocking_dims``, ``op_skip_spec``);
* fusion structure (``groups``, ``edges``) that used to live on
  ``KernelIR``.

Downstream passes (rewrites, sampler, validator, codegen) consume
``KernelIR`` directly.
"""

import heapq
from dataclasses import dataclass, field, replace
from pathlib import Path

import graphviz

from nkigym.kernel_ir.compute_skip_spec import SkipPredicate
from nkigym.kernel_ir.fusion_group import FusionGroup
from nkigym.kernel_ir.types import DimInfo, TensorInfo
from nkigym.ops.base import NKIOp
from nkigym.ops.dma import NKILoad, NKIStore


@dataclass
class KernelIR:
    """Flat kernel IR: globals + per-op data + fusion structure.

    Attributes:
        func_name: Name of the math function.
        param_names: Input parameter names.
        return_name: Name of the returned tensor.
        dimensions: ``{dim_id: DimInfo}`` — dim metadata.
        logical_tensors: ``{tensor_name: TensorInfo}`` — per-tensor
            shape/dtype/dim_ids.
        ltiles_per_block: ``{dim_id: int}`` — per-dim tiling factor.
        required_merges: Op clusters that must appear in a single
            fusion group.
        op_inputs, op_outputs, op_kwargs, op_axis_map, op_tile_sizes,
        op_blocking_dims, op_skip_spec: per-op resolved data keyed
            by ``NKIOp`` instance.
        groups: Ordered ``list[FusionGroup]``.
        edges: ``(producer_group_idx, consumer_group_idx,
            tensor_name, role)`` tuples, recomputed by
            ``rebuild_edges`` whenever a rewrite changes the group
            structure.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dimensions: dict[str, DimInfo]
    logical_tensors: dict[str, TensorInfo]
    ltiles_per_block: dict[str, int]
    required_merges: list[frozenset[int]] = field(default_factory=list)
    op_inputs: dict[NKIOp, dict[str, str]] = field(default_factory=dict)
    op_outputs: dict[NKIOp, list[str]] = field(default_factory=dict)
    op_kwargs: dict[NKIOp, dict[str, str]] = field(default_factory=dict)
    op_axis_map: dict[NKIOp, dict[str, str]] = field(default_factory=dict)
    op_tile_sizes: dict[NKIOp, dict[str, int]] = field(default_factory=dict)
    op_blocking_dims: dict[NKIOp, set[str]] = field(default_factory=dict)
    op_skip_spec: dict[NKIOp, SkipPredicate] = field(default_factory=dict)
    groups: list[FusionGroup] = field(default_factory=list)
    edges: list[tuple[int, int, str, str]] = field(default_factory=list)

    def __repr__(self) -> str:
        """Full IR detail for debugging."""
        parts = [self._repr_header(), self._repr_dimensions(), self._repr_tensors(), self._repr_ops()]
        if self.required_merges:
            parts.append(f"  required_merges: {self.required_merges}")
        parts.append(self._repr_groups_and_edges())
        return "\n".join(parts)

    def _repr_header(self) -> str:
        """Top-level identifying line."""
        return f"KernelIR(func={self.func_name}, params={self.param_names}, return={self.return_name})"

    def _repr_dimensions(self) -> str:
        """Per-dim info + ltiles_per_block."""
        lines = ["  dimensions:"]
        for dim_id, info in self.dimensions.items():
            tpb = self.ltiles_per_block.get(dim_id, 1)
            lines.append(
                f"    {dim_id}: size={info.dim_size}, "
                f"ltile={info.logical_tile_size}, ptile={info.physical_tile_size}, "
                f"role={info.role.name}, ltiles/block={tpb}"
            )
        return "\n".join(lines)

    def _repr_tensors(self) -> str:
        """Logical tensor catalog."""
        lines = ["  logical_tensors:"]
        for name, tinfo in self.logical_tensors.items():
            lines.append(f"    {name}: shape={tinfo.shape}, dims={tinfo.dim_ids}, dtype={tinfo.dtype}")
        return "\n".join(lines)

    def _repr_ops(self) -> str:
        """Per-op resolved wiring."""
        lines = [f"  ops ({len(self.op_inputs)}):"]
        for op in self.op_inputs:
            lines.extend(self._repr_one_op(op))
        return "\n".join(lines)

    def _repr_one_op(self, op: NKIOp) -> list[str]:
        """Indented lines for one op's inputs/outputs/kwargs/axes/tiles/blocking."""
        inputs = self.op_inputs.get(op, {})
        outputs = self.op_outputs.get(op, [])
        kwargs = self.op_kwargs.get(op, {})
        axis_map = self.op_axis_map.get(op, {})
        tile_sizes = self.op_tile_sizes.get(op, {})
        blocking = sorted(self.op_blocking_dims.get(op, set()))
        lines = [f"    {type(op).__name__}:", f"      inputs={inputs}, outputs={outputs}"]
        if kwargs:
            lines.append(f"      kwargs={kwargs}")
        lines.append(f"      axis_map={axis_map}, tile_sizes={tile_sizes}, blocking={blocking}")
        return lines

    def _repr_groups_and_edges(self) -> str:
        """Groups + edges block."""
        lines = [f"  groups ({len(self.groups)}) + edges ({len(self.edges)}):"]
        for gi, group in enumerate(self.groups):
            lines.append(f"  group {gi}:")
            lines.extend(group.summary_lines(indent="    "))
        if self.edges:
            lines.append("  edges:")
            for gp, gc, tensor, role in self.edges:
                lines.append(f"    g{gp} -> g{gc}: {tensor} ({role})")
        return "\n".join(lines)

    def op_input_tensors(self, op: NKIOp) -> list[str]:
        """Tensor names for op's inputs (positional + tensor-valued kwargs)."""
        names = list(self.op_inputs.get(op, {}).values())
        tensors_set = set(self.logical_tensors)
        for _name, expr in self.op_kwargs.get(op, {}).items():
            if expr in tensors_set and expr not in names:
                names.append(expr)
        return names

    def op_tensor_names(self, op: NKIOp) -> list[str]:
        """Every tensor name touched by ``op`` (inputs + outputs)."""
        return [*self.op_inputs.get(op, {}).values(), *self.op_outputs.get(op, [])]

    def op_index_of(self, op: NKIOp) -> tuple[int, int]:
        """Return ``(group_idx, local_idx)`` for ``op``. Raises if absent."""
        for gi, group in enumerate(self.groups):
            for li, candidate in enumerate(group.ops):
                if candidate is op:
                    return gi, li
        raise ValueError(f"op {op!r} not in ir")

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

    def to_json(self) -> dict:
        """Serialize the IR to a JSON-friendly dict.

        Ops are represented by stable indexes (the order they
        appear across ``self.groups``), so dicts keyed by
        ``NKIOp`` serialize as lists indexed by op position.
        """
        op_order: list[NKIOp] = [op for group in self.groups for op in group.ops]
        op_to_idx = {id(op): i for i, op in enumerate(op_order)}

        def _per_op_list(d: dict, transform=lambda v: v) -> list:
            """Convert per-op dict to list by op_order."""
            return [transform(d.get(op, None)) for op in op_order]

        return {
            "func_name": self.func_name,
            "param_names": list(self.param_names),
            "return_name": self.return_name,
            "dimensions": {
                d: {
                    "dim_size": info.dim_size,
                    "logical_tile_size": info.logical_tile_size,
                    "physical_tile_size": info.physical_tile_size,
                    "role": info.role.name,
                }
                for d, info in self.dimensions.items()
            },
            "logical_tensors": {
                name: {"dim_ids": list(tinfo.dim_ids), "shape": list(tinfo.shape), "dtype": tinfo.dtype}
                for name, tinfo in self.logical_tensors.items()
            },
            "ltiles_per_block": dict(self.ltiles_per_block),
            "required_merges": [sorted(m) for m in self.required_merges],
            "ops": [
                {
                    "idx": i,
                    "class": type(op).__name__,
                    "name": type(op).NAME,
                    "inputs": self.op_inputs.get(op, {}),
                    "outputs": self.op_outputs.get(op, []),
                    "kwargs": self.op_kwargs.get(op, {}),
                    "axis_map": self.op_axis_map.get(op, {}),
                    "tile_sizes": self.op_tile_sizes.get(op, {}),
                    "blocking_dims": sorted(self.op_blocking_dims.get(op, set())),
                    "skip_spec": repr(self.op_skip_spec[op]) if op in self.op_skip_spec else None,
                }
                for i, op in enumerate(op_order)
            ],
            "groups": [
                {
                    "ops": [op_to_idx[id(op)] for op in group.ops],
                    "dim_order": list(group.dim_order),
                    "buffer_degrees": [
                        {"kind": k, "tensor": t, "dim": d, "degree": deg}
                        for (k, t, d), deg in sorted(group.buffer_degrees.items())
                    ],
                    "tensor_placements": [
                        {"kind": k, "tensor": t, "dim": d, "tier": tier}
                        for (k, t, d), tier in sorted(group.tensor_placements.items())
                    ],
                }
                for group in self.groups
            ],
            "edges": [
                {"producer_group": gp, "consumer_group": gc, "tensor": tensor, "role": role}
                for gp, gc, tensor, role in self.edges
            ],
        }

    def render_dag(self, path: str | Path) -> Path:
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


def rebuild_edges(ir: KernelIR) -> None:
    """Recompute ``ir.edges`` from tensor producer/consumer relationships.

    For each tensor-name edge A -> B where group ``gi`` produces
    the tensor and group ``gj != gi`` consumes it, emit an edge
    ``(gi, gj, tensor, role)``.
    """
    producer_of: dict[str, int] = {}
    for gi, group in enumerate(ir.groups):
        for op in group.ops:
            for name in ir.op_outputs.get(op, []):
                producer_of[name] = gi
    edges: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int, str, str]] = set()
    for gi, group in enumerate(ir.groups):
        for op in group.ops:
            for role, name in ir.op_inputs.get(op, {}).items():
                producer = producer_of.get(name)
                if producer is None or producer == gi:
                    continue
                key = (producer, gi, name, role)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(key)
    ir.edges = edges


def insert_dma_nodes(ir: KernelIR) -> KernelIR:
    """Return a new ``KernelIR`` with explicit ``NKILoad`` / ``NKIStore`` nodes.

    Each kernel input gets one ``NKILoad`` inserted right before
    its first consumer group (lazy DFS ordering); the return
    tensor gets one ``NKIStore`` appended at the end. Every
    inserted op gets an alias tensor (``<name>_sbuf`` for loads,
    ``<name>_hbm`` for stores) and the original consumer inputs
    are rewired to the Load's output.
    """
    load_out_of: dict[str, str] = {p: f"{p}_sbuf" for p in ir.param_names}
    store_in = ir.return_name
    store_out = f"{store_in}_hbm"

    op_inputs = {op: dict(v) for op, v in ir.op_inputs.items()}
    op_outputs = {op: list(v) for op, v in ir.op_outputs.items()}
    op_kwargs = {op: dict(v) for op, v in ir.op_kwargs.items()}
    op_axis_map = {op: dict(v) for op, v in ir.op_axis_map.items()}
    op_tile_sizes = {op: dict(v) for op, v in ir.op_tile_sizes.items()}
    op_blocking_dims = {op: set(v) for op, v in ir.op_blocking_dims.items()}

    load_ops: dict[str, NKIOp] = {}
    for p in ir.param_names:
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

    params = set(ir.param_names)
    for op in list(op_inputs):
        if op is store_op or op in load_ops.values():
            continue
        rewired = {
            role: load_out_of.get(name, name) if name in params else name for role, name in op_inputs[op].items()
        }
        op_inputs[op] = rewired

    new_tensors = _extend_tensors_with_dma(ir.logical_tensors, ir.param_names, load_out_of, store_in, store_out)
    new_groups = _build_groups_with_dma(ir.groups, load_ops, store_op, op_inputs)
    new_ir = replace(
        ir,
        logical_tensors=new_tensors,
        op_inputs=op_inputs,
        op_outputs=op_outputs,
        op_kwargs=op_kwargs,
        op_axis_map=op_axis_map,
        op_tile_sizes=op_tile_sizes,
        op_blocking_dims=op_blocking_dims,
        groups=new_groups,
        edges=[],
    )
    rebuild_edges(new_ir)
    return new_ir


def _extend_tensors_with_dma(
    tensors: dict[str, TensorInfo], param_names: list[str], load_out_of: dict[str, str], store_in: str, store_out: str
) -> dict[str, TensorInfo]:
    """Alias load / store output tensors to their source's ``(dim_ids, shape, dtype)``."""
    result: dict[str, TensorInfo] = dict(tensors)
    for p in param_names:
        source = tensors[p]
        result[load_out_of[p]] = TensorInfo(source.dim_ids, source.shape, source.dtype)
    source = tensors[store_in]
    result[store_out] = TensorInfo(source.dim_ids, source.shape, source.dtype)
    return result


def _build_groups_with_dma(
    groups: list[FusionGroup], load_ops: dict[str, NKIOp], store_op: NKIOp, op_inputs: dict[NKIOp, dict[str, str]]
) -> list[FusionGroup]:
    """Insert singleton Load groups lazily before first consumer; append a Store group at the end."""
    output_to_load: dict[str, NKIOp] = {}
    for load_op in load_ops.values():
        alias = f"{op_inputs[load_op]['data']}_sbuf"
        output_to_load[alias] = load_op
    loaded: set[NKIOp] = set()
    new_groups: list[FusionGroup] = []
    for group in groups:
        needed: list[NKIOp] = []
        for op in group.ops:
            for name in op_inputs.get(op, {}).values():
                load_op = output_to_load.get(name)
                if load_op is not None and load_op not in loaded:
                    needed.append(load_op)
                    loaded.add(load_op)
        for load_op in needed:
            new_groups.append(FusionGroup(ops=[load_op]))
        new_groups.append(FusionGroup(ops=list(group.ops)))
    new_groups.append(FusionGroup(ops=[store_op]))
    return new_groups
