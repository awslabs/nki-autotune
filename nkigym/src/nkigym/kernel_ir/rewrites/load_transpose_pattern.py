"""Load+Transpose -> NKIDMATranspose as a ``PatternRewrite``.

Operates on ``(KernelContext, KernelGraph)``. Matches a
singleton group holding ``NKILoad`` feeding a singleton group
holding ``NKITranspose`` (where the transpose has exactly one
consumer — the transpose itself — and its output is not the
kernel return tensor). Replaces both groups with one group
containing a single ``NKIDMATranspose`` instance; updates
``context.op_*`` dicts accordingly; recomputes group edges.
"""

from dataclasses import dataclass, replace

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.fusion_group import FusionGroup
from nkigym.kernel_ir.graph.graph import KernelGraph, rebuild_edges
from nkigym.ops.base import NKIOp
from nkigym.ops.dma import NKIDMATranspose, NKILoad
from nkigym.ops.transpose import NKITranspose


@dataclass(frozen=True)
class _Match:
    """One Load+Transpose match: the two group indices to fold."""

    load_group_idx: int
    transpose_group_idx: int


class LoadTransposePattern:
    """Collapse ``NKILoad``-in-group → ``NKITranspose``-in-group into ``NKIDMATranspose``-in-group."""

    name = "load_transpose"

    def match(self, context: KernelContext, graph: KernelGraph) -> list[_Match]:
        """Return every independent Load+Transpose application."""
        matches: list[_Match] = []
        for load_gi, load_group in enumerate(graph.groups):
            if len(load_group.ops) != 1 or not isinstance(load_group.ops[0], NKILoad):
                continue
            load_op = load_group.ops[0]
            outputs = context.op_outputs.get(load_op, [])
            if len(outputs) != 1:
                continue
            load_out = outputs[0]
            transpose_gi = _sole_consumer_group_via_data(context, graph, load_out)
            if transpose_gi is None:
                continue
            transpose_group = graph.groups[transpose_gi]
            if len(transpose_group.ops) != 1 or not isinstance(transpose_group.ops[0], NKITranspose):
                continue
            transpose_op = transpose_group.ops[0]
            transpose_outputs = context.op_outputs.get(transpose_op, [])
            if transpose_outputs and transpose_outputs[0] == context.return_name:
                continue
            matches.append(_Match(load_group_idx=load_gi, transpose_group_idx=transpose_gi))
        return matches

    def apply(self, context: KernelContext, graph: KernelGraph, instance: _Match) -> tuple[KernelContext, KernelGraph]:
        """Fuse Load+Transpose groups into a single NKIDMATranspose group."""
        load_group = graph.groups[instance.load_group_idx]
        transpose_group = graph.groups[instance.transpose_group_idx]
        load_op = load_group.ops[0]
        transpose_op = transpose_group.ops[0]

        hbm_name = context.op_inputs[load_op]["data"]
        composite_output = context.op_outputs[transpose_op][0]

        composite_op: NKIOp = NKIDMATranspose()
        new_op_inputs = dict(context.op_inputs)
        new_op_outputs = dict(context.op_outputs)
        new_op_kwargs = dict(context.op_kwargs)
        new_op_axis_map = dict(context.op_axis_map)
        new_op_tile_sizes = dict(context.op_tile_sizes)
        new_op_blocking_dims = dict(context.op_blocking_dims)

        new_op_inputs[composite_op] = {"data": hbm_name}
        new_op_outputs[composite_op] = [composite_output]
        new_op_kwargs[composite_op] = {"data": hbm_name}
        new_op_axis_map[composite_op] = dict(context.op_axis_map.get(transpose_op, {}))
        new_op_tile_sizes[composite_op] = dict(context.op_tile_sizes.get(transpose_op, {}))
        new_op_blocking_dims[composite_op] = set(context.op_blocking_dims.get(transpose_op, set()))

        for dead in (load_op, transpose_op):
            new_op_inputs.pop(dead, None)
            new_op_outputs.pop(dead, None)
            new_op_kwargs.pop(dead, None)
            new_op_axis_map.pop(dead, None)
            new_op_tile_sizes.pop(dead, None)
            new_op_blocking_dims.pop(dead, None)

        new_context = replace(
            context,
            op_inputs=new_op_inputs,
            op_outputs=new_op_outputs,
            op_kwargs=new_op_kwargs,
            op_axis_map=new_op_axis_map,
            op_tile_sizes=new_op_tile_sizes,
            op_blocking_dims=new_op_blocking_dims,
        )

        keep: list[FusionGroup] = []
        for gi, group in enumerate(graph.groups):
            if gi == instance.load_group_idx:
                keep.append(FusionGroup(ops=[composite_op]))
            elif gi == instance.transpose_group_idx:
                continue
            else:
                keep.append(
                    FusionGroup(
                        ops=list(group.ops),
                        dim_order=list(group.dim_order),
                        buffer_degrees=dict(group.buffer_degrees),
                        tensor_placements=dict(group.tensor_placements),
                    )
                )
        new_graph = KernelGraph(groups=keep)
        rebuild_edges(new_graph, new_context)
        return new_context, new_graph


def _sole_consumer_group_via_data(context: KernelContext, graph: KernelGraph, tensor_name: str) -> int | None:
    """Return the group index whose sole op reads ``tensor_name`` via the ``data`` role.

    Returns None if multiple ops read the tensor, or if the sole
    reader reads under a role other than ``data``, or if the
    consumer group holds more than one op.
    """
    consumers: list[tuple[int, str]] = []
    for gi, group in enumerate(graph.groups):
        for op in group.ops:
            for role, name in context.op_inputs.get(op, {}).items():
                if name == tensor_name:
                    consumers.append((gi, role))
    result: int | None = None
    if len(consumers) == 1 and consumers[0][1] == "data":
        result = consumers[0][0]
    return result
