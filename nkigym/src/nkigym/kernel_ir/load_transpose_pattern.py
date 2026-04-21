"""Load+Transpose -> NKIDMATranspose as a ``PatternRewrite``.

The rewrite fires on an adjacent ``NKILoad`` -> ``NKITranspose``
pair where the Load's output is consumed only by the transpose
AND the transpose's output is not the kernel return tensor.
Replacing the pair with a single ``NKIDMATranspose`` op encodes
the HBM dma_transpose ISA call at the graph level so codegen
can emit it directly (Phase 3 of the online-fusion plan).

This is a graph-only rewrite. Phase 2-step-2 wires it into the
pattern-rewrite driver for variant enumeration; codegen for
``NKIDMATranspose`` is staged separately.
"""

from dataclasses import dataclass, replace

from nkigym.kernel_ir.dim_analysis import DimAnalysis
from nkigym.kernel_ir.op_graph import OpGraph
from nkigym.ops.base import NKIOp
from nkigym.ops.dma import NKIDMATranspose, NKILoad
from nkigym.ops.transpose import NKITranspose


@dataclass(frozen=True)
class _Match:
    """One Load+Transpose match: the two op indices to fold."""

    load_op_idx: int
    transpose_op_idx: int


class LoadTransposePattern:
    """PatternRewrite collapsing ``NKILoad -> NKITranspose`` into ``NKIDMATranspose``.

    Match conditions (all required):

    * Load op has exactly one output tensor.
    * Load's output is consumed only by the transpose op (single
      consumer, single role).
    * Transpose op is an ``NKITranspose``.
    * Transpose's output is not the kernel return tensor —
      rewriting a return tensor's producer would break the Store
      that expects it.
    """

    name = "load_transpose"

    def match(self, da: DimAnalysis, graph: OpGraph) -> list[_Match]:
        """Return every independent Load+Transpose application."""
        matches: list[_Match] = []
        for op_idx, op_cls in enumerate(graph.op_classes):
            if op_cls is not NKILoad:
                continue
            _inputs, outputs = graph.op_tensors[op_idx]
            if len(outputs) != 1:
                continue
            load_out = outputs[0]
            consumer = _sole_consumer_via_data(graph, load_out)
            if consumer is None:
                continue
            if graph.op_classes[consumer] is not NKITranspose:
                continue
            transpose_outputs = graph.op_tensors[consumer][1]
            if transpose_outputs and transpose_outputs[0] == da.return_name:
                continue
            matches.append(_Match(load_op_idx=op_idx, transpose_op_idx=consumer))
        return matches

    def apply(self, da: DimAnalysis, graph: OpGraph, instance: _Match) -> tuple[DimAnalysis, OpGraph]:
        """Fuse the matched Load+Transpose into a single ``NKIDMATranspose``.

        The new composite takes the Load's HBM input as ``data``
        and produces the transpose's output tensor. All downstream
        consumers of the transpose output are unaffected — they
        still read the same tensor name, now produced by the
        composite. Per-op ``DimAnalysis`` arrays are reassembled in
        the new op-index order; the composite inherits the
        transpose's axis map / tile sizes / blocking set.
        """
        removed = {instance.load_op_idx, instance.transpose_op_idx}
        hbm_name = graph.op_tensors[instance.load_op_idx][0]["data"]
        composite_output = graph.op_tensors[instance.transpose_op_idx][1][0]

        new_op_classes: list[type[NKIOp]] = []
        new_op_tensors: list[tuple[dict[str, str], list[str]]] = []
        new_op_all_kwargs: list[dict[str, str]] = []
        new_axis_maps: list[dict[str, str]] = []
        new_tile_sizes: list[dict[str, int]] = []
        new_blocking: list[set[str]] = []

        for op_idx, op_cls in enumerate(graph.op_classes):
            if op_idx == instance.load_op_idx:
                new_op_classes.append(NKIDMATranspose)
                new_op_tensors.append(({"data": hbm_name}, [composite_output]))
                new_op_all_kwargs.append({"data": hbm_name})
                t_idx = instance.transpose_op_idx
                new_axis_maps.append(dict(da.per_op_axis_maps[t_idx]))
                new_tile_sizes.append(dict(da.op_tile_sizes[t_idx]))
                new_blocking.append(set(da.per_op_blocking_dims[t_idx]))
            elif op_idx in removed:
                continue
            else:
                new_op_classes.append(op_cls)
                new_op_tensors.append(graph.op_tensors[op_idx])
                new_op_all_kwargs.append(dict(graph.op_all_kwargs[op_idx]))
                new_axis_maps.append(dict(da.per_op_axis_maps[op_idx]))
                new_tile_sizes.append(dict(da.op_tile_sizes[op_idx]))
                new_blocking.append(set(da.per_op_blocking_dims[op_idx]))

        tensor_producers: dict[str, int] = {}
        for new_idx, (_inputs, outputs) in enumerate(new_op_tensors):
            for oname in outputs:
                tensor_producers[oname] = new_idx

        new_edges: list[tuple[int, int, str, str]] = []
        for new_idx, (inputs, _outputs) in enumerate(new_op_tensors):
            for role, name in inputs.items():
                producer = tensor_producers.get(name)
                if producer is not None and producer != new_idx:
                    new_edges.append((producer, new_idx, name, role))

        new_graph = OpGraph(
            op_classes=new_op_classes, edges=new_edges, op_tensors=new_op_tensors, op_all_kwargs=new_op_all_kwargs
        )
        new_da = replace(
            da, per_op_axis_maps=new_axis_maps, op_tile_sizes=new_tile_sizes, per_op_blocking_dims=new_blocking
        )
        return new_da, new_graph


def _sole_consumer_via_data(graph: OpGraph, tensor_name: str) -> int | None:
    """Return the op index that reads ``tensor_name`` via the ``data`` role iff it is unique.

    Returns None if zero or multiple ops read the tensor under any
    role, or if the sole consumer reads it under a role other than
    ``data``.
    """
    consumers: list[tuple[int, str]] = []
    for op_idx, (inputs, _outputs) in enumerate(graph.op_tensors):
        for role, name in inputs.items():
            if name == tensor_name:
                consumers.append((op_idx, role))
    result: int | None = None
    if len(consumers) == 1 and consumers[0][1] == "data":
        result = consumers[0][0]
    return result
