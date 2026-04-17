"""OpGraph: computation DAG tracking producer-consumer dependencies.

Usage::

    from nkigym.kernel_ir.op_graph import build_op_graph

    graph = build_op_graph(my_math_func)
    print(graph)
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import graphviz
import numpy as np

from nkigym.kernel_ir.parse import find_ops
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

    def __repr__(self) -> str:
        """Return summary string with node and edge counts."""
        return f"OpGraph({len(self.op_classes)} nodes, {len(self.edges)} edges)"

    def producer_op(self, tensor_name: str) -> int | None:
        """Return the op index that produces *tensor_name*, or None if it is a kernel input."""
        producer: int | None = None
        for op_idx, (_inputs, outputs) in enumerate(self.op_tensors):
            if tensor_name in outputs:
                producer = op_idx
                break
        return producer

    def producer_isa_loc(self, tensor_name: str) -> str | None:
        """Return the ISA_LOC of the op producing *tensor_name*, or None if it is a kernel input."""
        producer = self.producer_op(tensor_name)
        loc = self.op_classes[producer].ISA_LOC if producer is not None else None
        return loc

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


def build_op_graph(func: Callable[..., np.ndarray]) -> OpGraph:
    """Build an OpGraph from a math function.

    Parses the function's NKIOp calls and constructs the DAG
    from producer-consumer tensor relationships.

    Args:
        func: Math function using NKIOp classes.

    Returns:
        The computation DAG.
    """
    ops, _ = find_ops(func)

    op_classes_list: list[type[NKIOp]] = []
    edges: list[tuple[int, int, str, str]] = []
    op_tensors: list[tuple[dict[str, str], list[str]]] = []
    op_all_kwargs: list[dict[str, str]] = []
    tensor_producers: dict[str, int] = {}

    for i, (op_cls, name_kwargs, output_names, all_kwargs) in enumerate(ops):
        op_classes_list.append(op_cls)
        op_all_kwargs.append(all_kwargs)

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
