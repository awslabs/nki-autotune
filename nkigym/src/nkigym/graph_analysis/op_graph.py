"""OpGraph: computation DAG tracking producer-consumer dependencies.

Usage::

    from nkigym.graph_analysis.op_graph import build_op_graph

    graph = build_op_graph(my_math_func)
    print(graph)
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import graphviz
import numpy as np

from nkigym.dim_analysis.parse import find_ops


@dataclass
class OpGraph:
    """Computation DAG.

    Attributes:
        nodes: ``op_idx -> op_type`` (ISA name).
        edges: ``(producer, consumer, tensor, role)`` tuples.
    """

    nodes: list[str]
    edges: list[tuple[int, int, str, str]]

    def __repr__(self) -> str:
        """Render the DAG as a per-node flow.

        Each node shows its inputs (tensor:role from which producer)
        and its outputs (tensor -> which consumers).
        """
        inputs: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(len(self.nodes))}
        outputs: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(len(self.nodes))}
        for producer, consumer, tensor, role in self.edges:
            inputs[consumer].append((producer, tensor, role))
            outputs[producer].append((consumer, tensor, role))

        lines = [f"OpGraph({len(self.nodes)} nodes, {len(self.edges)} edges)", ""]
        for i, op_type in enumerate(self.nodes):
            lines.append(f"  [{i}] {op_type}")
            for src, tensor, role in inputs[i]:
                lines.append(f"       <- {tensor} ({role}) from [{src}]")
            for dst, tensor, _role in outputs[i]:
                lines.append(f"       -> {tensor} to [{dst}]")

        return "\n".join(lines)

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

        for i, op_type in enumerate(self.nodes):
            dot.node(str(i), f"[{i}] {op_type}")

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

    nodes: list[str] = []
    edges: list[tuple[int, int, str, str]] = []
    tensor_producers: dict[str, int] = {}

    for i, (op_cls, name_kwargs, output_names) in enumerate(ops):
        nodes.append(op_cls.NAME)

        for role, var_name in name_kwargs.items():
            if var_name in tensor_producers:
                edges.append((tensor_producers[var_name], i, var_name, role))

        for oname in output_names:
            tensor_producers[oname] = i

    return OpGraph(nodes=nodes, edges=edges)
