import logging
from collections import defaultdict, deque
from typing import Any

from compute_graph.graph import ComputeGraph

logger = logging.getLogger(__name__)


class NKICodegen:
    """Generates NKI kernel code from a ComputeGraph."""

    def __init__(self, graph: ComputeGraph, kernel_name: str) -> None:
        """
        Args:
            graph: ComputeGraph to generate code from
            kernel_name: Name for the generated kernel function
        """
        if not graph.subgraphs:
            raise ValueError("Graph must have subgraphs before code generation")
        self.graph = graph
        self.kernel_name = kernel_name
        self.code = self._generate_kernel()

    def _generate_kernel(self) -> str:
        """Generate complete NKI kernel code.

        Returns:
            Complete NKI kernel code as a string
        """
        imports = self._generate_imports()
        decorator = "@nki.jit"
        signature = self._generate_signature(self.kernel_name)
        body = self._generate_body()

        return f"{imports}\n\n{decorator}\n{signature}\n{body}"

    def _generate_imports(self) -> str:
        """Generate import statements."""
        return (
            "import neuronxcc.nki.isa as nisa\n"
            "import neuronxcc.nki.language as nl\n"
            "import numpy as np\n"
            "from neuronxcc import nki"
        )

    def _generate_signature(self, kernel_name: str) -> str:
        """Generate function signature from graph inputs."""
        input_names = list(self.graph.hbm.input_tensors.keys())
        params_str = ", ".join(input_names)
        return f"def {kernel_name}({params_str}):"

    def _generate_body(self) -> str:
        """Generate function body with code for all subgraphs."""
        lines: list[str] = []

        for name, tensor in self.graph.hbm.input_tensors.items():
            lines.append(
                f'    assert {name}.shape == {tensor.shape}, f"Expected {name}.shape={tensor.shape}, got {{{name}.shape}}"'
            )

        if self.graph.hbm.input_tensors:
            lines.append("")

        for name, tensor in self.graph.hbm.output_tensors.items():
            shape = tensor.shape
            lines.append(f"    {name} = nl.ndarray({shape}, dtype=nl.float32, buffer=nl.shared_hbm)")

        if self.graph.hbm.output_tensors:
            lines.append("")

        for subgraph in self.graph.subgraphs:
            subgraph_lines = self._generate_subgraph(subgraph)
            lines.extend(subgraph_lines)
            lines.append("")

        output_names = list(self.graph.hbm.output_tensors.keys())
        if len(output_names) == 1:
            lines.append(f"    return {output_names[0]}")
        elif len(output_names) > 1:
            return_str = ", ".join(output_names)
            lines.append(f"    return {return_str}")

        return "\n".join(lines)

    def _generate_subgraph(self, subgraph: Any) -> list[str]:
        """Generate code for a single subgraph using topological sort."""
        lines: list[str] = []
        lines.append(f"    # Subgraph {subgraph.index}")

        # Topologically sort nodes based on edges
        sorted_indices = self._topological_sort(len(subgraph.nodes), subgraph.edges)

        for idx in sorted_indices:
            node = subgraph.nodes[idx]
            code = node.codegen()
            for code_line in code.split("\n"):
                lines.append(f"    {code_line}")

        return lines

    def _topological_sort(self, num_nodes: int, edges: list[tuple[int, int]]) -> list[int]:
        """Topologically sort nodes based on dependency edges.

        Args:
            num_nodes: Total number of nodes
            edges: List of (producer_idx, consumer_idx) tuples

        Returns:
            List of node indices in topological order
        """
        in_degree = [0] * num_nodes
        successors: dict[int, list[int]] = defaultdict(list)

        for producer, consumer in edges:
            successors[producer].append(consumer)
            in_degree[consumer] += 1

        queue = deque(i for i in range(num_nodes) if in_degree[i] == 0)
        result: list[int] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for successor in successors[node]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(result) != num_nodes:
            raise ValueError("Cycle detected in subgraph dependencies")

        return result
