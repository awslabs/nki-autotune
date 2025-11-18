from compute_graph.graph import ComputeGraph


class NKICodegen:
    """Generates NKI kernel code from a ComputeGraph."""

    def __init__(self, graph: ComputeGraph) -> None:
        """
        Args:
            graph: Specialized ComputeGraph to generate code from
        """
        if not hasattr(graph, "nodes") or not graph.nodes:
            raise ValueError("Graph must be specialized before code generation")
        self.graph = graph

    def generate_kernel(self, kernel_name: str) -> str:
        """Generate complete NKI kernel code.

        Args:
            kernel_name: Name for the generated kernel function

        Returns:
            Complete NKI kernel code as a string
        """
        imports = self._generate_imports()
        decorator = "@nki.jit"
        signature = self._generate_signature(kernel_name)
        body = self._generate_body()

        kernel_code = f"{imports}\n\n{decorator}\n{signature}\n{body}"
        return kernel_code

    def _generate_imports(self) -> str:
        """Generate import statements."""
        return (
            "import neuronxcc.nki.isa as nisa\n"
            "import neuronxcc.nki.language as nl\n"
            "import numpy as np\n"
            "from neuronxcc import nki"
        )

    def _generate_signature(self, kernel_name: str) -> str:
        """Generate function signature from graph inputs only."""
        input_names = [tensor.name for tensor in self.graph.hbm]
        params_str = ", ".join(input_names)
        return f"def {kernel_name}({params_str}):"

    def _generate_body(self) -> str:
        """Generate function body with code for all nodes."""
        lines = []

        for output_tensor in self.graph.outputs:
            shape = output_tensor.shape
            lines.append(f"    {output_tensor.name} = nl.ndarray({shape}, dtype=nl.float32, buffer=nl.shared_hbm)")

        if self.graph.outputs:
            lines.append("")

        current_subgraph = None

        for node_id in sorted(self.graph.nodes.keys()):
            node = self.graph.nodes[node_id]
            subgraph_idx = self._get_subgraph_index(node)

            if subgraph_idx != current_subgraph:
                if current_subgraph is not None:
                    lines.append("")
                lines.append(f"    # Subgraph {subgraph_idx}")
                current_subgraph = subgraph_idx

            code = node.codegen()
            for code_line in code.split("\n"):
                lines.append(f"    {code_line}")

        if self.graph.outputs:
            lines.append("")
            output_names = [tensor.name for tensor in self.graph.outputs]
            if len(output_names) == 1:
                lines.append(f"    return {output_names[0]}")
            else:
                return_str = ", ".join(output_names)
                lines.append(f"    return {return_str}")

        return "\n".join(lines)

    def _get_subgraph_index(self, node) -> int:
        """Extract subgraph index from node's actual buffer names.

        Args:
            node: Node to extract subgraph index from

        Returns:
            Subgraph index (0-based)
        """
        for tensor in node.tensors.values():
            tensor_name = tensor.name
            if "_" in tensor_name:
                parts = tensor_name.split("_")
                if parts[-1].isdigit():
                    return int(parts[-1])
        return 0
