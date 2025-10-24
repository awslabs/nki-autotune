import math
from typing import Dict, List

from compute_graph.axes import linear_counter_to_indices, make_axes
from compute_graph.operators import ComputeNode, LoadNode, Node, Operator, StoreNode
from compute_graph.primitives import AXIS, INPUT_TENSOR_SHAPE


class ComputeGraph:
    """compute graph specification."""

    def __init__(
        self,
        input_tensors: Dict[str, INPUT_TENSOR_SHAPE],
        parallel_axes: List[AXIS],
        operators: List[Operator],
        output_tensors: List[str],
    ) -> None:
        self.input_tensors = input_tensors
        self.parallel_axes = make_axes(input_tensors, parallel_axes)
        self.operators = operators
        self.output_tensors = output_tensors
        self.num_parallel_tiles = math.prod([axis.num_tiles for axis in self.parallel_axes])
        self._generate_graph()

    def __repr__(self) -> str:
        ops_str = ",\n    ".join(str(op) for op in self.operators)
        return (
            f"ComputeGraph(\n"
            f"  input_tensors={self.input_tensors},\n"
            f"  parallel_axes={self.parallel_axes},\n"
            f"  operators=[\n"
            f"    {ops_str}\n"
            f"  ]\n"
            f")"
        )

    def _generate_graph(self) -> None:
        """Generate initial completely parallel compute graph."""
        self.nodes: Dict[int, Node] = {}
        self.edges = []
        self.variable_counter: Dict[str, int] = {}
        for parallel_counter in range(self.num_parallel_tiles):
            self._generate_subgraph(parallel_counter)

    def _generate_subgraph(self, subgraph_index: int) -> None:
        """Generate Load -> Compute -> Store subgraph for one parallel counter."""
        parallel_indices = linear_counter_to_indices(subgraph_index, self.parallel_axes)
        intermediate_outputs: Dict[str, int] = {}

        print(f"Subgraph {subgraph_index} {parallel_indices}")
        for operator in self.operators:
            source_nodes = []
            compute_node_sources: List[str] = []
            for input_tensor in operator.src:
                if input_tensor in self.input_tensors:
                    load_node = self._create_load_node(input_tensor, parallel_indices)
                    source_nodes.append(load_node)
                    compute_node_sources.append(self.nodes[load_node].dest)
                elif input_tensor in intermediate_outputs:
                    producer_node_id = intermediate_outputs[input_tensor]
                    source_nodes.append(producer_node_id)
                    compute_node_sources.append(self.nodes[producer_node_id].dest)
                else:
                    raise ValueError(
                        f"Input tensor '{input_tensor}' not found in inputs or intermediate outputs. Check your ComputeGraph dependency."
                    )
            compute_node = self._create_compute_node(
                op=operator.op, op_params=operator.params, sources=compute_node_sources, dest=operator.output
            )
            intermediate_outputs[operator.output] = compute_node
            for node_id in source_nodes:
                self.edges.append((node_id, compute_node))
            if operator.output in self.output_tensors:
                print(f"Compute node dest = {self.nodes[compute_node].dest}")
                store_node = self._create_store_node(operator.output, parallel_indices, self.nodes[compute_node].dest)
                self.edges.append((compute_node, store_node))
        print(intermediate_outputs)
        print()

    def _create_load_node(self, tensor_name: str, parallel_indices: Dict[str, Dict[int, int]]) -> int:
        """Create a load node for input tensor."""
        tensor_indices = {}
        for axis in self.parallel_axes:
            if axis.tensor_name == tensor_name:
                tile_index = parallel_indices[tensor_name][axis.axis_index]
                tensor_indices[axis.axis_index] = (tile_index, axis.tile_size)

        node_id = len(self.nodes)
        load_node = LoadNode(
            index=node_id,
            src=tensor_name,
            indices=tensor_indices,
            dest=self._get_intermediate_name(basename=f"{tensor_name}_buffer"),
        )
        print(load_node)
        self.nodes[node_id] = load_node
        return node_id

    def _create_compute_node(self, op: str, op_params: Dict, sources: List[str], dest: str) -> int:
        """Create a compute node for operator."""
        node_id = len(self.nodes)
        output_buffer = self._get_intermediate_name(basename=f"{dest}_buffer")
        compute_node = ComputeNode(index=node_id, op=op, src=sources, params=op_params, dest=output_buffer)
        print(compute_node)
        self.nodes[node_id] = compute_node
        return node_id

    def _create_store_node(self, tensor_name: str, parallel_indices: Dict[str, Dict[int, int]], src: str) -> int:
        """Create a store node for output tensor."""
        tensor_indices = {}
        for axis in self.parallel_axes:
            if axis.tensor_name == tensor_name:
                tile_index = parallel_indices[tensor_name][axis.axis_index]
                tensor_indices[axis.axis_index] = (tile_index, axis.tile_size)

        print(f"Store {tensor_name}")
        node_id = len(self.nodes)
        store_node = StoreNode(index=node_id, src=src, indices=tensor_indices, dest=f"{tensor_name}_HBM")
        self.nodes[node_id] = store_node
        return node_id

    def _get_intermediate_name(self, basename: str) -> str:
        if basename in self.variable_counter:
            self.variable_counter[basename] += 1
        else:
            self.variable_counter[basename] = 0
        return f"{basename}_{self.variable_counter[basename]}"
