import math
from typing import Dict, List

import networkx as nx

from compute_graph.axes import linear_counter_to_indices, make_axes
from compute_graph.operators import Operator
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
        self.graph = nx.DiGraph()
        for parallel_counter in range(self.num_parallel_tiles):
            self._generate_subgraph(parallel_counter)

    def _generate_subgraph(self, counter: int) -> None:
        """Generate Load -> Compute -> Store subgraph for one parallel counter."""
        print(f"Generating subgraph {counter}")
        parallel_indices = linear_counter_to_indices(counter, self.parallel_axes)

        """
        Go through the operators
        For each operator:
        Add load nodes if it needs input tensors from self.input_tensors
        Add compute node for the operation itself
        If the node output is in self.output_tensors, store to output HBM
        """
        print(parallel_indices)
        print(self.parallel_axes)
        for operator in self.operators:
            for input_tensor in operator.inputs:
                if input_tensor in self.input_tensors:
                    load_node = self._create_load_node(input_tensor, counter, parallel_indices)

    def _create_load_node(self, tensor_name: str, counter: int, parallel_indices: Dict[str, Dict[int, int]]) -> int:
        """Create a load node for input tensor."""
        node_id = self.graph.number_of_nodes()
        tile_indices = {}
        for axis in self.parallel_axes:
            if axis.tensor_name == tensor_name:
                tile_index = parallel_indices[tensor_name][axis.axis_index]
                tile_indices[axis.axis_index] = tile_index

        self.graph.add_node(
            node_id,
            type="load",
            tensor_name=tensor_name,
            tile_indices=tile_indices,
            buffer_name=f"{tensor_name}_sbuf_{counter}",
            parallel_counter=counter,
        )
        return node_id

    def _create_compute_node(self, op: Operator, op_idx: int, counter: int, parallel_indices: Dict[str, int]) -> int:
        """Create a compute node for operator."""
        node_id = self.node_counter
        self.node_counter += 1

        self.graph.add_node(
            node_id,
            type="compute",
            op_type=op.op_type,
            op_index=op_idx,
            inputs=op.inputs,
            params=op.params,
            output_buffer=f"op_{op_idx}_buffer_{counter}",
            parallel_counter=counter,
        )
        return node_id

    def _create_store_node(self, counter: int, parallel_indices: Dict[str, int]) -> int:
        """Create a store node for output tensor."""
        node_id = self.node_counter
        self.node_counter += 1

        self.graph.add_node(
            node_id,
            type="store",
            tensor_name=self.workload.output_tensor,
            tile_indices=parallel_indices,
            parallel_counter=counter,
        )
        return node_id
