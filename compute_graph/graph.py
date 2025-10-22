import math
from typing import Dict, List, Tuple

from compute_graph.axes import linear_counter_to_indices, make_axes
from compute_graph.operators import Operator
from compute_graph.primitives import AXIS, INPUT_TENSOR_SHAPE


class LoadNode:
    type = "load"

    def __init__(self, index: int, input_tensor: str, load_indices: Dict[int, Tuple[int, int]]) -> None:
        self.index = index
        self.input_tensor = input_tensor
        self.load_indices = load_indices

    def __repr__(self) -> str:
        return (
            f"LoadNode("
            f"index={self.index}, "
            f"input_tensor='{self.input_tensor}', "
            f"load_indices={self.load_indices})"
        )


class ComputeNode:
    type = "compute"

    def __init__(self, index: int, op_type: str, inputs: List[str], params: Dict, output_buffer: str) -> None:
        self.index = index
        self.op_type = op_type
        self.inputs = inputs
        self.params = params
        self.output_buffer = output_buffer

    def __repr__(self) -> str:
        return (
            f"ComputeNode("
            f"index={self.index}, "
            f"op_type='{self.op_type}', "
            f"inputs={self.inputs}, "
            f"output_buffer='{self.output_buffer}')"
        )


class StoreNode:
    type = "store"

    def __init__(self, index: int, output_tensor: str, store_indices: Dict[int, Tuple[int, int]]) -> None:
        self.index = index
        self.output_tensor = output_tensor
        self.store_indices = store_indices

    def __repr__(self) -> str:
        return (
            f"StoreNode("
            f"index={self.index}, "
            f"output_tensor='{self.output_tensor}', "
            f"store_indices={self.store_indices})"
        )


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
        print(self.parallel_axes)
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
        self.nodes = {}
        self.edges = []
        for parallel_counter in range(self.num_parallel_tiles):
            self._generate_subgraph(parallel_counter)

    def _generate_subgraph(self, subgraph_index: int) -> None:
        """Generate Load -> Compute -> Store subgraph for one parallel counter."""
        print(f"Generating subgraph {subgraph_index}")
        parallel_indices = linear_counter_to_indices(subgraph_index, self.parallel_axes)

        """
        Go through the operators
        For each operator:
        Add load nodes if it needs input tensors from self.input_tensors
        Add compute node for the operation itself
        If the node output is in self.output_tensors, store to output HBM
        """
        print(parallel_indices)
        for operator in self.operators:
            source_nodes = []
            for input_tensor in operator.inputs:
                if input_tensor in self.input_tensors:
                    load_node = self._create_load_node(input_tensor, parallel_indices)
                    source_nodes.append(load_node)
            # self._create_compute_node(operator, counter, parallel_indices)

    def _create_load_node(self, tensor_name: str, parallel_indices: Dict[str, Dict[int, int]]) -> int:
        """Create a load node for input tensor."""
        tensor_indices = {}
        for axis in self.parallel_axes:
            if axis.tensor_name == tensor_name:
                tile_index = parallel_indices[tensor_name][axis.axis_index]
                tensor_indices[axis.axis_index] = (tile_index, axis.tile_size)

        node_id = len(self.nodes)
        self.nodes[node_id] = LoadNode(index=node_id, input_tensor=tensor_name, load_indices=tensor_indices)
        return node_id

    def _create_compute_node(self, op: Operator, counter: int, parallel_indices: Dict[str, Dict[int, int]]) -> int:
        """Create a compute node for operator."""
        node_id = self.graph.number_of_nodes()
        self.graph.add_node(
            node_id,
            type="compute",
            op_type=op.op_type,
            inputs=op.inputs,
            params=op.params,
            output_buffer=f"{op.outputs}_buffer",
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
