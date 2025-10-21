import math
from typing import TYPE_CHECKING, Any, Dict, Tuple

import networkx as nx

from compute_graph.operators import Operator, Workload

if TYPE_CHECKING:
    from compute_graph.axes import Axis


class InitialGraphGenerator:
    """Generates naive completely parallel compute graphs from workload specifications."""

    def __init__(self, workload: Workload) -> None:
        self.workload = workload
        self.graph = nx.DiGraph()
        self.node_counter = 0

    def generate(self) -> nx.DiGraph:
        """Generate initial completely parallel compute graph."""
        parallel_config = self._compute_parallel_structure()
        num_parallel_counters = parallel_config["total_blocks"]

        for counter in range(num_parallel_counters):
            self._generate_subgraph(counter, parallel_config)

        return self.graph

    def _compute_parallel_structure(self) -> Dict:
        """Compute parallel tiling structure for all parallel axes."""
        parallel_info = []
        total_blocks = 1

        for tensor_name, axis_idx, tile_size in self.workload.parallel_axes:
            tensor_shape = self.workload.input_tensors[tensor_name]
            size = tensor_shape[axis_idx]
            num_tiles = math.ceil(size / tile_size)
            total_blocks *= num_tiles

            parallel_info.append(
                {
                    "tensor_name": tensor_name,
                    "axis_idx": axis_idx,
                    "tile_size": tile_size,
                    "size": size,
                    "num_tiles": num_tiles,
                }
            )

        return {"axes": parallel_info, "total_blocks": total_blocks}

    def _get_parallel_indices(self, counter: int, config: Dict) -> Dict[str, int]:
        """Convert linear counter to parallel axis indices."""
        indices = {}
        stride = config["total_blocks"]

        for axis_info in config["axes"]:
            stride = stride // axis_info["num_tiles"]
            tile_idx = (counter // stride) % axis_info["num_tiles"]
            key = f"{axis_info['tensor_name']}_{axis_info['axis_idx']}"
            indices[key] = tile_idx

        return indices

    def _generate_subgraph(self, counter: int, config: Dict) -> None:
        """Generate Load -> Compute -> Store subgraph for one parallel counter."""
        parallel_indices = self._get_parallel_indices(counter, config)

        load_nodes = {}
        for tensor_name in self.workload.input_tensors.keys():
            load_node_id = self._create_load_node(tensor_name, counter, parallel_indices)
            load_nodes[tensor_name] = load_node_id

        prev_compute_node = None
        compute_outputs = {}

        for op_idx, op in enumerate(self.workload.operators):
            compute_node_id = self._create_compute_node(op, op_idx, counter, parallel_indices)

            for input_name in op.inputs:
                if input_name in load_nodes:
                    self.graph.add_edge(load_nodes[input_name], compute_node_id)
                elif input_name in compute_outputs:
                    self.graph.add_edge(compute_outputs[input_name], compute_node_id)

            if prev_compute_node is not None:
                if not self.graph.has_edge(prev_compute_node, compute_node_id):
                    self.graph.add_edge(prev_compute_node, compute_node_id)

            compute_outputs[f"op_{op_idx}_output"] = compute_node_id
            prev_compute_node = compute_node_id

        store_node_id = self._create_store_node(counter, parallel_indices)
        self.graph.add_edge(prev_compute_node, store_node_id)

    def _create_load_node(self, tensor_name: str, counter: int, parallel_indices: Dict[str, int]) -> int:
        """Create a load node for input tensor."""
        node_id = self.node_counter
        self.node_counter += 1

        tile_indices = {}
        for key, tile_idx in parallel_indices.items():
            if key.startswith(tensor_name):
                axis_idx = int(key.split("_")[-1])
                tile_indices[axis_idx] = tile_idx

        self.graph.add_node(
            node_id,
            type="load",
            tensor_name=tensor_name,
            tile_indices=tile_indices,
            buffer_name=f"{tensor_name}_buffer_{counter}",
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


class ComputeGraph:
    """Manages parallel axis sharding for standard fusion chains."""

    def __init__(self, parallel_axes: "Tuple[Axis]") -> None:
        """Initialize chain with parallel axis configurations."""
        self.parallel_axes = parallel_axes
        num_parallel_blocks = 1
        for parallel_axis in parallel_axes:
            num_parallel_blocks *= parallel_axis.num_blocks
        self.num_parallel_blocks = num_parallel_blocks

    def __call__(self, input_tensors: Dict[str, Any], verbose: bool = False):
        """Execute fusion chain with given input tensors."""
        import neuronxcc.nki.language as nl

        for counter in nl.affine_range(self.num_parallel_blocks):
            stride = self.num_parallel_blocks
            for parallel_axis in self.parallel_axes:
                stride = stride // parallel_axis.num_blocks
                block_index = (counter // stride) % parallel_axis.num_blocks
