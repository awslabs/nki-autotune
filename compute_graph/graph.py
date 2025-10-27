from compute_graph.nodes import ComputeNode, LoadNode, Node, StoreNode
from compute_graph.operators import Operator
from compute_graph.primitives import TENSOR
from compute_graph.tensors import HBMTensor, TensorBuffer, TensorCoordinate, compute_num_parallel_tiles


class ComputeGraph:
    """compute graph specification."""

    def __init__(self, input_tensors: list[HBMTensor], operators: list[Operator], output_tensors: list[str]) -> None:
        self.input_tensors = input_tensors
        print(self.input_tensors)
        self.operators = operators
        self.output_tensors = output_tensors
        self.num_parallel_tiles = compute_num_parallel_tiles(input_tensors)
        self._generate_graph()

    def __repr__(self) -> str:
        ops_str = ",\n    ".join(str(op) for op in self.operators)
        return (
            f"ComputeGraph(\n"
            f"  input_tensors={self.input_tensors},\n"
            f"  operators=[\n"
            f"    {ops_str}\n"
            f"  ]\n"
            f"  output_tensors={self.output_tensors}\n"
            f")"
        )

    def _compute_tile_indices(self, parallel_index: int) -> dict:
        """
        Compute tile indices for all parallel axes given a parallel iteration counter.

        Args:
            parallel_index: Linear counter across all parallel tiles (0 to num_parallel_tiles-1)

        Returns:
            Dictionary mapping {tensor_name: {axis_idx: tile_idx}}
        """
        parallel_axes = []
        for tensor in self.input_tensors:
            for axis_idx, axis in enumerate(tensor.axes):
                if axis.dependency == "parallel":
                    parallel_axes.append((tensor.name, axis_idx, axis.num_tiles))

        tile_indices = {}
        stride = self.num_parallel_tiles

        for tensor_name, axis_idx, num_tiles in parallel_axes:
            stride = stride // num_tiles
            tile_idx = (parallel_index // stride) % num_tiles

            if tensor_name not in tile_indices:
                tile_indices[tensor_name] = {}
            tile_indices[tensor_name][axis_idx] = tile_idx

        return tile_indices

    def _generate_graph(self) -> None:
        """Generate initial completely parallel compute graph."""
        self.nodes: dict[int, Node] = {}
        self.edges = []
        self.intermediates_tracker: dict[str, int] = {}
        for parallel_counter in range(self.num_parallel_tiles):
            self._generate_subgraph(parallel_counter)

    def _generate_subgraph(self, subgraph_index: int) -> None:
        """Generate Load -> Compute -> Store subgraph for one parallel counter."""
        tile_indices = self._compute_tile_indices(subgraph_index)

        print(f"Subgraph {subgraph_index}, tile_indices: {tile_indices}")
        for hbm_tensor in self.input_tensors:
            load_node = self._create_load_node(hbm_tensor, tile_indices[hbm_tensor.name])
        print()

    def _create_load_node(self, hbm_tensor: HBMTensor, tile_indices: dict[int, int]) -> int:
        """
        Create a load node for input tensor.

        Args:
            hbm_tensor_name: Name of the HBM tensor to load
            tile_indices: Precomputed tile indices for all parallel axes
                Format: {tensor_name: {axis_idx: tile_idx}}

        Returns:
            Node ID of the created load node
        """
        hbm_coordinates = []
        for axis_idx, axis in enumerate(hbm_tensor.axes):
            if axis.dependency == "parallel":
                tile_idx = tile_indices[axis_idx]
                hbm_coordinates.append(
                    TensorCoordinate(start_tile_index=tile_idx, num_tiles=1, tile_size=axis.tile_size)
                )
            else:
                hbm_coordinates.append(
                    TensorCoordinate(start_tile_index=0, num_tiles=axis.num_tiles, tile_size=axis.tile_size)
                )
        buffer_name = self._get_intermediate_name(basename=hbm_tensor.name)
        dest_tensor = TensorBuffer(name=buffer_name, hbm_coordinates=hbm_coordinates)

        node_id = len(self.nodes)
        load_node = LoadNode(index=node_id, src=hbm_tensor, dest=dest_tensor)
        print(load_node)
        self.nodes[node_id] = load_node
        return node_id

    def _create_compute_node(self, op: str, op_params: dict, sources: list[TENSOR], dest: str) -> int:
        """Create a compute node for operator."""
        node_id = len(self.nodes)
        dest_name = self._get_intermediate_name(basename=dest)
        compute_node = ComputeNode(index=node_id, op=op, src=sources, params=op_params, dest=(dest_name, {}))
        print(compute_node)
        self.nodes[node_id] = compute_node
        return node_id

    def _create_store_node(self, tensor_name: str, parallel_indices: dict[str, dict[int, int]], src: str) -> int:
        """Create a store node for output tensor."""
        tensor_indices = {}
        for axis in self.parallel_axes:
            if axis.tensor_name == tensor_name:
                tile_index = parallel_indices[tensor_name][axis.axis_index]
                tensor_indices[axis.axis_index] = (tile_index, axis.tile_size)

        node_id = len(self.nodes)
        store_node = StoreNode(index=node_id, src=(src, tensor_indices), dest=(f"{tensor_name}_HBM", {}))
        print(store_node)
        self.nodes[node_id] = store_node
        return node_id

    def _get_intermediate_name(self, basename: str) -> str:
        if basename in self.intermediates_tracker:
            self.intermediates_tracker[basename] += 1
        else:
            self.intermediates_tracker[basename] = 0
        return f"{basename}_{self.intermediates_tracker[basename]}"
