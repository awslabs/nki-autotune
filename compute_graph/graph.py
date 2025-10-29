from compute_graph.nodes import AllocateNode, ComputeNode, LoadNode, Node, StoreNode
from compute_graph.operators import Operator
from compute_graph.tensors import Axis, HBMTensor, TensorBuffer, TileRange


class ComputeGraph:
    """Compute graph with load-compute-store nodes and dependency edges."""

    def __init__(
        self,
        input_tensors: dict[str, tuple[Axis, ...]],
        operators: list[Operator],
        output_tensors: dict[str, tuple[Axis, ...]],
    ) -> None:
        """
        Args:
            input_tensors: Dictionary mapping tensor names to axis configurations
            operators: List of compute operators in the graph
            output_tensors: Dictionary mapping output tensor names to axis configurations
        """

        self.input_tensors: dict[str, HBMTensor] = {
            name: HBMTensor(name, list(axes)) for name, axes in input_tensors.items()
        }
        self.output_tensors: dict[str, HBMTensor] = {
            name: HBMTensor(name, list(axes)) for name, axes in output_tensors.items()
        }
        self.operators = operators

        input_num_parallel_tiles = self._compute_num_parallel_tiles(self.input_tensors)
        output_num_parallel_tiles = self._compute_num_parallel_tiles(self.output_tensors)
        assert input_num_parallel_tiles == output_num_parallel_tiles, "Input and output parallel tiles must match"
        self.num_parallel_tiles = input_num_parallel_tiles
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

    def _compute_num_parallel_tiles(self, tensors: dict[str, HBMTensor]) -> int:
        """
        Args:
            tensors: Dictionary of HBM tensors

        Returns:
            Product of num_tiles for all parallel axes across tensors
        """
        num_parallel_tiles = 1
        for tensor in tensors.values():
            for axis in tensor.axes:
                if axis.dependency == "parallel":
                    num_parallel_tiles *= axis.num_tiles
        return num_parallel_tiles

    def _linear_index_to_coordinates(
        self, parallel_index: int, tensors: dict[str, HBMTensor]
    ) -> dict[str, dict[int, TileRange]]:
        """
        Args:
            parallel_index: Linear counter across all parallel tiles
            tensors: Dictionary of HBM tensors

        Returns:
            Dictionary mapping tensor names to their axis tile coordinates
        """
        stride = self.num_parallel_tiles
        tile_indices: dict[str, dict[int, TileRange]] = {}

        for tensor_name, tensor in tensors.items():
            for axis_idx, axis in enumerate(tensor.axes):
                if axis.dependency == "parallel":
                    stride = stride // axis.num_tiles
                    tile_idx = (parallel_index // stride) % axis.num_tiles

                    if tensor_name not in tile_indices:
                        tile_indices[tensor_name] = {}
                    tile_range = TileRange(tile_idx, tile_idx + 1, 1)
                    tile_indices[tensor_name][axis_idx] = tile_range

        return tile_indices

    def _generate_graph(self) -> None:
        """Generate complete compute graph with all subgraphs."""
        self.nodes: dict[int, Node] = {}
        self.edges = []
        self.global_intermediates: dict[str, int] = {}
        for parallel_counter in range(self.num_parallel_tiles):
            self._generate_subgraph(parallel_counter)

    def _generate_subgraph(self, subgraph_index: int) -> None:
        """
        Args:
            subgraph_index: Index of the parallel subgraph to generate
        """
        input_tile_indices = self._linear_index_to_coordinates(subgraph_index, self.input_tensors)
        output_tile_indices = self._linear_index_to_coordinates(subgraph_index, self.output_tensors)

        print(f"Subgraph {subgraph_index}, {input_tile_indices}, {output_tile_indices}")
        loaded_inputs: dict[str, int] = {}
        for tensor_name in self.input_tensors:
            allocate_node_id = self._create_allocate_node(tensor_name, input_tile_indices[tensor_name])
            allocated_buffer = self.nodes[allocate_node_id].dest
            assert type(allocated_buffer) is TensorBuffer
            load_node_id = self._create_load_node(
                tensor_name, input_tile_indices[tensor_name], allocated_buffer, allocate_node_id
            )
            loaded_inputs[tensor_name] = load_node_id

        print(f"loaded_inputs = {loaded_inputs}")
        subgraph_intermediates: dict[str, int] = {}
        for operator in self.operators:
            compute_node = self._create_compute_node(operator, loaded_inputs, subgraph_intermediates)
            subgraph_intermediates[operator.dest] = compute_node
        for tensor_name in self.output_tensors:
            self._create_store_node(tensor_name, output_tile_indices[tensor_name], subgraph_intermediates)
        print()

    def _create_allocate_node(self, tensor_name: str, tile_indices: dict[int, TileRange]) -> int:
        """
        Args:
            tensor_name: Name of the HBM tensor to allocate buffer for
            tile_indices: Tile coordinates for all parallel axes

        Returns:
            Tuple of (node ID, allocated TensorBuffer)
        """
        hbm_tensor = self.input_tensors[tensor_name]
        buffer_shape = []
        for axis_idx, axis in enumerate(hbm_tensor.axes):
            if axis.dependency == "parallel":
                tile_range = tile_indices[axis_idx]
                buffer_shape.append(axis.tile_size * tile_range.num_tiles)
            else:
                buffer_shape.append(axis.size)

        buffer_name = self._get_intermediate_name(basename=tensor_name)
        node_id = len(self.nodes)
        allocate_node = AllocateNode(index=node_id, shape=tuple(buffer_shape), dest_name=buffer_name)
        print(allocate_node)
        self.nodes[node_id] = allocate_node
        return node_id

    def _create_load_node(
        self, tensor_name: str, tile_indices: dict[int, TileRange], dest_buffer: TensorBuffer, allocate_node_id: int
    ) -> int:
        """
        Args:
            tensor_name: Name of the HBM tensor to load
            tile_indices: Tile coordinates for all parallel axes
            dest_buffer: Pre-allocated destination buffer
            allocate_node_id: Node ID of the corresponding allocate node

        Returns:
            Node ID of the created load node
        """
        hbm_tensor = self.input_tensors[tensor_name]
        node_id = len(self.nodes)
        load_node = LoadNode(index=node_id, src=hbm_tensor, dest=dest_buffer, src_coordinates=tile_indices)
        print(load_node)
        self.nodes[node_id] = load_node
        self.edges.append((allocate_node_id, node_id))
        return node_id

    def _create_compute_node(
        self, operator: Operator, loaded_inputs: dict[str, int], intermediates: dict[str, int]
    ) -> int:
        """
        Args:
            operator: Operator to create compute node for
            subgraph_intermediates: Mapping of tensor names to node IDs

        Returns:
            Node ID of the created compute node
        """
        dest_name = self._get_intermediate_name(basename=operator.dest)
        assert dest_name not in intermediates, f"{operator} intermediate {dest_name} already exists in the subgraph"
        src_buffers: list[TensorBuffer] = []
        src_shapes: dict[str, tuple[int, ...]] = {}
        src_node_ids: list[int] = []
        for src_name in operator.src:
            if src_name in intermediates:
                src_node_id = intermediates[src_name]
            elif src_name in loaded_inputs:
                src_node_id = loaded_inputs[src_name]
            else:
                raise ValueError(f"Source tensor {src_name} not found in subgraph")
            src_node_ids.append(src_node_id)
            subgraph_src_node = self.nodes[src_node_id]
            assert type(subgraph_src_node.dest) == TensorBuffer
            src_buffers.append(subgraph_src_node.dest)
            src_shapes[src_name] = subgraph_src_node.dest.shape
        output_shape = operator.forward(src_shapes=src_shapes)

        allocate_node_id = len(self.nodes)
        allocate_node = AllocateNode(index=allocate_node_id, shape=output_shape, dest_name=dest_name)
        print(allocate_node)
        self.nodes[allocate_node_id] = allocate_node

        dest_tensor = allocate_node.dest
        compute_node_id = len(self.nodes)
        assert type(dest_tensor) is TensorBuffer
        compute_node = ComputeNode(
            index=compute_node_id, op=operator.op, src=src_buffers, dest=dest_tensor, params=operator.params
        )
        print(compute_node)
        self.nodes[compute_node_id] = compute_node

        self.edges.append((allocate_node_id, compute_node_id))
        for src_node_id in src_node_ids:
            self.edges.append((src_node_id, compute_node_id))

        return compute_node_id

    def _create_store_node(
        self, output_tensor_name: str, output_tile_indices: dict[int, TileRange], intermediates: dict[str, int]
    ) -> int:
        """
        Args:
            tensor_name: Name of the HBM tensor to store to
            tile_indices: Tile coordinates for all parallel axes
            intermediates: Mapping of tensor names to node IDs

        Returns:
            Node ID of the created store node
        """
        hbm_tensor = self.output_tensors[output_tensor_name]

        src_node_id = intermediates[output_tensor_name]
        src_node = self.nodes[src_node_id]
        assert type(src_node.dest) == TensorBuffer

        node_id = len(self.nodes)
        store_node = StoreNode(index=node_id, src=src_node.dest, dest=hbm_tensor, dest_coordinates=output_tile_indices)
        print(store_node)
        self.nodes[node_id] = store_node
        self.edges.append((src_node_id, node_id))
        return node_id

    def _get_intermediate_name(self, basename: str) -> str:
        """
        Args:
            basename: Base name for the intermediate tensor

        Returns:
            Unique name with counter suffix
        """
        if basename in self.global_intermediates:
            self.global_intermediates[basename] += 1
        else:
            self.global_intermediates[basename] = 0
        return f"{basename}_{self.global_intermediates[basename]}"
