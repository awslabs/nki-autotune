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

        self.input_tensors: dict[str, HBMTensor] = {name: HBMTensor(name, axes) for name, axes in input_tensors.items()}
        self.output_tensors: dict[str, HBMTensor] = {}
        for name, axes in output_tensors.items():
            for axis in axes:
                assert axis.dependency == "parallel", "Output tensor axes must be parallel"
            self.output_tensors[name] = HBMTensor(name, axes)
        self.operators = operators

        input_num_parallel_tiles = compute_num_parallel_tiles(self.input_tensors)
        output_num_parallel_tiles = compute_num_parallel_tiles(self.output_tensors)
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
        self.edges: list[tuple[int, int]] = []
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
        local_intermediates: dict[str, int] = {}

        print(f"Subgraph {subgraph_index}, {input_tile_indices}, {output_tile_indices}")
        for operator in self.operators:
            operator_srcs: dict[str, int] = {}
            for src in operator.src:
                if src in self.input_tensors:
                    allocate_node_id = self._create_allocate_node(src, input_tile_indices[src])
                    allocated_buffer = self.nodes[allocate_node_id].dest
                    assert type(allocated_buffer) is TensorBuffer
                    load_node_id = self._create_load_node(src, input_tile_indices[src], allocated_buffer)
                    self.edges.append((allocate_node_id, load_node_id))
                    operator_srcs[src] = load_node_id
                elif src in local_intermediates:
                    parent_compute_node = local_intermediates[src]
                    operator_srcs[src] = parent_compute_node
                else:
                    raise ValueError(f"Unknown source tensor {src}. Check ComputeGraph definition.")
            if operator.dest not in local_intermediates:
                buffer_name = self._get_intermediate_name(basename=operator.dest)
                src_shapes: dict[str, tuple[int, ...]] = {}
                for src in operator_srcs:
                    node_id = operator_srcs[src]
                    buffer = self.nodes[node_id].dest
                    assert type(buffer) is TensorBuffer
                    src_shapes[src] = buffer.shape
                dest_shape = operator.forward(src_shapes)
                node_id = len(self.nodes)
                allocate_dest_node = AllocateNode(index=node_id, shape=tuple(dest_shape), dest_name=buffer_name)
                print(allocate_dest_node)
                self.nodes[node_id] = allocate_dest_node
                operator_dest_node_id = node_id
            else:
                operator_dest_node_id = local_intermediates[operator.dest]
            compute_node_id = self._create_compute_node(operator, operator_srcs, operator_dest_node_id)
            local_intermediates[operator.dest] = compute_node_id
            for src in operator_srcs:
                node_id = operator_srcs[src]
                self.edges.append((node_id, compute_node_id))
            self.edges.append((operator_dest_node_id, compute_node_id))
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

    def _create_load_node(self, tensor_name: str, tile_indices: dict[int, TileRange], dest_buffer: TensorBuffer) -> int:
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
        return node_id

    def _create_compute_node(self, operator: Operator, sources: dict[str, int], dest: int) -> int:
        """
        Args:
            operator: Operator to create compute node for
            sources: List of source TensorBuffers

        Returns:
            Node ID of the created compute node
        """
        src_buffers: dict[str, TensorBuffer] = {}
        for arg_name in sources:
            node_id = sources[arg_name]
            buffer = self.nodes[node_id].dest
            assert type(buffer) is TensorBuffer
            src_buffers[arg_name] = buffer
        dest_buffer = self.nodes[dest].dest
        assert type(dest_buffer) is TensorBuffer
        node_id = len(self.nodes)
        compute_node = ComputeNode(
            index=node_id, op=operator.op, src=src_buffers, dest=dest_buffer, params=operator.kwargs
        )
        self.nodes[node_id] = compute_node
        print(compute_node)
        return node_id

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


def compute_num_parallel_tiles(tensors: dict[str, HBMTensor]) -> int:
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
