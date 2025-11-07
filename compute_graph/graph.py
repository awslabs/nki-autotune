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
        for subgraph_index in range(self.num_parallel_tiles):
            input_tile_indices = self._linear_index_to_coordinates(subgraph_index, self.input_tensors)
            output_tile_indices = self._linear_index_to_coordinates(subgraph_index, self.output_tensors)
            local_intermediates: dict[str, int] = {}
            print(f"Subgraph {subgraph_index}, {input_tile_indices}, {output_tile_indices}")
            for operator in self.operators:
                self._create_operator(operator, local_intermediates, input_tile_indices)
            print()

    def _create_operator(
        self,
        operator: Operator,
        local_intermediates: dict[str, int],
        input_tile_indices: dict[str, dict[int, TileRange]],
    ):
        operator_srcs: dict[str, int] = {}
        for src in operator.src:
            operator_srcs[src] = self._resolve_operator_source(src, local_intermediates, input_tile_indices)
        src_shapes = self._extract_source_shapes(operator_srcs)
        dest_buffer_node_id = self._ensure_operator_destination(operator, local_intermediates, src_shapes)
        compute_node_id = self._create_compute_node(operator, operator_srcs, dest_buffer_node_id, local_intermediates)
        self._connect_operator_edges(operator_srcs, compute_node_id, dest_buffer_node_id)

    def _resolve_operator_source(
        self, src_name: str, local_intermediates: dict[str, int], input_tile_indices: dict[str, dict[int, TileRange]]
    ) -> int:
        """Returns node ID for the given source tensor."""
        if src_name in self.input_tensors:
            src_node_id = self._create_input_tensor_node(src_name, input_tile_indices)
        elif src_name in local_intermediates:
            src_node_id = local_intermediates[src_name]
        else:
            raise ValueError(f"Unknown source tensor {src_name}")
        return src_node_id

    def _extract_source_shapes(self, operator_srcs: dict[str, int]) -> dict[str, tuple[int, ...]]:
        """Extract shapes from resolved source nodes."""
        src_shapes = {}
        for src_name, node_id in operator_srcs.items():
            buffer = self.nodes[node_id].dest
            assert type(buffer) is TensorBuffer
            src_shapes[src_name] = buffer.shape
        return src_shapes

    def _ensure_operator_destination(
        self, operator: Operator, local_intermediates: dict[str, int], src_shapes: dict[str, tuple[int, ...]]
    ) -> int:
        """Ensures destination buffer exists, returns node ID (AllocateNode for new buffer, or ComputeNode if reusing)."""
        if operator.dest in local_intermediates:
            dest_buffer_node_id = local_intermediates[operator.dest]
        else:
            dest_shape = operator.forward(src_shapes)
            buffer_name = self._get_intermediate_name(basename=operator.dest)
            dest_buffer_node_id = self._create_allocate_node(buffer_name, dest_shape)
        return dest_buffer_node_id

    def _connect_operator_edges(self, operator_srcs: dict[str, int], compute_node_id: int, dest_buffer_node_id: int):
        """Create edges from sources and destination buffer node to compute node."""
        for node_id in operator_srcs.values():
            self.edges.append((node_id, compute_node_id))
        self.edges.append((dest_buffer_node_id, compute_node_id))

    def _create_input_tensor_node(self, tensor_name: str, tile_indices: dict[str, dict[int, TileRange]]) -> int:
        """Creates allocate and load nodes for input tensor, returns load node ID."""
        hbm_tensor = self.input_tensors[tensor_name]
        buffer_shape = []
        for axis_idx, axis in enumerate(hbm_tensor.axes):
            if axis.dependency == "parallel":
                tile_range = tile_indices[tensor_name][axis_idx]
                buffer_shape.append(axis.tile_size * tile_range.num_tiles)
            else:
                buffer_shape.append(axis.size)

        buffer_name = self._get_intermediate_name(basename=tensor_name)
        allocate_node_id = self._create_allocate_node(buffer_name, tuple(buffer_shape))
        allocated_buffer = self.nodes[allocate_node_id].dest
        assert type(allocated_buffer) is TensorBuffer
        load_node_id = self._create_load_node(tensor_name, tile_indices[tensor_name], allocated_buffer)
        self.edges.append((allocate_node_id, load_node_id))
        return load_node_id

    def _create_allocate_node(self, buffer_name: str, shape: tuple[int, ...]) -> int:
        """Creates and registers an AllocateNode, returns its node ID."""
        node_id = len(self.nodes)
        allocate_node = AllocateNode(index=node_id, shape=shape, dest_name=buffer_name)
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

    def _create_compute_node(
        self, operator: Operator, sources: dict[str, int], dest: int, local_intermediates: dict[str, int]
    ) -> int:
        """
        Args:
            operator: Operator to create compute node for
            sources: List of source TensorBuffers
            dest: Destination buffer node ID
            local_intermediates: Dictionary to register the compute node as producer

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
        local_intermediates[operator.dest] = node_id
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
