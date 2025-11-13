from compute_graph.nodes import Node
from compute_graph.operators import Allocate, Load
from compute_graph.tensors import HBMTensor, TensorBuffer


class ComputeGraph:
    """
    Compute graph with allocate-load-compute-store nodes and dependency edges.
    - Supply indices information for tensors involved in each node.
    - Add allocate and load nodes
    """

    def __init__(self, operators: list[Node]) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        self.operators = operators

    def specialize(self, inputs: list[HBMTensor], output_names: list[str]) -> None:
        """
        Specialize with given inputs

        Args:
            input_tensors: Dictionary mapping tensor names to axis configurations
        """
        self.num_parallel_tiles = compute_num_parallel_tiles(inputs)
        self.nodes: dict[int, Node] = {}
        self.edges: list[tuple[int, int]] = []
        self.global_intermediates: dict[str, int] = {}
        for subgraph_index in range(self.num_parallel_tiles):
            subgraph_hbm_inputs = shard_tensors(subgraph_index, self.num_parallel_tiles, inputs)
            print(f"Subgraph {subgraph_index}: {subgraph_hbm_inputs}")
            local_intermediates: dict[str, int] = {}  # Track which opeartor last modified a tensor
            """
            - Resolve HBM input tensors
            - Resolve existing tensors from local_intermediates
            - Resolve new tensors
            """
            for hbm_tensor in subgraph_hbm_inputs:
                self._create_input_tensor_node(hbm_tensor)
            # self._specialize_hbm_inputs(subgraph_hbm_inputs)
            # self._create_compute_node(operator, input_node_ids, local_intermediates)
            # self._create_operator_outputs(operator, local_intermediates)
            # for output_name in self.output_tensors:
            #     self._create_store_node(output_name, output_tile_indices[output_name], local_intermediates)
            print(local_intermediates)
            print()

    def __repr__(self) -> str:
        operators_str = "\n    ".join(str(op) for op in self.operators)

        result = f"ComputeGraph(\n" f"  operators=[\n" f"    {operators_str}\n" f"  ]\n"
        result += ")"
        return result

    def _create_input_tensor_node(self, hbm_tensor: HBMTensor) -> int:
        """
        Creates allocate and load nodes for input tensor, returns load node ID.
        Load the entire reduction axis
        """
        print(f"Creating allocate + load for {hbm_tensor}")
        buffer_name = self._get_intermediate_name(basename=hbm_tensor.name)
        allocate_node_id = self._create_allocate_node(buffer_name, hbm_tensor.shape)
        allocate_node = self.nodes[allocate_node_id]
        assert isinstance(allocate_node, Allocate)
        load_node_id = self._create_load_node(hbm_tensor, allocate_node)
        self.edges.append((allocate_node_id, load_node_id))
        return load_node_id

    def _create_allocate_node(self, buffer_name: str, shape: tuple[int, ...]) -> int:
        """Creates and registers an AllocateNode, returns its node ID."""
        node_id = len(self.nodes)
        allocate_node = Allocate(dest=buffer_name, shape=shape)
        allocate_node.specialize_tensor(buffer_name, TensorBuffer(buffer_name, shape))
        print(allocate_node)
        self.nodes[node_id] = allocate_node
        return node_id

    def _create_load_node(self, hbm_tensor: HBMTensor, allocate_node: Allocate) -> int:
        node_id = len(self.nodes)
        load_node = Load(dest=allocate_node.dest, src=hbm_tensor.name)
        load_node.specialize_tensor(hbm_tensor.name, hbm_tensor)
        load_node.specialize_tensor(allocate_node.dest, allocate_node.tensors[allocate_node.dest])
        print(load_node)
        self.nodes[node_id] = load_node
        return node_id

    def _specialize_hbm_inputs(self, hbm_inputs: list[HBMTensor]) -> None:
        # NOTE: in-place modification on operators
        for hbm_tensor in hbm_inputs:
            for operator in self.operators:
                op_tensor_names = operator.get_tensor_names()
                if hbm_tensor.name in op_tensor_names:
                    operator.specialize_tensor(hbm_tensor.name, hbm_tensor)
                    break

    def _create_operator_outputs(self, operator, local_intermediates: dict[str, int]):
        for tensor_name in operator.output_tensors:
            if tensor_name in self.output_tensors:
                raise Exception
            elif tensor_name in local_intermediates:
                node_id = local_intermediates[tensor_name]
            else:
                raise ValueError(f"Tensor '{tensor_name}' does not exist")
            local_intermediates[tensor_name] = node_id

    def _create_compute_node(self, operator, input_node_ids: list[int], local_intermediates: dict[str, int]) -> int:
        """
        Args:
            operator: Operator to create compute node for
            local_intermediates: Dictionary to register the compute node as producer

        Returns:
            Node ID of the created compute node
        """
        input_buffers: dict[str, TensorBuffer] = {}
        for node_id in input_node_ids:
            buffer = self.nodes[node_id].dest
            assert type(buffer) is TensorBuffer
            input_buffers[operator.input_tensor_names[node_id]] = buffer
        dest_buffer = self.nodes[dest].dest
        assert type(dest_buffer) is TensorBuffer
        node_id = len(self.nodes)
        compute_node = ComputeNode(
            index=node_id, operator=operator, input_tensors=src_buffers, output_tensors=dest_buffer
        )
        self.nodes[node_id] = compute_node
        print(compute_node)
        local_intermediates[operator.dest] = node_id
        return node_id

    def _create_store_node(self, output_tensor_name: str, output_tile_indices, intermediates: dict[str, int]) -> int:
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


def compute_num_parallel_tiles(tensors: list[HBMTensor]) -> int:
    """
    Args:
        tensors: Dictionary of HBM tensors

    Returns:
        Product of num_tiles for all parallel axes across tensors
    """
    num_parallel_tiles = 1
    for tensor in tensors:
        for axis in tensor.axes:
            if axis.dependency == "parallel":
                num_parallel_tiles *= axis.num_tiles
    return num_parallel_tiles


def shard_tensors(parallel_index: int, parallel_size: int, tensors: list[HBMTensor]) -> list[HBMTensor]:
    stride = parallel_size
    sharded_tensors: list[HBMTensor] = []
    for tensor in tensors:
        sharded_indices: list[tuple[int, int, int]] = []
        for axis in tensor.axes:
            if axis.dependency == "parallel":
                stride = stride // axis.num_tiles
                start_tile = (parallel_index // stride) % axis.num_tiles
                end_tile = start_tile + 1
            else:
                start_tile = 0
                end_tile = axis.num_tiles
            sharded_indices.append((start_tile, end_tile, axis.stride))
        sharded_tensor = tensor.access(sharded_indices)
        sharded_tensors.append(sharded_tensor)
    return sharded_tensors
