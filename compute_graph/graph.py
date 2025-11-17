import copy

from compute_graph.nodes import Node
from compute_graph.operators import Allocate, Load
from compute_graph.tensors import HBMTensor, TensorBuffer


class ComputeGraph:
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[Node]) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        self.operators = operators

    def specialize(self, inputs: list[HBMTensor], output_names: list[str]) -> None:
        """Specialize graph with given input tensors and output names.

        Args:
            inputs: List of HBM input tensors
            output_names: List of output tensor names
        """
        self.num_parallel_tiles = compute_num_parallel_tiles(inputs)
        self.hbm = inputs
        self.nodes: dict[int, Node] = {}
        self.edges: list[tuple[int, int]] = []
        self.buffer_name_counter: dict[str, int] = {}
        for subgraph_index in range(self.num_parallel_tiles):
            print(f"Subgraph {subgraph_index}")
            subgraph_buffer: dict[str, int] = {}  # Track the latest tensor
            subgraph_hbm = shard_tensors(subgraph_index, self.num_parallel_tiles, inputs)
            for operator in self.operators:
                source_node_ids = self.resolve_operator_tensors(subgraph_index, operator, subgraph_hbm, subgraph_buffer)
                self._create_compute_node(source_node_ids, operator, subgraph_buffer)
                print()
            print()

    def __repr__(self) -> str:
        operators_str = "\n    ".join(str(op) for op in self.operators)

        result = f"ComputeGraph(\n" f"  operators=[\n" f"    {operators_str}\n" f"  ]\n"
        result += ")"
        return result

    def resolve_operator_tensors(
        self, subgraph_index: int, operator: Node, hbm: list[HBMTensor], buffer: dict[str, int]
    ) -> list[int]:
        """Resolve and specialize all tensors for an operator in a subgraph.

        Args:
            subgraph_index: Index of the current parallel subgraph
            operator: Operator node to resolve tensors for
            hbm: List of sharded HBM tensors for this subgraph
            buffer: Mapping of tensor names to source node IDs
        """
        operator.clear_specialization()
        print(operator)
        hbm_tensor_lookup = {hbm_tensor.name: hbm_tensor for hbm_tensor in hbm}
        source_node_ids: list[int] = []
        for tensor_name in operator.tensor_names:
            if tensor_name in buffer:
                print(f"Access {tensor_name} from buffer")
                source_node_id = buffer[tensor_name]
            elif tensor_name in hbm_tensor_lookup:
                print(f"Load {tensor_name} from HBM")
                hbm_tensor = hbm_tensor_lookup[tensor_name]
                allocate_node_id = self._create_allocate_node(
                    buffer_name=f"{tensor_name}_{subgraph_index}", shape=hbm_tensor.shape
                )
                allocate_node = self.nodes[allocate_node_id]
                assert isinstance(allocate_node, Allocate)
                load_node_id = self._create_load_node(hbm_tensor, allocate_node)
                self.edges.append((allocate_node_id, load_node_id))
                source_node_id = load_node_id
            else:
                print(f"Allocate {tensor_name} in buffer")
                tensor_shape = operator.infer_tensor_shape(tensor_name)
                source_node_id = self._create_allocate_node(
                    buffer_name=f"{tensor_name}_{subgraph_index}", shape=tensor_shape
                )
            buffer[tensor_name] = source_node_id
            source_node = self.nodes[source_node_id]
            operator.specialize_tensor(tensor_name, source_node.tensors[source_node.dest])
            source_node_ids.append(source_node_id)
        return source_node_ids

    def _create_compute_node(self, source_node_ids: list[int], operator: Node, buffer: dict[str, int]) -> int:
        assert operator.is_specialized, f"{operator} is not yet fully specialized"
        operator_copy = copy.deepcopy(operator)
        node_id = len(self.nodes)
        print(operator_copy, operator_copy.tensor_names)
        self.nodes[node_id] = operator_copy
        for source_node_id in source_node_ids:
            self.edges.append((source_node_id, node_id))
        for tensor_name in operator_copy.write_tensor_names:
            buffer[tensor_name] = node_id
        return node_id

    def _create_allocate_node(self, buffer_name: str, shape: tuple[int, ...]) -> int:
        """Creates and registers an Allocate node, returns its node ID.

        Args:
            buffer_name: Name for the allocated buffer
            shape: Shape of the tensor buffer to allocate

        Returns:
            Node ID of the created allocate node
        """
        node_id = len(self.nodes)
        allocate_node = Allocate(dest=buffer_name, shape=shape)
        allocate_node.specialize_tensor(buffer_name, TensorBuffer(buffer_name, shape))
        print(allocate_node)
        self.nodes[node_id] = allocate_node
        return node_id

    def _create_load_node(self, hbm_tensor: HBMTensor, allocate_node: Allocate) -> int:
        """Creates and registers a Load node, returns its node ID.

        Args:
            hbm_tensor: Source HBM tensor to load from
            allocate_node: Target allocate node for the loaded data

        Returns:
            Node ID of the created load node
        """
        node_id = len(self.nodes)
        load_node = Load(dest=allocate_node.dest, src=hbm_tensor.name)
        load_node.specialize_tensor(hbm_tensor.name, hbm_tensor)
        load_node.specialize_tensor(allocate_node.dest, allocate_node.tensors[allocate_node.dest])
        print(load_node)
        self.nodes[node_id] = load_node
        return node_id


def compute_num_parallel_tiles(tensors: list[HBMTensor]) -> int:
    """Compute total number of parallel tiles across all tensor axes.

    Args:
        tensors: List of HBM tensors

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
    """Shard tensors for a specific parallel tile index.

    Args:
        parallel_index: Index of the current parallel tile
        parallel_size: Total number of parallel tiles
        tensors: List of HBM tensors to shard

    Returns:
        List of sharded HBM tensors with updated tile indices
    """
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
