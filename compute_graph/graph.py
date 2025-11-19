import neuronxcc.nki.language as nl

from compute_graph.memory import Memory
from compute_graph.nodes import Node
from compute_graph.operators import Allocate, Load
from compute_graph.tensors import Axis, Tensor


class ComputeGraph:
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[Node]) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        self.operators = operators

    def specialize(self, inputs: dict[str, tuple[int, ...]], outputs: list[str]) -> None:
        """Specialize graph with given input and output tensors.

        Args:
            inputs: List of HBM input tensors
            outputs: List of HBM output tensors
        """
        self.hbm = inputs
        self.buffer = Memory("Buffer")
        self.nodes: dict[int, Node] = {}
        self.edges: list[tuple[int, int]] = []
        for operator in self.operators:
            self.specialize_operator(operator)

    def specialize_operator(self, operator: Node) -> list[int]:
        print("-" * 10, f"{operator}", "-" * 10)
        for tensor_name in operator.read_tensor_names:
            if tensor_name in self.buffer.tensors:
                tensor = self.buffer.tensors[tensor_name]
                print(f"Read {tensor} from buffer")
                raise NotImplementedError
            elif tensor_name in self.hbm:
                self._load_from_hbm(tensor_name)
            else:
                raise ValueError(f"{tensor_name} does not exist")
        for tensor_name in operator.write_tensor_names:
            if tensor_name in self.buffer.tensors:
                tensor = self.buffer.tensors[tensor_name]
                print(f"Write {tensor} to buffer")
                raise NotImplementedError
            else:
                self._allocate_output_tensor(operator, tensor_name)
        print()

    def _load_from_hbm(self, hbm_tensor: str):
        print(f"Load {hbm_tensor} from HBM")
        pmax = nl.tile_size.pmax
        par_size, free_size = self.hbm[hbm_tensor]
        assert par_size % pmax == 0
        num_load_nodes = par_size // pmax
        free_axis = Axis(start_tile=0, end_tile=1, stride=1, tile_size=free_size)
        for i in range(num_load_nodes):
            par_axis = Axis(start_tile=i, end_tile=i + 1, stride=1, tile_size=pmax)
            loaded_tensor = Tensor(name=f"{hbm_tensor}_{i}", axes=(par_axis, free_axis))
            allocate_node = Allocate(dest=loaded_tensor.name, shape=loaded_tensor.shape, buffer="nl.sbuf")
            self.add_node(allocate_node)
            load_node = Load(dest=allocate_node.dest, src=hbm_tensor, axes=(par_axis, free_axis))
            load_node_id = self.add_node(load_node)
            self.buffer.add_tensor(hbm_tensor, load_node_id)

    def _allocate_output_tensor(self, operator: Node, tensor_name: str):
        print(f"Allocate new {tensor_name}")
        print(operator.read_tensor_names)
        print(self.buffer)
        tensor_shape = operator.infer_tensor_shape(tensor_name)

    def add_node(self, node: Node) -> int:
        node_id = len(self.nodes)
        self.nodes[node_id] = node
        print(node)
        return node_id


def shard_tensor(parallel_index: int, parallel_size: int, tensor: str):
    """Shard a tensor for a specific parallel tile index.
    Args:
        parallel_index: Index of the current parallel tile
        parallel_size: Total number of parallel tiles
        tensor: HBM tensor to shard

    Returns:
        Sharded HBM tensor with updated tile indices
    """
    stride = parallel_size
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
    return sharded_tensor
