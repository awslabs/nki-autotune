from compute_graph.hbm_ops import Node
from compute_graph.tensors import Tensor, create_tensor


class ComputeGraph:
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[Node]) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        self.operators = operators

    def trace(self, inputs: dict[str, tuple[int, ...]], outputs: list[str]) -> None:
        """Trace compute graph with given input and output tensors.

        Args:
            inputs: List of HBM input tensors
            outputs: List of HBM output tensors
        """
        self.tensors: dict[str, Tensor] = {}
        for tensor_name in inputs:
            shape = inputs[tensor_name]
            self.tensors[tensor_name] = create_tensor(tensor_name, shape)
        for operator in self.operators:
            self.specialize_operator(operator)

    def specialize_operator(self, operator: Node) -> None:
        print("-" * 10, f"{operator}", "-" * 10)
        for tensor_name in operator.read_tensor_names:
            tensor = self.tensors[tensor_name]
            operator.specialize(tensor_name, tensor)
        for tensor_name in operator.write_tensor_names:
            if tensor_name not in self.tensors:
                shape = operator.infer_tensor_shape(tensor_name)
                self.tensors[tensor_name] = create_tensor(tensor_name, shape)
            tensor = self.tensors[tensor_name]
            operator.specialize(tensor_name, tensor)
        print(operator)
        print()


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
