from compute_graph.buffer_ops import BufferNode
from compute_graph.tensors import HBMTensor


class ComputeGraph:
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[BufferNode]) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        self.operators = operators

    def specialize(self, inputs: dict[str, tuple[int, ...]], output: str) -> None:
        """Specialize compute graph with given input and output tensors.

        Args:
            inputs: Dictionary mapping tensor names to shapes
            output: Name of the HBM output tensor
        """
        self.inputs = inputs
        self.output = output
        self._trace()

    def _trace(self) -> None:
        intermediate_tensors: dict[str, tuple[int, ...]] = {}
        for operator in self.operators:
            print("-" * 10, f"{operator}", "-" * 10)
            for tensor_name in operator.inputs:
                if tensor_name in self.inputs:
                    tensor_shape = self.inputs[tensor_name]
                elif tensor_name in intermediate_tensors:
                    tensor_shape = intermediate_tensors[tensor_name]
                else:
                    raise ValueError(f"Tensor {tensor_name} does not exist")
                operator.specialize(tensor_name, tensor_shape)
            print(operator)
            for tensor_name in operator.outputs:
                if tensor_name not in intermediate_tensors:
                    intermediate_tensors[tensor_name] = operator.get_tensor_shape(tensor_name)
            print()


def shard_tensor(parallel_index: int, parallel_size: int, tensor: HBMTensor) -> HBMTensor:
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
