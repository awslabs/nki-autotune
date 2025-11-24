from compute_graph.buffer_ops import BufferNode, Matmul, TileTranspose
from compute_graph.tensors import HBMTensor


class ComputeGraph:
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[BufferNode]) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        self.operators = self._insert_transpose(operators)

    def _insert_transpose(self, operators: list[BufferNode]) -> list[BufferNode]:
        """
        Out-of-place tile level transpose.
        """
        full_operators: list[BufferNode] = []
        for operator in operators:
            if isinstance(operator, Matmul) and not operator.lhs_transposed:
                lhs_name = operator.semantic_to_name["lhs"]
                lhs_tileT = f"{lhs_name}_tileT"
                tile_transpose_op = TileTranspose(dest=lhs_tileT, data=lhs_name)
                full_operators.append(tile_transpose_op)
                operator.semantic_to_name["lhs"] = lhs_tileT
            full_operators.append(operator)
        return full_operators

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
            for semantic_name in operator.input_semantics:
                tensor_name = operator.semantic_to_name[semantic_name]
                if tensor_name in self.inputs:
                    tensor_shape = self.inputs[tensor_name]
                elif tensor_name in intermediate_tensors:
                    tensor_shape = intermediate_tensors[tensor_name]
                else:
                    raise ValueError(
                        f"Tensor '{tensor_name}' (semantic: '{semantic_name}') not found for operator {operator}"
                    )
                operator.specialize(semantic_name, tensor_shape)
            print(operator)
            assert operator.is_specialized
            for semantic_name in operator.output_semantics:
                tensor_name = operator.semantic_to_name[semantic_name]
                if tensor_name not in intermediate_tensors:
                    intermediate_tensors[tensor_name] = operator.get_tensor_shape(semantic_name)
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
