from compute_graph.tensors import Tensor


class Memory:
    def __init__(self, name: str) -> None:
        """
        Logical tensor name to a list of node IDs that have the latest tensor
        """
        self.name = name
        self.tensors: dict[str, list[int]] = {}

    def update_tensor(self, tensor_name: str, node_id: int):
        self.tensors[tensor_name] = [node_id]

    def add_tensor(self, tensor_name: str, node_id: int):
        if tensor_name not in self.tensors:
            self.update_tensor(tensor_name, node_id)
        else:
            self.tensors[tensor_name].append(node_id)

    def __repr__(self) -> str:
        tensor_strs = [f"{self.name} ({len(self.tensors)} tensors):"]
        for tensor_name, node_ids in self.tensors.items():
            tensor_strs.append(f"  {tensor_name}: {node_ids}")
        return "\n".join(tensor_strs)


class Buffer:
    def __init__(self) -> None:
        self.logical_name_to_tensor: dict[str, Tensor] = {}

    def add(self, logical_name: str, tensor: Tensor):
        self.logical_name_to_tensor[logical_name] = tensor

    def has_tensor(self, name: str) -> bool:
        return name in self.logical_name_to_tensor

    def get_tensor(self, name: str) -> Tensor:
        tensor = self.logical_name_to_tensor[name]
        return tensor


def list_to_dict(tensor_list: list[Tensor]) -> dict[str, Tensor]:
    tensor_lookup = {tensor.name: tensor for tensor in tensor_list}
    return tensor_lookup


def compute_num_parallel_tiles(tensors: list[Tensor]) -> int:
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
