from compute_graph.tensor import Tensor


class Memory:
    """
    SBUF | PSUM | HBM
    """

    def __init__(self, location: str) -> None:
        assert location in ["SBUF", "PSUM", "HBM"], f"Illegal memory location {location}"
        self.location = location
        self.tensors: dict[str, Tensor] = {}

    def add_tensor(self, tensor: Tensor) -> None:
        assert tensor.location == self.location, f"Cannot add {tensor} to {self.location} memory"
        self.tensors[tensor.name] = tensor

    def __repr__(self) -> str:
        lines = [f"{self.location} ({len(self.tensors)} tensors):"]
        if self.tensors:
            lines.append("  Tensors:")
            for tensor in self.tensors.values():
                lines.append(f"    {tensor}")
        return "\n".join(lines)
