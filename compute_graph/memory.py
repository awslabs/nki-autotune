from compute_graph.buffer_tensor import BufferTensor
from compute_graph.hbm_tensor import HBMTensor


class HBM:
    def __init__(self) -> None:
        self.input_tensors: dict[str, HBMTensor] = {}
        self.output_tensors: dict[str, HBMTensor] = {}

    def add_input(self, tensor: HBMTensor) -> None:
        self.input_tensors[tensor.name] = tensor

    def add_output(self, tensor: HBMTensor) -> None:
        self.output_tensors[tensor.name] = tensor

    def __repr__(self) -> str:
        lines = [f"HBM ({len(self.input_tensors)} inputs, {len(self.output_tensors)} outputs):"]
        if self.input_tensors:
            lines.append("  Inputs:")
            for tensor in self.input_tensors.values():
                lines.append(f"    {tensor}")
        if self.output_tensors:
            lines.append("  Outputs:")
            for tensor in self.output_tensors.values():
                lines.append(f"    {tensor}")
        return "\n".join(lines)


class Buffer:
    """
    SBUF or PSUM buffer
    """

    def __init__(self, buffer: str) -> None:
        assert buffer in ["SBUF", "PSUM"], f"Illegal buffer type {buffer}"
        self.buffer = buffer
        self.tensors: dict[str, BufferTensor] = {}

    def add_tensor(self, tensor: BufferTensor) -> None:
        assert tensor.buffer == self.buffer, f"Cannot add {tensor} to {self.buffer} buffer"
        self.tensors[tensor.name] = tensor

    def __repr__(self) -> str:
        lines = [f"{self.buffer} ({len(self.tensors)} tensors):"]
        if self.tensors:
            lines.append("  Tensors:")
            for tensor in self.tensors.values():
                lines.append(f"    {tensor}")
        return "\n".join(lines)
