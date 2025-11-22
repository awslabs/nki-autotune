class BufferNode:
    """Base class for on-chip operators."""

    def __init__(self, op_code: str, inputs: list[str], outputs: list[str], tensor_axes: dict[str, list[str]]) -> None:
        """
        Args:
            op_code: Operation code identifier
            inputs: List of input tensor names
            outputs: List of output tensor names
            tensor_axes: Mapping of tensor names to their axis names
        """
        self.op_code = op_code
        self.inputs = inputs
        self.outputs = outputs
        self.tensor_axes = tensor_axes
        self.axis_sizes: dict[str, dict[str, int]] = {}

    @property
    def is_specialized(self) -> bool:
        specialized = True
        for tensor in self.inputs:
            if tensor in self.axis_sizes:
                tensor_axes = self.tensor_axes[tensor]
                for axis in tensor_axes:
                    if axis not in self.axis_sizes[tensor]:
                        specialized = False
                        break
            else:
                specialized = False
                break
        return specialized

    def specialize(self, tensor_name: str, axis: str, size: int) -> None:
        if tensor_name not in self.axis_sizes:
            self.axis_sizes[tensor_name] = {}
        self.axis_sizes[tensor_name][axis] = size

    def get_tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        raise NotImplementedError(f"get_tensor_shape is not implemented for {self}")

    def codegen(self) -> str:
        """Generate NKI code for this node."""
        raise NotImplementedError(f"codegen is not implemented for {self}")

    def clear_specialization(self) -> None:
        self.axis_sizes.clear()
