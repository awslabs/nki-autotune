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
        self.axis_sizes: dict[str, int] = {}

    @property
    def is_specialized(self) -> bool:
        specialized = True
        for tensor in self.inputs:
            for axis in self.tensor_axes[tensor]:
                if axis not in self.axis_sizes:
                    specialized = False
                    break
        return specialized

    def specialize(self, axis: str, size: int) -> None:
        if axis in self.axis_sizes:
            raise ValueError(f"Cannot overwrite axis size {axis} in {self}.")
        self.axis_sizes[axis] = size

    def get_tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        """Get the shape of a tensor by looking up axis sizes.

        Args:
            tensor_name: Name of the tensor to get shape for

        Returns:
            Tuple of axis sizes representing the tensor shape

        Raises:
            ValueError: If tensor_name is not in tensor_axes or if any required axis is not specialized
        """
        if tensor_name not in self.tensor_axes:
            raise ValueError(f"Tensor '{tensor_name}' not found in tensor_axes of {self}")

        axes = self.tensor_axes[tensor_name]
        shape = []

        for axis in axes:
            if axis not in self.axis_sizes:
                raise ValueError(f"Axis '{axis}' for tensor '{tensor_name}' is not specialized yet in {self}. ")
            shape.append(self.axis_sizes[axis])

        return tuple(shape)

    def codegen(self) -> str:
        """Generate NKI code for this node."""
        raise NotImplementedError(f"codegen is not implemented for {self}")

    def clear_specialization(self) -> None:
        self.axis_sizes.clear()


class TensorScalar(BufferNode):
    """
    Tensor-scalar operator.
    dest[...] = nisa.tensor_scalar(data, op0, operand0, op1, operand1)
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nisa.tensor_scalar", dest=dest, **kwargs)

    def infer_tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        data_tensor_name = self.kwargs["data"]
        assert (
            data_tensor_name in self.tensors
        ), f"Data tensor {data_tensor_name} not specialized. Cannot infer tensor shapes."
        data_tensor = self.tensors[data_tensor_name]
        data_shape = data_tensor.shape
        if tensor_name == self.dest:
            tensor_shape = data_shape
        else:
            raise ValueError(f"Tensor name {tensor_name} not found in {self}")
        return tensor_shape

    @property
    def read_tensor_names(self) -> list[str]:
        tensor_names = [self.kwargs["data"]]
        if isinstance(self.kwargs["operand0"], str):
            tensor_names.append(self.kwargs["operand0"])
        if "operand1" in self.kwargs and isinstance(self.kwargs["operand1"], str):
            tensor_names.append(self.kwargs["operand1"])
        return tensor_names

    @property
    def write_tensor_names(self) -> list[str]:
        return [self.dest]


class Activation(BufferNode):
    """
    Activation operator with optional reduce.
    With reduce:
        activate_res[...] = nisa.activation(op, data[...], reduce_op, reduce_res[...])
    No reduce:
        activate_res[...] = nisa.activation(op, data[...])
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nisa.activation", dest=dest, **kwargs)

    def infer_tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        data_tensor_name = self.kwargs["data"]
        assert (
            data_tensor_name in self.tensors
        ), f"Data tensor {data_tensor_name} not specialized. Cannot infer tensor shapes."
        data_tensor = self.tensors[data_tensor_name]
        data_shape = data_tensor.shape
        if tensor_name == self.kwargs["reduce_res"]:
            tensor_shape = (*data_shape[:-1], 1)
        elif tensor_name == self.dest:
            tensor_shape = data_shape
        else:
            raise ValueError(f"Tensor name {tensor_name} not found in {self}")
        return tensor_shape

    @property
    def read_tensor_names(self) -> list[str]:
        return [self.kwargs["data"]]

    @property
    def write_tensor_names(self) -> list[str]:
        write_names = [self.dest]
        if "reduce_res" in self.kwargs:
            write_names.append(self.kwargs["reduce_res"])
        return write_names


class Transpose(BufferNode):
    """Transpose operator."""

    def __init__(self, dest: str, **kwargs) -> None:
        """
        Args:
            data: Source tensor name
            dest: Destination tensor name
            transpose_axes: List of axes to transpose
        """
        super().__init__(op_code="nisa.nc_transpose", dest=dest, **kwargs)

    @property
    def read_tensor_names(self) -> list[str]:
        return [self.kwargs["data"]]

    @property
    def write_tensor_names(self) -> list[str]:
        return [self.dest]


class Matmul(BufferNode):
    """Matrix multiplication operator."""

    def __init__(self, dest: str, **kwargs) -> None:
        """
        Args:
            src: List of two source tensor names [A, B]
            dest: Destination tensor name
        """
        super().__init__(op_code="nisa.nc_matmul", dest=dest, **kwargs)

    def infer_tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        lhs_tensor_name = self.kwargs["stationary"]
        rhs_tensor_name = self.kwargs["moving"]
        assert (
            lhs_tensor_name in self.tensors
        ), f"lhs_tensor {lhs_tensor_name} not specialized. Cannot infer tensor shapes."
        assert (
            rhs_tensor_name in self.tensors
        ), f"rhs_tensor {rhs_tensor_name} not specialized. Cannot infer tensor shapes."
        lhs_tensor = self.tensors[lhs_tensor_name]
        rhs_tensor = self.tensors[rhs_tensor_name]
        M, K = lhs_tensor.shape
        _K, N = rhs_tensor.shape
        assert K == _K, f"Matmul contraction dimension mismatch: {lhs_tensor.shape} and {rhs_tensor.shape}"
        if tensor_name == self.dest:
            tensor_shape = (M, N)
        else:
            raise ValueError(f"Tensor name {tensor_name} not found in {self}")
        return tensor_shape

    @property
    def read_tensor_names(self) -> list[str]:
        return [self.kwargs["stationary"], self.kwargs["moving"]]

    @property
    def write_tensor_names(self) -> list[str]:
        return [self.dest]


class Allocate(BufferNode):
    """
    Allocate operator.
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nl.ndarray", dest=dest, **kwargs)

    @property
    def read_tensor_names(self) -> list[str]:
        return []

    @property
    def write_tensor_names(self) -> list[str]:
        return [self.dest]
