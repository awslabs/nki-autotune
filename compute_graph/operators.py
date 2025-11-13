from compute_graph.nodes import Node


class TensorScalar(Node):
    """
    Tensor-scalar operator.
    dest[...] = nisa.tensor_scalar(data, op0, operand0, op1, operand1)
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nisa.tensor_scalar", dest=dest, **kwargs)

    def get_tensor_names(self) -> list[str]:
        tensor_names = [self.dest, self.kwargs["data"]]
        return tensor_names


class Activation(Node):
    """
    Activation operator with optional reduce.
    With reduce:
        activate_res[...] = nisa.activation(op, data[...], reduce_op, reduce_res[...])
    No reduce:
        activate_res[...] = nisa.activation(op, data[...])
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nisa.activation", dest=dest, **kwargs)

    def get_tensor_names(self) -> list[str]:
        tensor_names = [self.dest, self.kwargs["data"]]
        if "reduce_res" in self.kwargs:
            tensor_names.append(self.kwargs["reduce_res"])
        return tensor_names


class Transpose(Node):
    """Transpose operator."""

    def __init__(self, dest: str, **kwargs) -> None:
        """
        Args:
            data: Source tensor name
            dest: Destination tensor name
            transpose_axes: List of axes to transpose
        """
        super().__init__(op_code="nisa.nc_transpose", dest=dest, **kwargs)

    def get_tensor_names(self) -> list[str]:
        tensor_names = [self.dest, self.kwargs["data"]]
        return tensor_names


class Matmul(Node):
    """Matrix multiplication operator."""

    def __init__(self, dest: str, **kwargs) -> None:
        """
        Args:
            src: List of two source tensor names [A, B]
            dest: Destination tensor name
        """
        super().__init__(op_code="nisa.nc_matmul", dest=dest, **kwargs)

    def get_tensor_names(self) -> list[str]:
        tensor_names = [self.dest, self.kwargs["stationary"], self.kwargs["moving"]]
        return tensor_names


class Allocate(Node):
    """
    Allocate operator.
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nl.ndarray", dest=dest, **kwargs)

    def get_tensor_names(self) -> list[str]:
        return [self.dest]


class Load(Node):
    """
    Load operator.
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nl.load", dest=dest, **kwargs)

    def get_tensor_names(self) -> list[str]:
        return [self.dest, self.kwargs["src"]]
