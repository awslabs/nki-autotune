from compute_graph.nodes import Node
from compute_graph.tensors import Axis


class TensorScalar(Node):
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

    @property
    def read_tensor_names(self) -> list[str]:
        return [self.kwargs["data"]]

    @property
    def write_tensor_names(self) -> list[str]:
        return [self.dest]


class Matmul(Node):
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


class Allocate(Node):
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


class Load(Node):
    """
    Load operator.
    """

    def __init__(self, dest: str, src: str, axes: tuple[Axis, ...]) -> None:
        super().__init__(op_code="nl.load", dest=dest)
        self.src = src
        self.axes = axes

    @property
    def read_tensor_names(self) -> list[str]:
        return [self.src]

    @property
    def write_tensor_names(self) -> list[str]:
        return [self.dest]

    def __repr__(self) -> str:
        code = f"{self.dest} = nl.load(src={self.src}{list(self.axes)})"
        return code


class Store(Node):
    """
    Store operator.
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nl.store", dest=dest, **kwargs)

    @property
    def read_tensor_names(self) -> list[str]:
        return [self.kwargs["value"]]

    @property
    def write_tensor_names(self) -> list[str]:
        return [self.dest]
