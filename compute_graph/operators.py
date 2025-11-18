from compute_graph.nodes import Node


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

    def codegen(self) -> str:
        data_name = self.kwargs["data"]
        data_tensor = self.tensors[data_name]
        data = data_tensor.name
        dest = self.tensors[self.dest].name

        shape = data_tensor.shape
        lines = []

        if len(shape) == 2:
            M, K = shape
            idx_name = f"i_{data}"
            lines.append(f"{idx_name} = nl.mgrid[0:{M}, 0:{K}]")
            data_idx = f"{data}[{idx_name}.p, {idx_name}.x]"
            dest_idx = f"{dest}[{idx_name}.p, {idx_name}.x]"
        else:
            raise ValueError(f"Unsupported tensor shape {shape} for TensorScalar")

        op0 = self.kwargs["op0"].__name__
        operand0 = self.kwargs["operand0"]
        if isinstance(operand0, str):
            operand0_tensor = self.tensors[operand0]
            operand0_name = operand0_tensor.name
            op0_shape = operand0_tensor.shape
            if len(op0_shape) == 2:
                if op0_shape[1] == 1:
                    operand0 = f"{operand0_name}[{idx_name}.p, 0]"
                else:
                    operand0 = f"{operand0_name}[{idx_name}.p, {idx_name}.x]"
            else:
                raise ValueError(f"Unsupported operand0 shape {op0_shape}")
        else:
            operand0 = repr(operand0)

        code = f"{dest_idx} = nisa.tensor_scalar({data_idx}, np.{op0}, {operand0}"

        if "op1" in self.kwargs:
            op1 = self.kwargs["op1"].__name__
            operand1 = self.kwargs["operand1"]
            if isinstance(operand1, str):
                operand1_tensor = self.tensors[operand1]
                operand1_name = operand1_tensor.name
                op1_shape = operand1_tensor.shape
                if len(op1_shape) == 2:
                    if op1_shape[1] == 1:
                        operand1 = f"{operand1_name}[{idx_name}.p, 0]"
                    else:
                        operand1 = f"{operand1_name}[{idx_name}.p, {idx_name}.x]"
                else:
                    raise ValueError(f"Unsupported operand1 shape {op1_shape}")
            else:
                operand1 = repr(operand1)
            code += f", op1=np.{op1}, operand1={operand1}"

        code += ")"
        lines.append(code)
        return "\n".join(lines)


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

    def codegen(self) -> str:
        data_name = self.kwargs["data"]
        data_tensor = self.tensors[data_name]
        data = data_tensor.name
        dest = self.tensors[self.dest].name

        lines = []
        shape = data_tensor.shape

        if len(shape) == 2:
            M, K = shape
            idx_name = f"i_{data}"
            lines.append(f"{idx_name} = nl.mgrid[0:{M}, 0:{K}]")
            data_idx = f"{data}[{idx_name}.p, {idx_name}.x]"
            dest_idx = f"{dest}[{idx_name}.p, {idx_name}.x]"
        else:
            raise ValueError(f"Unsupported tensor shape {shape} for Activation")

        op = self.kwargs["op"]
        if hasattr(op, "__name__"):
            op_name = f"nl.{op.__name__}"
        else:
            op_name = f"np.{op.__name__}"

        code = f"{dest_idx} = nisa.activation(op={op_name}, data={data_idx}"

        if "reduce_op" in self.kwargs and "reduce_res" in self.kwargs:
            reduce_op = self.kwargs["reduce_op"].__name__
            reduce_res_name = self.kwargs["reduce_res"]
            reduce_res_tensor = self.tensors[reduce_res_name]
            reduce_res = reduce_res_tensor.name
            reduce_res_shape = reduce_res_tensor.shape
            if len(reduce_res_shape) == 2 and reduce_res_shape[1] == 1:
                reduce_res_idx = f"{reduce_res}[{idx_name}.p, 0]"
            else:
                raise ValueError(f"Unsupported reduce_res shape {reduce_res_shape}")
            code += f", reduce_op=np.{reduce_op}, reduce_res={reduce_res_idx}"

        code += ")"
        lines.append(code)
        return "\n".join(lines)


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

    def codegen(self) -> str:
        data_name = self.kwargs["data"]
        data_tensor = self.tensors[data_name]
        data = data_tensor.name
        dest = self.tensors[self.dest].name

        shape = data_tensor.shape
        if len(shape) == 2:
            M, K = shape
            lines = []
            lines.append(f"i_{data} = nl.mgrid[0:{M}, 0:{K}]")
            lines.append(f"{dest}[i_{data}.p, i_{data}.x] = nisa.nc_transpose({data}[i_{data}.p, i_{data}.x])")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported tensor shape {shape} for Transpose")


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

    def codegen(self) -> str:
        stationary_name = self.kwargs["stationary"]
        stationary_tensor = self.tensors[stationary_name]
        stationary = stationary_tensor.name

        moving_name = self.kwargs["moving"]
        moving_tensor = self.tensors[moving_name]
        moving = moving_tensor.name

        dest_tensor = self.tensors[self.dest]
        dest = dest_tensor.name

        M, K = stationary_tensor.shape
        _, N = moving_tensor.shape

        lines = []
        lines.append(f"i_{stationary} = nl.mgrid[0:{M}, 0:{K}]")
        lines.append(f"i_{moving} = nl.mgrid[0:{K}, 0:{N}]")
        lines.append(f"i_{dest} = nl.mgrid[0:{M}, 0:{N}]")

        code = f"{dest}[i_{dest}.p, i_{dest}.x] = nisa.nc_matmul({stationary}[i_{stationary}.p, i_{stationary}.x], {moving}[i_{moving}.p, i_{moving}.x])"
        lines.append(code)
        return "\n".join(lines)


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

    def codegen(self) -> str:
        shape = self.kwargs["shape"]
        dest = self.tensors[self.dest].name
        return f"{dest} = nl.ndarray({shape}, dtype=nl.float32, buffer=nl.sbuf)"


class Load(Node):
    """
    Load operator.
    """

    def __init__(self, dest: str, **kwargs) -> None:
        super().__init__(op_code="nl.load", dest=dest, **kwargs)

    @property
    def read_tensor_names(self) -> list[str]:
        return [self.kwargs["src"]]

    @property
    def write_tensor_names(self) -> list[str]:
        return [self.dest]

    def codegen(self) -> str:
        src_name = self.kwargs["src"]
        src_tensor = self.tensors[src_name]
        src = src_tensor.name
        dest_tensor = self.tensors[self.dest]
        dest = dest_tensor.name

        shape = dest_tensor.shape
        if len(shape) == 2:
            M, K = shape
            lines = []
            idx_name = f"i_{dest}"
            lines.append(f"{idx_name} = nl.mgrid[0:{M}, 0:{K}]")
            lines.append(f"{dest}[{idx_name}.p, {idx_name}.x] = nl.load({src}[{idx_name}.p, {idx_name}.x])")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported tensor shape {shape} for Load")


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

    def codegen(self) -> str:
        value_name = self.kwargs["value"]
        value_tensor = self.tensors[value_name]
        value = value_tensor.name
        dest_tensor = self.tensors[self.dest]
        dest = dest_tensor.name

        shape = value_tensor.shape
        if len(shape) == 2:
            M, K = shape
            lines = []
            idx_name = f"i_{value}"
            lines.append(f"{idx_name} = nl.mgrid[0:{M}, 0:{K}]")
            lines.append(f"nl.store({dest}[{idx_name}.p, {idx_name}.x], value={value}[{idx_name}.p, {idx_name}.x])")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported tensor shape {shape} for Store")
