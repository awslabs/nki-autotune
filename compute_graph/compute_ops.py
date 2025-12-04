from typing import Any

from compute_graph.buffer_tensor import BufferAxis, BufferTensor


class ComputeOp:
    """Base class for on-chip operators."""

    def __init__(
        self,
        input_args: list[str],
        output_args: list[str],
        arg_to_axes: dict[str, list[str]],
        arg_to_var: dict[str, str],
    ) -> None:
        self.input_args = input_args
        self.output_args = output_args
        self.arg_to_axes = arg_to_axes
        self.arg_to_var = arg_to_var
        for arg in input_args + output_args:
            assert arg in arg_to_axes, f"Tensor arg {arg} is missing axes"
            assert arg in arg_to_var, f"Tensor arg {arg} is missing variable name"
        self.symbolic_axes: dict[str, BufferAxis] = {}
        self._populate_constant_axes()

    def _populate_constant_axes(self) -> None:
        """Populate symbolic_axes for constant axes (numeric axis names like "1" or "128")."""
        for arg in self.arg_to_axes:
            axes = self.arg_to_axes[arg]
            for axis in axes:
                try:
                    size = int(axis)
                    self.symbolic_axes[axis] = BufferAxis(name=axis, size=size)
                except ValueError:
                    pass

    @property
    def input_names(self) -> list[str]:
        return [self.arg_to_var[arg] for arg in self.input_args]

    @property
    def is_specialized(self) -> bool:
        args = self.input_args + self.output_args
        return all(axis in self.symbolic_axes for arg in args for axis in self.arg_to_axes[arg])

    def specialize(self, arg: str, tensor: BufferTensor) -> None:
        """Map operator symbolic axes to tensor's BufferAxis objects.

        Args:
            arg: The operator argument name (e.g., "lhs", "rhs", "data")
            tensor: The BufferTensor providing concrete axis info
        """
        expected_axes = self.arg_to_axes[arg]
        if len(tensor.axes) != len(expected_axes):
            raise ValueError(
                f"Shape mismatch for '{arg}': expected {len(expected_axes)} dimensions {expected_axes} "
                f"but got {len(tensor.axes)} dimensions {tensor.shape}"
            )
        for symbolic, buffer_axis in zip(expected_axes, tensor.axes):
            if symbolic in self.symbolic_axes:
                if self.symbolic_axes[symbolic].size != buffer_axis.size:
                    raise ValueError(
                        f"Axis size conflict {arg}.{symbolic}: "
                        f"expected {self.symbolic_axes[symbolic].size}, got {buffer_axis.size} in {self}."
                    )
            else:
                self.symbolic_axes[symbolic] = buffer_axis

    def get_tensor_shape(self, arg: str) -> tuple[int, ...]:
        """Get the shape of a tensor by looking up axis sizes."""
        axes = self.arg_to_axes[arg]
        shape = []
        for axis in axes:
            if axis not in self.symbolic_axes:
                raise ValueError(f"Axis '{axis}' for tensor '{arg}' is not specialized yet in {self}. ")
            shape.append(self.symbolic_axes[axis].size)
        return tuple(shape)

    def get_output_axes(self, arg: str) -> tuple[BufferAxis, ...]:
        """Get the output BufferAxis tuple for an output tensor."""
        symbolic_axes = self.arg_to_axes[arg]
        return tuple(self.symbolic_axes[axis] for axis in symbolic_axes)

    def codegen(self) -> str:
        """Generate NKI code for this node."""
        raise NotImplementedError(f"codegen is not implemented for {self}")

    def clear_specialization(self) -> None:
        self.symbolic_axes.clear()
        self._populate_constant_axes()

    def _format_tensor(self, arg: str) -> str:
        """Format tensor as 'name[axes]' showing sizes if specialized."""
        axes = self.arg_to_axes[arg]
        axis_strs = []
        for axis in axes:
            if axis in self.symbolic_axes:
                axis_strs.append(f"{self.symbolic_axes[axis].size}")
            else:
                axis_strs.append(axis)
        tensor_name = self.arg_to_var[arg]
        axis_str = ", ".join(axis_strs)
        result = f"{tensor_name}[{axis_str}]"
        return result

    def __repr__(self) -> str:
        """String representation of the node."""
        raise NotImplementedError(f"repr is not implemented for the base ComputeOp class.")


class TensorScalar(ComputeOp):
    """Element-wise operations on data tiles with scalar/vector operands.

    Supports chaining up to two operations with broadcasting along partition axis.
    Data and destination have shape (P, F); operands are scalars or (P, 1) vectors.
    """

    def __init__(
        self,
        dest: str,
        data: str,
        op0: Any,
        operand0: float | str,
        op1: Any = None,
        operand1: float | str | None = None,
    ) -> None:
        input_args = ["data"]
        output_args = ["dest"]
        arg_to_axes = {"data": ["P", "F"], "dest": ["P", "F"]}
        arg_to_var = {"data": data, "dest": dest}

        if isinstance(operand0, str):
            input_args.append("operand0")
            arg_to_axes["operand0"] = ["P", "1"]
            arg_to_var["operand0"] = operand0

        if isinstance(operand1, str):
            input_args.append("operand1")
            arg_to_axes["operand1"] = ["P", "1"]
            arg_to_var["operand1"] = operand1

        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

        self.op0 = op0
        self.operand0 = operand0
        self.op1 = op1
        self.operand1 = operand1

    def codegen(self) -> str:
        """Generate NKI code for tensor_scalar operation."""
        dest = self.arg_to_var["dest"]
        data = self.arg_to_var["data"]
        operand0 = self.arg_to_var.get("operand0", self.operand0)

        args = [f"data={data}", f"op0={self.op0}", f"operand0={operand0}"]

        if self.op1 is not None:
            operand1 = self.arg_to_var.get("operand1", self.operand1)
            args.append(f"op1={self.op1}")
            args.append(f"operand1={operand1}")

        args_str = ", ".join(args)
        return f"nisa.tensor_scalar({dest}, {args_str})"

    def __repr__(self) -> str:
        args = [f"data={self._format_tensor('data')}"]
        args.append(f"op0={self.op0}")

        if isinstance(self.operand0, str):
            args.append(f"operand0={self._format_tensor('operand0')}")
        else:
            args.append(f"operand0={self.operand0}")

        if self.op1 is not None:
            args.append(f"op1={self.op1}")

            if isinstance(self.operand1, str):
                args.append(f"operand1={self._format_tensor('operand1')}")
            else:
                args.append(f"operand1={self.operand1}")

        args_str = ", ".join(args)
        return f"{self._format_tensor('dest')} = TensorScalar({args_str})"


class Activation(ComputeOp):
    """Apply activation functions element-wise to input tiles.

    Optionally reduces along the free axis to shape (P, 1).
    Input and output shapes: (P, F) where P is partition, F is free axis.
    """

    def __init__(self, dest: str, op: Any, data: str, reduce_op: Any = None, reduce_res: str | None = None) -> None:
        input_args = ["data"]
        output_args = ["dest"]
        arg_to_axes = {"data": ["P", "F"], "dest": ["P", "F"]}
        arg_to_var = {"dest": dest, "data": data}
        if reduce_res:
            output_args.append("reduce_res")
            arg_to_axes["reduce_res"] = ["P", "1"]
            arg_to_var["reduce_res"] = reduce_res
        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

        self.op = op
        self.reduce_op = reduce_op

    def codegen(self) -> str:
        """Generate NKI code for activation operation."""
        dest = self.arg_to_var["dest"]
        data = self.arg_to_var["data"]

        args = [f"op={self.op}", f"data={data}"]

        if self.reduce_op is not None and "reduce_res" in self.arg_to_var:
            reduce_res = self.arg_to_var["reduce_res"]
            args.append(f"reduce_op={self.reduce_op}")
            args.append(f"reduce_res={reduce_res}")

        args_str = ", ".join(args)
        return f"nisa.activation({dest}, {args_str})"

    def __repr__(self) -> str:
        data_str = self._format_tensor("data")
        args = [f"op={self.op}", f"data={data_str}"]
        result = self._format_tensor("dest")
        if "reduce_res" in self.arg_to_var and self.arg_to_var["reduce_res"]:
            reduce_res_str = self._format_tensor("reduce_res")
            args.append(f"reduce_op={self.reduce_op}")
            args.append(f"reduce_res={reduce_res_str}")
        args_str = ", ".join(args)
        return f"{result} = Activation({args_str})"


class Transpose(ComputeOp):
    """2D transpose swapping partition and free axes.

    Transforms input (P, F) to output (F, P).
    """

    def __init__(self, dest: str, data: str) -> None:
        input_args = ["data"]
        output_args = ["dest"]
        arg_to_axes = {"data": ["P", "F"], "dest": ["F", "P"]}
        arg_to_var = {"data": data, "dest": dest}

        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

    def codegen(self) -> str:
        """Generate NKI code for nc_transpose operation."""
        dest = self.arg_to_var["dest"]
        data = self.arg_to_var["data"]
        return f"nisa.nc_transpose({dest}, {data})"

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = nisa.nc_transpose(data={self._format_tensor('data')})"


class TileTranspose(ComputeOp):
    """In-tile transpose maintaining (P, F) shape.

    Rearranges element layout within the tile without changing axes,
    unlike nc_transpose which swaps partition and free dimensions.
    """

    def __init__(self, dest: str, data: str) -> None:
        input_args = ["data"]
        output_args = ["dest"]
        arg_to_axes = {"data": ["P", "F"], "dest": ["P", "F"]}
        arg_to_var = {"data": data, "dest": dest}

        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

    def codegen(self) -> str:
        """Generate NKI code for in-tile transpose using nc_transpose."""
        dest = self.arg_to_var["dest"]
        data = self.arg_to_var["data"]
        return f"nisa.nc_transpose({dest}, {data})"

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = TileTranspose(data={self._format_tensor('data')})"


class Matmul(ComputeOp):
    """Matrix multiplication: lhs @ rhs with optional lhs transpose.

    Computes (M, K) @ (K, N) → (M, N), or (K, M).T @ (K, N) → (M, N).
    M, K, N represent rows, contraction, and columns axes respectively.
    """

    def __init__(self, dest: str, lhs: str, rhs: str, lhs_transposed: bool) -> None:
        input_args = ["lhs", "rhs"]
        output_args = ["dest"]

        if lhs_transposed:
            arg_to_axes = {"lhs": ["K", "M"], "rhs": ["K", "N"], "dest": ["M", "N"]}
        else:
            arg_to_axes = {"lhs": ["M", "K"], "rhs": ["K", "N"], "dest": ["M", "N"]}

        arg_to_var = {"lhs": lhs, "rhs": rhs, "dest": dest}

        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

        self.lhs_transposed = lhs_transposed

    def codegen(self) -> str:
        """Generate NKI code for nc_matmul operation.

        nc_matmul computes: dst = stationary.T @ moving
        For lhs @ rhs where lhs is (M, K) and rhs is (K, N):
        - stationary = lhs (will be transposed internally)
        - moving = rhs
        """
        dest = self.arg_to_var["dest"]
        lhs = self.arg_to_var["lhs"]
        rhs = self.arg_to_var["rhs"]
        return f"{dest} = nisa.nc_matmul({lhs}, {rhs})"

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = Matmul(lhs={self._format_tensor('lhs')}, rhs={self._format_tensor('rhs')})"
