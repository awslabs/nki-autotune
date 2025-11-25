import neuronxcc.nki.language as nl


class BufferOp:
    """Base class for on-chip operators."""

    def __init__(
        self,
        input_args: list[str],
        output_args: list[str],
        arg_to_axes: dict[str, list[str]],
        arg_to_var: dict[str, str],
    ) -> None:
        """
        Args:
            input_args: Input tensor args
            output_args: Output tensor args
            arg_to_axes: Maps tensor arg to their axes
            arg_to_var: Maps arg to variable name
        """
        self.input_args = input_args
        self.output_args = output_args
        self.arg_to_axes = arg_to_axes
        self.arg_to_var = arg_to_var
        for arg in input_args + output_args:
            assert arg in arg_to_axes, f"Tensor arg {arg} is missing axes"
            assert arg in arg_to_var, f"Tensor arg {arg} is missing variable name"
        self.axis_sizes: dict[str, int] = {}
        self._populate_constant_axes()

    def _populate_constant_axes(self) -> None:
        """Populate axis_sizes for constant axes (numeric axis names like "1" or "128")."""
        for arg in self.arg_to_axes:
            axes = self.arg_to_axes[arg]
            for axis in axes:
                try:
                    size = int(axis)
                    self.axis_sizes[axis] = size
                except ValueError:
                    pass

    @property
    def input_names(self) -> list[str]:
        return [self.arg_to_var[arg] for arg in self.input_args]

    @property
    def is_specialized(self) -> bool:
        args = self.input_args + self.output_args
        return all(axis in self.axis_sizes for arg in args for axis in self.arg_to_axes[arg])

    def specialize(self, arg: str, shape: tuple[int, ...]) -> None:
        expected_axes = self.arg_to_axes[arg]
        if len(shape) != len(expected_axes):
            raise ValueError(
                f"Shape mismatch for '{arg}': expected {len(expected_axes)} dimensions {expected_axes} "
                f"but got {len(shape)} dimensions {shape}"
            )
        for axis, size in zip(expected_axes, shape):
            if axis in self.axis_sizes and self.axis_sizes[axis] != size:
                raise ValueError(
                    f"Axis size conflict {arg}.{axis}: "
                    f"already set to {self.axis_sizes[axis]}, trying to set to {size} in {self}."
                )
            self.axis_sizes[axis] = size

    def get_tensor_shape(self, arg: str) -> tuple[int, ...]:
        """Get the shape of a tensor by looking up axis sizes.

        Args:
            arg: Name of the tensor to get shape for

        Returns:
            Tuple of axis sizes representing the tensor shape

        Raises:
            ValueError: If arg is not in tensor_axes or if any required axis is not specialized
        """
        axes = self.arg_to_axes[arg]
        shape = []
        for axis in axes:
            if axis not in self.axis_sizes:
                raise ValueError(f"Axis '{axis}' for tensor '{arg}' is not specialized yet in {self}. ")
            shape.append(self.axis_sizes[axis])
        return tuple(shape)

    def codegen(self) -> str:
        """Generate NKI code for this node."""
        raise NotImplementedError(f"codegen is not implemented for {self}")

    def clear_specialization(self) -> None:
        self.axis_sizes.clear()

    def _format_tensor(self, arg: str) -> str:
        """Format tensor as 'name[axes]' showing sizes if specialized."""
        axes = self.arg_to_axes[arg]
        axis_strs = []
        for axis in axes:
            if axis in self.axis_sizes:
                axis_strs.append(f"{self.axis_sizes[axis]}")
            else:
                axis_strs.append(axis)
        tensor_name = self.arg_to_var[arg]
        axis_str = ", ".join(axis_strs)
        result = f"{tensor_name}[{axis_str}]"
        return result

    def __repr__(self) -> str:
        """String representation of the node."""
        raise NotImplementedError(f"repr is not implemented for the base BufferOp class.")


class TensorScalar(BufferOp):
    """Element-wise operations on data tiles with scalar/vector operands.

    Supports chaining up to two operations with broadcasting along partition axis.
    Data and destination have shape (P, F); operands are scalars or (P, 1) vectors.
    """

    def __init__(
        self, dest: str, data: str, op0, operand0: float | str, op1=None, operand1: float | str | None = None
    ) -> None:
        """
        Args:
            dest: Destination tensor name
            data: Input tensor name
            op0: First operator
            operand0: First operand (scalar or tensor name)
            op1: Second operator
            operand1: Second operand (scalar or tensor name)
        """
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

    def __repr__(self) -> str:
        args = [f"data={self._format_tensor('data')}"]

        op0_name = getattr(self.op0, "__name__", str(self.op0))
        args.append(f"op0={op0_name}")

        if isinstance(self.operand0, str):
            args.append(f"operand0={self._format_tensor('operand0')}")
        else:
            args.append(f"operand0={self.operand0}")

        if self.op1 is not None:
            op1_name = getattr(self.op1, "__name__", str(self.op1))
            args.append(f"op1={op1_name}")

            if isinstance(self.operand1, str):
                args.append(f"operand1={self._format_tensor('operand1')}")
            else:
                args.append(f"operand1={self.operand1}")

        args_str = ", ".join(args)
        return f"{self._format_tensor('dest')} = TensorScalar({args_str})"


class Activation(BufferOp):
    """Apply activation functions element-wise to input tiles.

    Optionally reduces along the free axis to shape (P, 1).
    Input and output shapes: (P, F) where P is partition, F is free axis.
    """

    def __init__(self, dest: str, op, data: str, reduce_op=None, reduce_res: str | None = None) -> None:
        """
        Args:
            dest: Destination tensor name
            op: Activation operation
            data: Input tensor name
            reduce_op: Reduction operator for free dimension
            reduce_res: Reduction result tensor name
        """
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

    def __repr__(self) -> str:
        op_name = getattr(self.op, "__name__", str(self.op))
        data_str = self._format_tensor("data")
        args = [f"op={op_name}", f"data={data_str}"]
        result = self._format_tensor("dest")
        if "reduce_res" in self.arg_to_var and self.arg_to_var["reduce_res"]:
            reduce_op_name = getattr(self.reduce_op, "__name__", str(self.reduce_op))
            reduce_res_str = self._format_tensor("reduce_res")
            args.append(f"reduce_op={reduce_op_name}")
            args.append(f"reduce_res={reduce_res_str}")
        args_str = ", ".join(args)
        return f"{result} = Activation({args_str})"


class Transpose(BufferOp):
    """2D transpose swapping partition and free axes.

    Transforms input (P, F) to output (F, P).
    """

    def __init__(self, dest: str, data: str) -> None:
        """
        Args:
            dest: Destination tensor name
            data: Source tensor name
        """
        input_args = ["data"]
        output_args = ["dest"]
        arg_to_axes = {"data": ["P", "F"], "dest": ["F", "P"]}
        arg_to_var = {"data": data, "dest": dest}

        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = nisa.nc_transpose(data={self._format_tensor('data')})"


class TileTranspose(BufferOp):
    """In-tile transpose maintaining (P, F) shape.

    Rearranges element layout within the tile without changing axes,
    unlike nc_transpose which swaps partition and free dimensions.
    """

    def __init__(self, dest: str, data: str) -> None:
        """
        Args:
            dest: Destination tensor name
            data: Source tensor name
        """
        input_args = ["data"]
        output_args = ["dest"]
        arg_to_axes = {"data": ["P", "F"], "dest": ["P", "F"]}
        arg_to_var = {"data": data, "dest": dest}

        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = TileTranspose(data={self._format_tensor('data')})"


class Matmul(BufferOp):
    """Matrix multiplication: lhs @ rhs with optional lhs transpose.

    Computes (M, K) @ (K, N) → (M, N), or (K, M).T @ (K, N) → (M, N).
    M, K, N represent rows, contraction, and columns axes respectively.
    """

    def __init__(self, dest: str, lhs: str, rhs: str, lhs_transposed: bool) -> None:
        """
        Args:
            dest: Destination tensor name
            lhs: Left operand (M, K) or (K, M) if transposed
            rhs: Right operand (K, N)
            lhs_transposed: Whether lhs is transposed
        """
        input_args = ["lhs", "rhs"]
        output_args = ["dest"]

        if lhs_transposed:
            arg_to_axes = {"lhs": ["K", "M"], "rhs": ["K", "N"], "dest": ["M", "N"]}
        else:
            arg_to_axes = {"lhs": ["M", "K"], "rhs": ["K", "N"], "dest": ["M", "N"]}

        arg_to_var = {"lhs": lhs, "rhs": rhs, "dest": dest}

        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

        self.lhs_transposed = lhs_transposed

    def __repr__(self) -> str:
        return f"{self._format_tensor('dest')} = Matmul(lhs={self._format_tensor('lhs')}, rhs={self._format_tensor('rhs')})"


class Allocate(BufferOp):
    """Allocate a tensor in on-chip memory.

    Creates nl.ndarray with specified shape, and buffer location.
    Axes are auto-generated as dim0, dim1, etc. based on shape.
    """

    def __init__(self, dest: str, shape: tuple[int, ...]) -> None:
        """
        Args:
            dest: Tensor name
            shape: Shape tuple
            buffer: Memory buffer location
        """
        axes = [f"dim{i}" for i in range(len(shape))]

        input_args = []
        output_args = ["dest"]
        arg_to_axes = {"dest": axes}
        arg_to_var = {"dest": dest}

        super().__init__(input_args=input_args, output_args=output_args, arg_to_axes=arg_to_axes, arg_to_var=arg_to_var)

        for i in range(len(shape)):
            axis = axes[i]
            size = shape[i]
            self.axis_sizes[axis] = size

        self.shape = shape
        self.buffer = nl.sbuf

    def __repr__(self) -> str:
        return f"{self.arg_to_var['dest']} = Allocate(shape={self.shape}, buffer={self.buffer.name})"
