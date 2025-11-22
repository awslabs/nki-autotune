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
        return all(axis in self.axis_sizes for tensor in self.inputs for axis in self.tensor_axes[tensor])

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

    def __repr__(self) -> str:
        """String representation of the node."""
        return f"{self.op_code}(inputs={self.inputs}, outputs={self.outputs})"


class TensorScalar(BufferNode):
    """
    Tensor-scalar operator: Apply up to two math operators to input data tile
    by broadcasting scalar/vector operands.

    NKI Specification:
    - Data/Dest shape: (P, F) where P=partition axis, F=free axis
    - Operand tensors (if not scalar): (P, 1) - broadcasts along partition axis
    - Operation: dest = (data op0 operand0) op1 operand1
    - Engine: Vector, Scalar, or GpSimd Engine

    Semantic Axes (hardcoded):
    - "P": Partition axis (for parallelism)
    - "F": Free axis (for data layout)
    - "1": Reduced/singleton dimension
    """

    def __init__(
        self,
        dest: str,
        data: str,
        op0,
        operand0: float | str,
        op1: str | None = None,
        operand1: float | str | None = None,
    ) -> None:
        """
        Args:
            dest: Destination tensor name
            data: Input data tensor name
            op0: First operator (e.g., np.multiply, np.add)
            operand0: First operand (scalar value or tensor name with shape (P, 1))
            op1: Second operator (optional)
            operand1: Second operand (optional, scalar value or tensor name with shape (P, 1))
        """
        inputs = [data]
        tensor_axes = {data: ["P", "F"], dest: ["P", "F"]}

        if isinstance(operand0, str):
            inputs.append(operand0)
            tensor_axes[operand0] = ["P", "1"]

        if isinstance(operand1, str):
            inputs.append(operand1)
            tensor_axes[operand1] = ["P", "1"]

        super().__init__(op_code="nisa.tensor_scalar", inputs=inputs, outputs=[dest], tensor_axes=tensor_axes)

        self.dest = dest
        self.data = data
        self.op0 = op0
        self.operand0 = operand0
        self.op1 = op1
        self.operand1 = operand1

    def __repr__(self) -> str:
        args = [f"data={self.data}[P, F]"]

        if self.op0 is not None:
            op0_name = getattr(self.op0, "__name__", str(self.op0))
            args.append(f"op0={op0_name}")

            if isinstance(self.operand0, str):
                args.append(f"operand0={self.operand0}[P, 1]")
            else:
                args.append(f"operand0={self.operand0}")

        if self.op1 is not None:
            op1_name = getattr(self.op1, "__name__", str(self.op1))
            args.append(f"op1={op1_name}")

            if isinstance(self.operand1, str):
                args.append(f"operand1={self.operand1}[P, 1]")
            else:
                args.append(f"operand1={self.operand1}")

        args_str = ", ".join(args)
        return f"{self.dest}[P, F] = nisa.tensor_scalar({args_str})"


class Activation(BufferNode):
    """
    Activation operator: Apply an activation function on every element of the input tile.

    NKI Specification:
    - Data/Dest shape: (P ≤ 128, F) where P=partition, F=free
    - reduce_res shape: (P, 1) - reduces free axis to size 1
    - Engine: Scalar Engine (float32 precision)

    With reduce:
        dest[P, F] = nisa.activation(op, data[P, F], reduce_op, reduce_res[P, 1])
    No reduce:
        dest[P, F] = nisa.activation(op, data[P, F])

    Semantic Axes (hardcoded):
    - "P": Partition axis (≤ 128)
    - "F": Free axis (reduced if reduce_op specified)
    - "1": Reduced/singleton dimension
    """

    def __init__(self, dest: str, op, data: str, reduce_op=None, reduce_res: str | None = None) -> None:
        """
        Args:
            dest: Destination tensor name (activation result)
            op: Activation operation (e.g., nl.relu, nl.tanh, np.square)
            data: Input data tensor name
            reduce_op: Optional reduction operator on free dimension (e.g., np.add, np.max)
            reduce_res: Optional reduction result tensor name with shape (P, 1)
        """
        inputs = [data]
        outputs = [dest]

        tensor_axes = {data: ["P", "F"], dest: ["P", "F"]}

        if reduce_res is not None:
            outputs.append(reduce_res)
            tensor_axes[reduce_res] = ["P", "1"]

        super().__init__(op_code="nisa.activation", inputs=inputs, outputs=outputs, tensor_axes=tensor_axes)

        self.dest = dest
        self.op = op
        self.data = data
        self.reduce_op = reduce_op
        self.reduce_res = reduce_res

    def __repr__(self) -> str:
        op_name = getattr(self.op, "__name__", str(self.op))
        args = [f"op={op_name}", f"data={self.data}[P, F]"]

        result = f"{self.dest}[P, F]"
        if self.reduce_res is not None:
            reduce_op_name = getattr(self.reduce_op, "__name__", str(self.reduce_op))
            args.append(f"reduce_op={reduce_op_name}")
            args.append(f"reduce_res={self.reduce_res}[P, 1]")
            result = f"{self.dest}[P, F], {self.reduce_res}[P, 1]"

        args_str = ", ".join(args)
        return f"{result} = nisa.activation({args_str})"


class Transpose(BufferNode):
    """
    Transpose operator: Perform a 2D transpose swapping partition and free axes.

    NKI Specification:
    - Input shape: (P, F) where P=partition, F=free
    - Output shape: (F, P) - axes swapped
    - Engine: Tensor or Vector Engine

    Constraints:
    - Tensor Engine: Input shape ≤ (128, 128)
    - Vector Engine: Input shape ≤ (32, 32)

    Semantic Axes (hardcoded):
    - Input: "P" (partition), "F" (free)
    - Output: "F" (partition), "P" (free)

    Operation: dest[F, P] = nisa.nc_transpose(data[P, F])
    """

    def __init__(self, dest: str, data: str) -> None:
        """
        Args:
            dest: Destination tensor name
            data: Source tensor name
        """
        tensor_axes = {data: ["P", "F"], dest: ["F", "P"]}  # Swapped

        super().__init__(op_code="nisa.nc_transpose", inputs=[data], outputs=[dest], tensor_axes=tensor_axes)

        self.dest = dest
        self.data = data

    def __repr__(self) -> str:
        return f"{self.dest}[F, P] = nisa.nc_transpose(data={self.data}[P, F])"


class Matmul(BufferNode):
    """
    Matrix multiplication operator: Compute stationary.T @ moving.

    NKI Specification:
    - Stationary shape: (K ≤ 128, M ≤ 128) where K=contraction, M=rows
    - Moving shape: (K ≤ 128, N ≤ 512) where K=contraction, N=cols
    - Result shape: (M, N)
    - Engine: Tensor Engine (accumulation in float32)

    Operation: dest[M, N] = stationary[K, M].T @ moving[K, N]
                          = [M, K] @ [K, N]
                          = [M, N]

    Key insight: Both inputs share the SAME partition axis K (contraction dimension).
                 Result uses the FREE axes M and N from both inputs.

    Semantic Axes (hardcoded):
    - "K": Contraction axis (partition, must match in both inputs)
    - "M": Rows axis (free axis of stationary, becomes partition of result)
    - "N": Columns axis (free axis of moving, becomes free axis of result)

    Constraints:
    - Stationary: K ≤ 128, M ≤ 128
    - Moving: K ≤ 128, N ≤ 512
    """

    def __init__(self, dest: str, stationary: str, moving: str) -> None:
        """
        Args:
            dest: Destination tensor name
            stationary: Stationary (LHS) tensor name with shape [K, M]
            moving: Moving (RHS) tensor name with shape [K, N]
        """
        tensor_axes = {stationary: ["K", "M"], moving: ["K", "N"], dest: ["M", "N"]}

        super().__init__(op_code="nisa.nc_matmul", inputs=[stationary, moving], outputs=[dest], tensor_axes=tensor_axes)

        self.dest = dest
        self.stationary = stationary
        self.moving = moving

    def __repr__(self) -> str:
        return f"{self.dest}[M, N] = nisa.nc_matmul(stationary={self.stationary}[K, M], moving={self.moving}[K, N])"


class Allocate(BufferNode):
    """
    Allocate operator: Create a new tensor in on-chip memory.

    NKI Specification (nl.ndarray):
    - Signature: ndarray(shape, dtype, *, buffer=None, name='', **kwargs)
    - shape: Tuple of integers defining tensor dimensions
    - dtype: Data type (e.g., nl.float32, nl.int32, nl.bfloat16)
    - buffer: Memory location (nl.sbuf, nl.psum, nl.shared_hbm)
    - name: Optional tensor name

    Axes are inferred from shape as dim0, dim1, dim2, etc.
    Example: shape (128, 512) → axes ["dim0", "dim1"]
    """

    def __init__(self, dest: str, shape: tuple[int, ...], dtype: str, buffer: str = "nl.sbuf") -> None:
        """
        Args:
            dest: Destination tensor name (used as 'name' parameter)
            shape: Concrete shape tuple (e.g., (128, 512))
            dtype: Data type string (e.g., "nl.float32", "nl.int32")
            buffer: Memory buffer (default: "nl.sbuf", can be "nl.psum", "nl.shared_hbm")
        """
        axes = [f"dim{i}" for i in range(len(shape))]
        tensor_axes = {dest: axes}

        super().__init__(op_code="nl.ndarray", inputs=[], outputs=[dest], tensor_axes=tensor_axes)

        self.dest = dest
        self.shape = shape
        self.dtype = dtype
        self.buffer = buffer

    def __repr__(self) -> str:
        axes_str = "[" + ", ".join(self.tensor_axes[self.dest]) + "]"
        return f"{self.dest}{axes_str} = nl.ndarray(shape={self.shape}, dtype={self.dtype}, buffer={self.buffer}, name='{self.dest}')"
