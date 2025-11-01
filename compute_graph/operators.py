class Operator:
    """Base class for compute operations in the workload."""

    def __init__(self, op: str, src: list[str], dest: str, **kwargs) -> None:
        """
        Args:
            op: Operation name
            src: List of source tensor names
            dest: Destination tensor name
            kwargs: Operation-specific parameters
        """
        self.op = op
        self.src = src
        self.dest = dest
        self.kwargs = kwargs

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """
        Compute output shape from input shapes.

        Args:
            src_shapes: Dictionary mapping tensor names to their shapes

        Returns:
            Output tensor shape
        """
        raise NotImplementedError(f"Forward pass not implemented for operator {self.op}")

    def __repr__(self) -> str:
        kwargs_str = f", kwargs={self.kwargs}" if self.kwargs else ""
        return f"Operator({self.op}({self.src}) -> {self.dest}{kwargs_str})"


class TensorScalar(Operator):
    """
    Tensor-scalar operator.
    op1(op(src, scalar), scalar1)
    """

    def __init__(self, op: str, src: str, dest: str, scalar: float, op1: str, scalar1: float) -> None:
        """

        Args:
            op: Operation name
            src: Source tensor name
            scalar: Scalar value
            dest: Destination tensor name
        """
        super().__init__(op=op, src=[src], dest=dest, scalar=scalar, op1=op1, scalar1=scalar1)

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """
        Args:
            src_shapes: Dictionary mapping tensor names to their shapes

        Returns:
            Output tensor shape
        """
        src_shape = src_shapes[self.src[0]]
        return src_shape


class ActivationReduce(Operator):
    def __init__(self, op: str, src: str, dest: str, reduce_op: str, reduction_axis: int) -> None:
        super().__init__(op=op, src=[src], dest=dest, reduce_op=reduce_op, reduction_axis=reduction_axis)

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        src_shape = src_shapes[self.src[0]]
        output_shape_list = list(src_shape)
        output_shape_list[self.kwargs["reduction_axis"]] = 1
        output_shape = tuple(output_shape_list)
        return output_shape


class Activation(Operator):
    """Activation operator."""

    def __init__(self, op: str, src: str, dest: str) -> None:
        """
        Args:
            op: Activation function name
            src: Source tensor name
            dest: Destination tensor name
        """
        super().__init__(op=op, src=[src], dest=dest)

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """
        Args:
            src_shapes: Dictionary mapping tensor names to their shapes

        Returns:
            Output tensor shape
        """
        src_shape = src_shapes[self.src[0]]
        return src_shape


class Transpose(Operator):
    """Transpose operator."""

    def __init__(self, src: str, dest: str, transpose_axes: tuple[int, int]) -> None:
        """
        Args:
            src: Source tensor name
            dest: Destination tensor name
            transpose_axes: List of axes to transpose
        """
        super().__init__(op="transpose", src=[src], dest=dest, transpose_axes=transpose_axes)

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """
        Args:
            src_shapes: Dictionary mapping tensor names to their shapes

        Returns:
            Output shape with transposed dimensions
        """
        src = self.src[0]
        src_shape = src_shapes[src]
        transpose_axes = self.kwargs["transpose_axes"]
        output_shape_list = list(src_shape)
        axis0, axis1 = transpose_axes
        output_shape_list[axis0], output_shape_list[axis1] = output_shape_list[axis1], output_shape_list[axis0]
        output_shape = tuple(output_shape_list)
        return output_shape


class Matmul(Operator):
    """Matrix multiplication operator."""

    def __init__(self, src: list[str], dest: str, **kwargs) -> None:
        """
        Args:
            src: List of two source tensor names [A, B]
            dest: Destination tensor name
        """
        super().__init__(op="matmul", src=src, dest=dest, **kwargs)

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """
        Args:
            src_shapes: Dictionary mapping tensor names to their shapes

        Returns:
            Output shape (M, N) from inputs (K, M) and (K, N)
        """
        src1_shape = src_shapes[self.src[0]]
        src2_shape = src_shapes[self.src[1]]
        assert src1_shape[0] == src2_shape[0], f"Matmul shape mismatch: {src1_shape} x {src2_shape}"
        output_shape = (src1_shape[1], src2_shape[1])
        return output_shape
