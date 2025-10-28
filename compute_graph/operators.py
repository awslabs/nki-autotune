class Operator:
    """Base class for compute operations in the workload."""

    def __init__(self, op: str, src: list[str], dest: str, params: dict) -> None:
        """
        Args:
            op: Operation name
            src: List of source tensor names
            dest: Destination tensor name
            params: Operation-specific parameters
        """
        self.op = op
        self.src = src
        self.dest = dest
        self.params = params

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
        params_str = f", params={self.params}" if self.params else ""
        return f"Operator('{self.op}': {self.src} -> {self.dest}{params_str})"


class RMSNorm(Operator):
    """RMS normalization operator."""

    def __init__(self, src: str, dest: str) -> None:
        """
        Args:
            src: Source tensor name
            dest: Destination tensor name
        """
        super().__init__(op="rms_norm", src=[src], dest=dest, params={})

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """
        Args:
            src_shapes: Dictionary mapping tensor names to their shapes

        Returns:
            Output shape (same as input shape)
        """
        src_shape = src_shapes[self.src[0]]
        output_shape = src_shape
        return output_shape


class Matmul(Operator):
    """Matrix multiplication operator."""

    def __init__(self, src: list[str], dest: str) -> None:
        """
        Args:
            src: List of two source tensor names [A, B]
            dest: Destination tensor name
        """
        super().__init__(op="matmul", src=src, dest=dest, params={})

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """
        Args:
            src_shapes: Dictionary mapping tensor names to their shapes

        Returns:
            Output shape (M, N) from inputs (M, K) and (K, N)
        """
        src1_shape = src_shapes[self.src[0]]
        src2_shape = src_shapes[self.src[1]]
        assert src1_shape[1] == src2_shape[0], f"Matmul shape mismatch: {src1_shape} x {src2_shape}"
        output_shape = (src1_shape[0], src2_shape[1])
        return output_shape
