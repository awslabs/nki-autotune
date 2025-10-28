class Operator:
    """Defines a compute operation in the workload."""

    def __init__(self, op: str, src: list[str], dest: str, params: dict) -> None:
        self.op = op
        self.src = src
        self.dest = dest
        self.params = params

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        """Forward pass of the operator."""
        raise NotImplementedError(f"Forward pass not implemented for operator {self.op}")

    def __repr__(self) -> str:
        params_str = f", params={self.params}" if self.params else ""
        return f"Operator('{self.op}': {self.src} -> {self.dest}{params_str})"


class RMSNorm(Operator):
    def __init__(self, src: str, dest: str) -> None:
        super().__init__(op="rms_norm", src=[src], dest=dest, params={})

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        src_shape = src_shapes[self.src[0]]
        output_shape = src_shape
        # print(f"RMSNorm forward pass: {src_shape} -> {output_shape}")
        return output_shape


class Matmul(Operator):
    def __init__(self, src: list[str], dest: str) -> None:
        super().__init__(op="matmul", src=src, dest=dest, params={})

    def forward(self, src_shapes: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
        src1_shape = src_shapes[self.src[0]]
        src2_shape = src_shapes[self.src[1]]
        assert src1_shape[1] == src2_shape[0], f"Matmul shape mismatch: {src1_shape} x {src2_shape}"
        output_shape = (src1_shape[0], src2_shape[1])
        # print(f"Matmul forward pass: {src1_shape} x {src2_shape} -> {output_shape}")
        return output_shape
