class BufferTensor:
    def __init__(self, name: str, shape: tuple[int, ...], buffer: str) -> None:
        self.name = name
        self.shape = shape
        self.buffer = buffer
        assert buffer in ["SBUF", "PSUM"], f"Illegal buffer type {buffer}"
        self.axis_names = ["unknown"] * len(shape)

    def add_axis_names(self, axis_names: list[str]) -> None:
        assert len(axis_names) == len(
            self.shape
        ), f"axis_names length {len(axis_names)} != shape length {len(self.shape)}"
        self.axis_names = axis_names

    def __repr__(self) -> str:
        axes_str = ", ".join(f"{name}:{size}" for name, size in zip(self.axis_names, self.shape))
        return f"{self.buffer}Tensor({self.name}[{axes_str}])"
