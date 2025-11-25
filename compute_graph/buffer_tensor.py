class BufferTensor:
    def __init__(self, name: str, shape: tuple[int, ...], buffer: str) -> None:
        self.name = name
        self.shape = shape
        self.buffer = buffer
        assert buffer in ["SBUF", "PSUM"], f"Illegal buffer type {buffer}"

    def __repr__(self) -> str:
        return f"{self.buffer}Tensor({self.name}, shape={self.shape})"
