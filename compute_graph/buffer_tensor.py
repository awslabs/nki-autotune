class BufferAxis:
    """Represents a single axis of a buffer tensor with name and size."""

    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size


class BufferTensor:
    """Tensor stored in on-chip buffer (SBUF or PSUM)."""

    def __init__(self, name: str, axes: tuple[BufferAxis, ...], buffer: str) -> None:
        self.name = name
        self.axes = axes
        self.buffer = buffer
        assert buffer in ["SBUF", "PSUM"], f"Illegal buffer type {buffer}"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(ax.size for ax in self.axes)

    @property
    def axis_names(self) -> list[str]:
        return [ax.name for ax in self.axes]

    def __repr__(self) -> str:
        axes_str = ", ".join(f"{name}:{size}" for name, size in zip(self.axis_names, self.shape))
        return f"{self.buffer}Tensor({self.name}[{axes_str}])"
