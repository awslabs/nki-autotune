from dataclasses import dataclass


@dataclass
class Axis:
    name: str
    tile_size: int
    num_tiles: int

    @property
    def size(self) -> int:
        return self.tile_size * self.num_tiles

    def __repr__(self) -> str:
        return f"{self.name}:{self.num_tiles}*{self.tile_size}={self.size}"


@dataclass
class Tensor:
    name: str
    location: str  # sbuf | psum | hbm
    axes: tuple[Axis, ...]

    def __post_init__(self) -> None:
        assert self.location in ("SBUF", "PSUM", "HBM"), f"location must be sbuf, psum, or hbm, got {self.location}"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(ax.size for ax in self.axes)

    @property
    def num_axes(self) -> int:
        return len(self.axes)

    def __repr__(self) -> str:
        return f"{self.location}Tensor({self.name}{self.axes})"


@dataclass
class TileRange:
    start_tile: int
    end_tile: int


def create_tensor(name: str, shape: tuple[int, ...], location: str) -> Tensor:
    axes: list[Axis] = []
    for counter, size in enumerate(shape):
        axis = Axis(name=f"{name}_axis_{counter}", tile_size=size, num_tiles=1)
        axes.append(axis)
    tensor = Tensor(name=name, location=location, axes=tuple(axes))
    return tensor
