import logging

from compute_graph.buffer_tensor import BufferAxis
from compute_graph.node.hbm_tensor import HBMAxis

logger = logging.getLogger(__name__)


class Node:
    """Base class for graph nodes."""

    def __init__(
        self,
        read_args: tuple[str, ...],
        write_args: tuple[str, ...],
        arg_to_axes: dict[str, tuple[str, ...]],
        arg_to_var: dict[str, str],
    ) -> None:
        """
        read_args: operator args to read
        write_args: operator args to write --> read and write may overlap from in-place updates
        arg_to_var: arg name to variable name. E.g. lhs (arg name) = q_tensor (var name)
        arg_to_axes: arg name to its axes semantics. May be string axis name or integer axis size for constant sizes.
        """
        self.read_args = read_args
        self.write_args = write_args
        self.arg_to_var = arg_to_var
        self.arg_to_axes = arg_to_axes
        for arg in read_args + write_args:
            assert arg in arg_to_axes, f"Tensor arg {arg} is missing axis semantics"
            assert arg in arg_to_var, f"Tensor arg {arg} is missing variable name"

        self.axes = self._populate_constant_axes()

    def _populate_constant_axes(self) -> dict[str, BufferAxis | HBMAxis]:
        axes: dict[str, BufferAxis | HBMAxis] = {}
        for arg in self.arg_to_axes:
            arg_axes = self.arg_to_axes[arg]
            for semantic in arg_axes:
                try:
                    size = int(semantic)
                    axes[semantic] = BufferAxis(name=semantic, size=size)
                except ValueError:
                    pass
        return axes

    @property
    def read_vars(self) -> tuple[str, ...]:
        return tuple([self.arg_to_var[arg] for arg in self.read_args])

    @property
    def write_vars(self) -> tuple[str, ...]:
        return tuple([self.arg_to_var[arg] for arg in self.write_args])

    @property
    def is_specialized(self) -> bool:
        args = self.read_args + self.write_args
        specialized = True
        for arg in args:
            for axis in self.arg_to_axes[arg]:
                if axis not in self.axes:
                    specialized = False
        return specialized

    def get_tensor_axes(self, arg: str) -> tuple[HBMAxis, ...] | tuple[BufferAxis, ...]:
        arg_to_axes = self.arg_to_axes[arg]
        arg_axes: tuple[HBMAxis, ...] | tuple[BufferAxis, ...] = ()
        for semantic in arg_to_axes:
            axis = self.axes[semantic]
            arg_axes += (axis,)
        return arg_axes

    def codegen(self) -> str:
        """Generate NKI code for this node."""
        raise NotImplementedError(f"codegen is not implemented for {self}")

    def _format_tensor(self, arg: str) -> str:
        """Format tensor as 'name[axes]' showing sizes if specialized."""
        var = self.arg_to_var[arg]
        return var

    def __repr__(self) -> str:
        args = []
        for arg in self.read_args:
            args.append(f"{arg}={self.arg_to_var[arg]}")
        for arg in self.write_args:
            args.append(f"{arg}={self.arg_to_var[arg]}")
        return f"{type(self).__name__}({', '.join(args)})"
