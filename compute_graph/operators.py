import logging

from compute_graph.buffer_tensor import BufferAxis, BufferTensor
from compute_graph.hbm_tensor import HBMAxis, HBMTensor

logger = logging.getLogger(__name__)


class Operator:
    """Base class for operators."""

    def __init__(
        self,
        read_args: tuple[str, ...],
        write_args: tuple[str, ...],
        arg_to_var: dict[str, str],
        axis_semantics: dict[str, tuple[str, ...]],
    ) -> None:
        """
        read_args: operator args to read
        write_args: operator args to write --> read and write may overlap from in-place updates
        arg_to_var: arg name to variable name. E.g. lhs (arg name) = q_tensor (var name)
        axis_semantics: arg name to its axes semantics. May be string axis name or integer axis size for constant sizes.
        """
        self.read_args = read_args
        self.write_args = write_args
        self.arg_to_var = arg_to_var
        self.axis_semantics = axis_semantics
        for arg in read_args + write_args:
            assert arg in axis_semantics, f"Tensor arg {arg} is missing axis semantics"
            assert arg in arg_to_var, f"Tensor arg {arg} is missing variable name"

        self.axis_specialization = self._populate_constant_axes()
        self.arg_to_tensor: dict[str, HBMTensor | BufferTensor] = {}

    def _populate_constant_axes(self) -> dict[str, BufferAxis | HBMAxis]:
        axis_specialization: dict[str, BufferAxis | HBMAxis] = {}
        for arg in self.axis_semantics:
            arg_axes = self.axis_semantics[arg]
            for semantic in arg_axes:
                try:
                    size = int(semantic)
                    axis_specialization[semantic] = BufferAxis(name=semantic, size=size)
                except ValueError:
                    pass
        return axis_specialization

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
            for axis in self.axis_semantics[arg]:
                if axis not in self.axis_specialization:
                    specialized = False
        return specialized

    def specialize(self, arg: str, tensor: BufferTensor | HBMTensor) -> None:
        """Map operator symbolic axes to tensor's BufferAxis objects.

        Args:
            arg: The operator argument name (e.g., "lhs", "rhs", "data")
            tensor: The BufferTensor providing concrete axis info
        """
        axis_semantics = self.axis_semantics[arg]
        if len(tensor.axes) != len(axis_semantics):
            raise ValueError(
                f"Shape mismatch for '{arg}': expected {len(axis_semantics)} dimensions {axis_semantics} "
                f"but got {len(tensor.axes)} dimensions for shape {tensor.shape}"
            )
        for semantic, tensor_axis in zip(axis_semantics, tensor.axes):
            if semantic in self.axis_specialization:
                expected_size = self.axis_specialization[semantic].size
                if expected_size != tensor_axis.size:
                    raise ValueError(
                        f"Axis size conflict {arg}.{semantic}: "
                        f"expected {expected_size}, got {tensor_axis.size} in {self}."
                    )
            else:
                self.axis_specialization[semantic] = tensor_axis
        self.arg_to_tensor[arg] = tensor

    def get_tensor_axes(self, arg: str) -> tuple[HBMAxis, ...] | tuple[BufferAxis, ...]:
        """FIXME: pylance type warning"""
        axis_semantics = self.axis_semantics[arg]
        arg_axes: tuple[HBMAxis, ...] | tuple[BufferAxis, ...] = ()
        for semantic in axis_semantics:
            axis = self.axis_specialization[semantic]
            arg_axes += (axis,)
        return arg_axes

    def get_tensor_shape(self, arg: str) -> tuple[int, ...]:
        """Get the shape of a tensor by looking up axis sizes."""
        tensor = self.arg_to_tensor[arg]
        return tensor.shape

    def codegen(self) -> str:
        """Generate NKI code for this node."""
        raise NotImplementedError(f"codegen is not implemented for {self}")

    def clear_specialization(self) -> None:
        self.axis_specialization = self._populate_constant_axes()
        self.arg_to_tensor: dict[str, HBMTensor | BufferTensor] = {}

    def _format_tensor(self, arg: str) -> str:
        """Format tensor as 'name[axes]' showing sizes if specialized."""
        if arg in self.arg_to_tensor:
            tensor = self.arg_to_tensor[arg]
            result = f"{tensor}"
        else:
            var = self.arg_to_var[arg]
            axes = self.axis_semantics[arg]
            axis_strs = []
            for axis in axes:
                if axis in self.axis_specialization:
                    axis_strs.append(f"{self.axis_specialization[axis].size}")
                else:
                    axis_strs.append(axis)
            axis_str = ", ".join(axis_strs)
            result = f"{var}[{axis_str}]"
        return result

    def __repr__(self) -> str:
        """String representation of the node."""
        raise NotImplementedError(f"repr is not implemented for the base ComputeOp class.")
