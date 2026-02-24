"""GymOp abstract base class and Tensor for numpy-level operation wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nkigym.codegen.context import _LoweringContext
    from nkigym.ir.types import GymStatement


@dataclass(frozen=True)
class Tensor:
    """Static descriptor for an op's input or output tensor.

    Declares the operand name and its axis layout. Each axis is either
    a ``str`` (named dimension that scales with tiling) or an ``int``
    (constant size, e.g. ``1`` for broadcast dims like ``[P, 1]``).

    Attributes:
        name: Operand slot name (e.g., ``"stationary"``, ``"data"``).
        axes: Dimension layout (e.g., ``("K", "M")`` or ``("P", 1)``).
    """

    name: str
    axes: tuple[str | int, ...]


class GymOp(ABC):
    """Thin stateless wrapper over a numpy operation.

    All state is class-level. Subclasses set class attributes and
    implement ``simulate`` with standard numpy. No ``__init__`` needed.

    Attributes:
        op_name: Unique name for registry lookup.
        inputs: Static descriptors for each input operand.
        outputs: Static descriptors for each output.
        tile_limits: Maximum tile sizes per named dimension. Empty means
            no limits (any merge size accepted).
    """

    op_name: str
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    tile_limits: dict[str, int] = {}

    _registry: dict[str, type["GymOp"]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register concrete GymOp subclasses by op_name."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "op_name") and not getattr(cls, "_abstract", False):
            GymOp._registry[cls.op_name] = cls

    @classmethod
    def get(cls, name: str) -> type["GymOp"]:
        """Look up a registered op by name.

        Args:
            name: The op_name to look up.

        Returns:
            The GymOp subclass registered under that name.

        Raises:
            KeyError: If no op is registered with the given name.
        """
        if name not in cls._registry:
            raise KeyError(f"Unknown op: {name}")
        return cls._registry[name]

    @classmethod
    def all_ops(cls) -> dict[str, type["GymOp"]]:
        """Return a copy of all registered GymOp subclasses.

        Returns:
            Dictionary mapping op_name to GymOp subclass.
        """
        return dict(cls._registry)

    @property
    def operand_names(self) -> tuple[str, ...]:
        """Input operand names derived from ``inputs``."""
        return tuple(t.name for t in self.inputs)

    @property
    def input_axes(self) -> tuple[tuple[str | int, ...], ...]:
        """Axis layouts derived from ``inputs``."""
        return tuple(t.axes for t in self.inputs)

    @property
    def output_axes(self) -> tuple[tuple[str | int, ...], ...]:
        """Axis layouts derived from ``outputs``."""
        return tuple(t.axes for t in self.outputs)

    def can_merge_operand_dim(self, operand_idx: int, dim: int, merged_size: int) -> bool:
        """Check whether merging a dimension to a given size is within tile limits.

        Maps (operand_idx, dim) to the named axis via the ``inputs`` and
        ``outputs`` descriptors, then checks ``tile_limits``. Integer
        constant axes (fixed-size dimensions like broadcast ``1``) cannot
        be widened. Named axes without a ``tile_limits`` entry are
        unconstrained.

        Args:
            operand_idx: Index into the combined (inputs + outputs) list.
            dim: Dimension index within that operand's axes tuple.
            merged_size: Proposed merged size for this dimension.

        Returns:
            True if within limits or no limit exists for the named axis.
            False if the axis is a fixed-size constant.

        Raises:
            IndexError: If operand_idx or dim is out of range.
        """
        all_tensors = self.inputs + self.outputs
        if operand_idx >= len(all_tensors):
            raise IndexError(
                f"operand_idx {operand_idx} out of range for {self.op_name} "
                f"with {len(all_tensors)} tensors (inputs + outputs)"
            )
        axes = all_tensors[operand_idx].axes
        if dim >= len(axes):
            raise IndexError(
                f"dim {dim} out of range for {self.op_name} operand "
                f"{all_tensors[operand_idx].name} with {len(axes)} axes"
            )
        axis_name = axes[dim]
        allowed = not isinstance(axis_name, int)
        if allowed:
            limit = self.tile_limits.get(axis_name)
            allowed = limit is None or merged_size <= limit
        return allowed

    def __call__(self, *args: np.ndarray, **kwargs: object) -> np.ndarray:
        """Dispatch to simulate.

        Args:
            *args: Input numpy arrays.
            **kwargs: Op-specific parameters (e.g., ``op=np.tanh``).

        Returns:
            Numpy array result.
        """
        return self.simulate(*args, **kwargs)

    @abstractmethod
    def simulate(self, *args: np.ndarray, **kwargs: object) -> np.ndarray:
        """Run the op on CPU with numpy.

        Args:
            *args: Input numpy arrays.
            **kwargs: Op-specific parameters.

        Returns:
            Numpy array result.
        """
        ...

    @abstractmethod
    def output_shape(self, input_shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
        """Infer output shape from input shapes.

        Args:
            input_shapes: Tuple of input tensor shapes.

        Returns:
            Output shape tuple.
        """
        ...

    @abstractmethod
    def to_nki(self, stmt: "GymStatement", ctx: "_LoweringContext") -> list[str]:
        """Lower a GymStatement for this op to NKI source lines.

        Args:
            stmt: The IR statement to lower.
            ctx: Mutable lowering context tracking buffers and aliases.

        Returns:
            List of NKI source code lines.
        """
        ...
