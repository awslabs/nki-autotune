"""GymOp abstract base class and Tensor for numpy-level operation wrappers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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
    """

    op_name: str
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]

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
