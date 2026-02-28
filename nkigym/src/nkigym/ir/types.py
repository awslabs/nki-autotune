"""IR type definitions: GymStatement and GymProgram."""

from typing import Any, NamedTuple

import numpy as np

from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import GymOp


class GymStatement(NamedTuple):
    """A single IR statement: output = op(**kwargs).

    Tensor kwargs map operand names to TensorRef (with shape and slices).
    Config kwargs map parameter names to Python objects (e.g.,
    ``("op", np.tanh)``, ``("dtype", np.float16)``).

    Attributes:
        op: GymOp name (e.g., ``"nc_matmul"``).
        kwargs: All arguments as ``(param_name, value)`` pairs.
        output: Output tensor reference.
    """

    op: str
    kwargs: tuple[tuple[str, Any], ...]
    output: TensorRef

    def __call__(self, env: dict[str, np.ndarray]) -> None:
        """Execute this statement against the variable environment.

        Resolves TensorRef kwargs from *env*, dispatches to the
        corresponding ``GymOp.simulate``, and writes the result back
        into *env*.  Store ops return ``None`` to signal mutation-only
        (no env assignment needed).

        Args:
            env: Variable environment mapping names to arrays.
        """
        args, kwargs = _resolve_args(self, env)
        result = GymOp.get(self.op).simulate(*args, **kwargs)
        if result is not None:
            env[self.output.name] = result


class GymProgram(NamedTuple):
    """Immutable program IR: a named function as a sequence of GymOp calls.

    Programs are hashable and can be used as dictionary keys for
    deduplication in the transform search graph.

    Attributes:
        name: Function name.
        params: Input parameter names.
        input_shapes: Per-parameter expected shapes as ``(name, shape)`` pairs.
        stmts: Sequence of GymStatement operations.
        return_var: Variable name returned by the function.
        output_dtype: Numpy dtype type for output allocation.
    """

    name: str
    params: tuple[str, ...]
    input_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    stmts: tuple[GymStatement, ...]
    return_var: str
    output_dtype: type

    def __repr__(self) -> str:
        """Return a readable multi-line representation."""
        lines = [f"GymProgram(name={self.name!r},"]
        lines.append(f"  params={self.params!r},")
        lines.append(f"  input_shapes={self.input_shapes!r},")
        lines.append("  stmts=(")
        for stmt in self.stmts:
            lines.append(f"    {stmt!r},")
        lines.append("  ),")
        lines.append(f"  return_var={self.return_var!r},")
        lines.append(f"  output_dtype={self.output_dtype!r})")
        return "\n".join(lines)

    def __call__(self, **inputs: np.ndarray) -> np.ndarray:
        """Execute the program by interpreting IR statements directly.

        Each statement resolves its own TensorRefs and dispatches to
        the corresponding GymOp.  Returns the final result array.

        Args:
            **inputs: Input arrays keyed by parameter name.

        Returns:
            The output array (value of the return variable).
        """
        env: dict[str, np.ndarray] = dict(inputs)
        for stmt in self.stmts:
            stmt(env)
        return env[self.return_var]


def _resolve_args(stmt: GymStatement, env: dict[str, np.ndarray]) -> tuple[list[np.ndarray], dict[str, object]]:
    """Resolve statement kwargs into positional and keyword args for simulate.

    TensorRef values are looked up in *env* and sliced via
    ``to_slices()``.  The ``acc`` key becomes a kwarg; other TensorRefs
    become positional args.  Non-TensorRef values pass through as kwargs.

    Args:
        stmt: IR statement whose kwargs to resolve.
        env: Variable environment mapping names to arrays.

    Returns:
        Tuple of (positional args, keyword kwargs) for simulate().
    """
    args: list[np.ndarray] = []
    kwargs: dict[str, object] = {}
    for key, value in stmt.kwargs:
        if isinstance(value, TensorRef):
            arr = env[value.name]
            if value.slices:
                arr = arr[value.to_slices()]
            if key == "acc":
                kwargs["acc"] = arr
            else:
                args.append(arr)
        else:
            kwargs[key] = value
    kwargs["output_shape"] = stmt.output.shape
    return (args, kwargs)
