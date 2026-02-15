"""IR type definitions: GymStatement and GymProgram."""

from typing import Any, NamedTuple

from nkigym.ir.tensor import TensorRef


class GymStatement(NamedTuple):
    """A single IR statement: output = op(**kwargs).

    Tensor kwargs map operand names to TensorRef (with shape and slices)
    or plain strings. Config kwargs map parameter names to source-level
    string literals (e.g., ``("op", "np.tanh")``).

    Attributes:
        op: GymOp name (e.g., ``"nc_matmul"``).
        kwargs: All arguments as ``(param_name, value)`` pairs.
        output: Output tensor reference.
    """

    op: str
    kwargs: tuple[tuple[str, Any], ...]
    output: TensorRef


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
