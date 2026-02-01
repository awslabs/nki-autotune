"""NumPy operation implementations for TracedTensor tracing.

This module provides traced implementations of NumPy operations that track
dimension information during symbolic execution.

Adding new operations:
    1. Define a function that takes TracedTensor args and returns TracedTensor
    2. Register it in HANDLED_FUNCTIONS: HANDLED_FUNCTIONS[np.your_op] = your_func
    3. Handle dimension unification as needed (see _traced_matmul for example)
    4. Add semantics to OP_SEMANTICS if the op needs code generation support
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

HANDLED_FUNCTIONS: dict[Callable, Callable] = {}

if TYPE_CHECKING:
    from nkigym.tensor import TracedTensor


@dataclass
class OpSemantics:
    """Defines how an operator generates code and handles reduction tiling.

    Attributes:
        op_name: Name of the operator (e.g., "matmul", "add").
        generate_expr: Function to generate the NumPy expression.
            Takes list of input variable names and returns expression string.
        combine_partials: Function to generate in-place accumulation expression.
            Takes (result_var, input_vars) and returns the += expression.
            None for ops without reduction dimensions.
    """

    op_name: str
    generate_expr: Callable[[list[str]], str]
    combine_partials: Callable[[str, list[str]], str] | None


def _matmul_expr(inputs: list[str]) -> str:
    """Generate matmul expression.

    Args:
        inputs: List of input variable names [a, b].

    Returns:
        NumPy matmul expression string.
    """
    return f"np.matmul({inputs[0]}, {inputs[1]})"


def _add_expr(inputs: list[str]) -> str:
    """Generate add expression.

    Args:
        inputs: List of input variable names [a, b].

    Returns:
        NumPy add expression string.
    """
    return f"np.add({inputs[0]}, {inputs[1]})"


def _multiply_expr(inputs: list[str]) -> str:
    """Generate multiply expression.

    Args:
        inputs: List of input variable names [a, b].

    Returns:
        NumPy multiply expression string.
    """
    return f"np.multiply({inputs[0]}, {inputs[1]})"


def _reduce_max_expr(inputs: list[str]) -> str:
    """Generate reduce_max expression.

    Args:
        inputs: List of input variable names [a].

    Returns:
        NumPy max expression string with axis=1.
    """
    return f"np.max({inputs[0]}, axis=1)"


def _reduce_sum_expr(inputs: list[str]) -> str:
    """Generate reduce_sum expression.

    Args:
        inputs: List of input variable names [a].

    Returns:
        NumPy sum expression string with axis=1.
    """
    return f"np.sum({inputs[0]}, axis=1)"


def _matmul_accumulate(result_var: str, inputs: list[str]) -> str:
    """Generate in-place matmul accumulation.

    Args:
        result_var: Name of the result variable to accumulate into.
        inputs: List of input variable names [a, b].

    Returns:
        In-place accumulation expression string.
    """
    return f"{result_var} += np.matmul({inputs[0]}, {inputs[1]})"


def _reduce_max_accumulate(result_var: str, inputs: list[str]) -> str:
    """Generate in-place reduce_max accumulation.

    Args:
        result_var: Name of the result variable to accumulate into.
        inputs: List of input variable names [a].

    Returns:
        In-place accumulation expression string using np.maximum.
    """
    return f"{result_var} = np.maximum({result_var}, np.max({inputs[0]}, axis=1))"


def _reduce_sum_accumulate(result_var: str, inputs: list[str]) -> str:
    """Generate in-place reduce_sum accumulation.

    Args:
        result_var: Name of the result variable to accumulate into.
        inputs: List of input variable names [a].

    Returns:
        In-place accumulation expression string (additive).
    """
    return f"{result_var} += np.sum({inputs[0]}, axis=1)"


OP_SEMANTICS: dict[str, OpSemantics] = {
    "matmul": OpSemantics(op_name="matmul", generate_expr=_matmul_expr, combine_partials=_matmul_accumulate),
    "add": OpSemantics(op_name="add", generate_expr=_add_expr, combine_partials=None),
    "multiply": OpSemantics(op_name="multiply", generate_expr=_multiply_expr, combine_partials=None),
    "reduce_sum": OpSemantics(
        op_name="reduce_sum", generate_expr=_reduce_sum_expr, combine_partials=_reduce_sum_accumulate
    ),
    "reduce_max": OpSemantics(
        op_name="reduce_max", generate_expr=_reduce_max_expr, combine_partials=_reduce_max_accumulate
    ),
}


def _traced_matmul(a: "TracedTensor", b: "TracedTensor") -> "TracedTensor":
    """Handle np.matmul for TracedTensor: C[m,n] = A[m,k] @ B[k,n].

    Matmul semantics: contracts a.dims[1] (K) with b.dims[0] (K).
    These dimensions are unified since they must be equal.

    Args:
        a: Left-hand side tensor of shape (M, K).
        b: Right-hand side tensor of shape (K, N).

    Returns:
        TracedTensor of shape (M, N) with dims [a.dims[0], b.dims[1]].

    Raises:
        TypeError: If inputs are not TracedTensors.
    """
    from nkigym.tensor import TracedTensor

    if not isinstance(a, TracedTensor) or not isinstance(b, TracedTensor):
        raise TypeError(f"Expected TracedTensor, got {type(a)} and {type(b)}")

    a.tracker.unify(a.dims[1], b.dims[0])

    result_dims = [a.dims[0], b.dims[1]]
    result_shape = (a.shape[0], b.shape[1])

    output_name = a.tracker.new_intermediate_name()
    a.tracker.record_op("matmul", [a.name, b.name], output_name)

    return TracedTensor(name=output_name, shape=result_shape, dims=result_dims, tracker=a.tracker)


HANDLED_FUNCTIONS[np.matmul] = _traced_matmul
