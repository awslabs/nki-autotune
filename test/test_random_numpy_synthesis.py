"""Random NumPy expression-graph coverage for programmatic synthesis."""

from __future__ import annotations

import linecache
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from nkigym.synthesis import synthesize_numpy_to_nkigym

InputSpecs = dict[str, tuple[tuple[int, ...], str]]
NumpyFunction = Callable[..., np.ndarray]
RANDOM_PROGRAM_COUNT = 12
MATRIX_EXTENTS = (128, 256)
INPUT_NAMES = ("input_a", "input_b", "input_c")
SCALARS = (0.25, 0.5, 0.75, 1.25)


@dataclass(frozen=True)
class RandomNumpyProgram:
    """One generated NumPy function and its replay metadata."""

    source: str
    input_specs: InputSpecs
    operations: tuple[str, ...]


def _combine_expression(current: str, operand: str, rng: random.Random) -> tuple[str, str]:
    """Combine the current value with one input through a random operation."""
    scalar = rng.choice(SCALARS)
    operation = rng.choice(("add", "subtract", "multiply", "maximum"))
    if operation == "add":
        expression = f"{current} + {scalar!r} * np.tanh({operand})"
    elif operation == "subtract":
        expression = f"{current} - {scalar!r} * np.tanh({operand})"
    elif operation == "multiply":
        expression = f"{current} * np.tanh({operand})"
    else:
        expression = f"np.maximum({current}, {operand})"
    return expression, operation


def _random_expression(current: str, rng: random.Random) -> tuple[str, str]:
    """Extend the graph with one randomly selected nonlinear or affine expression."""
    operand = rng.choice(INPUT_NAMES)
    scalar = rng.choice(SCALARS)
    operation = rng.choice(
        ("tanh", "square_tanh", "add_scalar", "subtract_scalar", "reverse_subtract", "multiply_scalar", "combine")
    )
    if operation == "tanh":
        expression = f"np.tanh({current})"
    elif operation == "square_tanh":
        expression = f"np.square(np.tanh({current}))"
    elif operation == "add_scalar":
        expression = f"{current} + {scalar!r}"
    elif operation == "subtract_scalar":
        expression = f"{current} - {scalar!r}"
    elif operation == "reverse_subtract":
        expression = f"{scalar!r} - {current}"
    elif operation == "multiply_scalar":
        expression = f"{current} * {scalar!r}"
    else:
        expression, operation = _combine_expression(current, operand, rng)
    return expression, operation


def _random_program(seed: int) -> RandomNumpyProgram:
    """Construct one replayable straight-line NumPy expression graph."""
    rng = random.Random(seed)
    shape = (rng.choice(MATRIX_EXTENTS), rng.choice(MATRIX_EXTENTS))
    remaining_inputs = list(INPUT_NAMES)
    rng.shuffle(remaining_inputs)
    current = remaining_inputs.pop()
    statements: list[str] = []
    operations: list[str] = []
    for operand in remaining_inputs:
        expression, operation = _combine_expression(current, operand, rng)
        target = f"value_{len(statements)}"
        statements.append(f"    {target} = {expression}")
        operations.append(operation)
        current = target
    statement_count = rng.randint(6, 10)
    while len(statements) < statement_count:
        expression, operation = _random_expression(current, rng)
        target = f"value_{len(statements)}"
        statements.append(f"    {target} = {expression}")
        operations.append(operation)
        current = target
    body = "\n".join((*statements, f"    return {current}"))
    source = f"def f_numpy({', '.join(INPUT_NAMES)}):\n{body}\n"
    input_specs: InputSpecs = {name: (shape, "bfloat16") for name in INPUT_NAMES}
    return RandomNumpyProgram(source=source, input_specs=input_specs, operations=tuple(operations))


def _compile_numpy_function(program: RandomNumpyProgram, seed: int, program_index: int) -> NumpyFunction:
    """Compile generated source while retaining it for AST-based synthesis."""
    filename = f"<random-numpy-synthesis-{seed}-{program_index}>"
    linecache.cache[filename] = (len(program.source), None, program.source.splitlines(keepends=True), filename)
    namespace: dict[str, object] = {"__name__": "__random_numpy_synthesis__", "np": np}
    exec(compile(program.source, filename, "exec"), namespace)  # noqa: S102
    function = namespace.get("f_numpy")
    if not callable(function):
        raise RuntimeError("generated source did not define f_numpy")
    return cast(NumpyFunction, function)


def _random_inputs(input_specs: InputSpecs, rng: random.Random) -> dict[str, np.ndarray]:
    """Generate independent replayable FP32 inputs."""
    numpy_rng = np.random.default_rng(rng.randrange(1 << 63))
    return {
        name: numpy_rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)
        for name, (shape, _dtype) in input_specs.items()
    }


@pytest.mark.parametrize(
    "program_index", [pytest.param(index, id=f"random_program_{index}") for index in range(RANDOM_PROGRAM_COUNT)]
)
def test_random_numpy_functions_synthesize_and_match_fp32(program_index: int) -> None:
    """Synthesize one random NumPy expression graph and compare fresh inputs."""
    seed = random.SystemRandom().randrange(1 << 63)
    rng = random.Random(seed)
    program = _random_program(seed)
    print(
        f"program={program_index} seed={seed} operations={','.join(program.operations)}\n{program.source}", flush=True
    )
    function = _compile_numpy_function(program, seed, program_index)
    kernel = synthesize_numpy_to_nkigym(function, program.input_specs, seed=rng.randrange(1 << 63))
    reference_inputs = _random_inputs(program.input_specs, rng)
    expected = function(**{name: value.copy() for name, value in reference_inputs.items()})
    actual = kernel.function(**{name: value.copy() for name, value in reference_inputs.items()})
    np.testing.assert_allclose(
        actual,
        expected,
        atol=5e-3,
        rtol=5e-3,
        err_msg=f"program={program_index} seed={seed} operations={program.operations}",
    )
