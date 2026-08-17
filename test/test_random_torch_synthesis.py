"""Random Torch expression-graph coverage for programmatic synthesis."""

from __future__ import annotations

import linecache
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest
import torch

from nkigym.profile import InputSpecs
from nkigym.synthesis import synthesize_torch_to_nkigym

TorchFunction = Callable[..., torch.Tensor]
RANDOM_PROGRAM_COUNT = 12
MATRIX_EXTENTS = (128, 256)
INPUT_NAMES = ("input_a", "input_b", "input_c")
SCALARS = (0.25, 0.5, 0.75, 1.25)


@dataclass(frozen=True)
class RandomTorchProgram:
    """One generated Torch function and its replay metadata."""

    source: str
    input_specs: InputSpecs
    operations: tuple[str, ...]


def _combine_expression(current: str, operand: str, rng: random.Random) -> tuple[str, str]:
    """Combine the current value with one input through a random operation."""
    scalar = rng.choice(SCALARS)
    operation = rng.choice(("add", "subtract", "multiply", "maximum"))
    if operation == "add":
        expression = f"{current} + {scalar!r} * torch.tanh({operand})"
    elif operation == "subtract":
        expression = f"{current} - {scalar!r} * torch.tanh({operand})"
    elif operation == "multiply":
        expression = f"{current} * torch.tanh({operand})"
    else:
        expression = f"torch.maximum({current}, {operand})"
    return expression, operation


def _random_expression(current: str, rng: random.Random) -> tuple[str, str]:
    """Extend the graph with one randomly selected nonlinear or affine expression."""
    operand = rng.choice(INPUT_NAMES)
    scalar = rng.choice(SCALARS)
    operation = rng.choice(
        ("tanh", "square_tanh", "add_scalar", "subtract_scalar", "reverse_subtract", "multiply_scalar", "combine")
    )
    if operation == "tanh":
        expression = f"torch.tanh({current})"
    elif operation == "square_tanh":
        expression = f"torch.square(torch.tanh({current}))"
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


def _random_program(seed: int) -> RandomTorchProgram:
    """Construct one replayable straight-line Torch expression graph."""
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
    source = f"def f_torch({', '.join(INPUT_NAMES)}):\n{body}\n"
    input_specs: InputSpecs = {name: (shape, "bfloat16") for name in INPUT_NAMES}
    return RandomTorchProgram(source=source, input_specs=input_specs, operations=tuple(operations))


def _compile_torch_function(program: RandomTorchProgram, seed: int, program_index: int) -> TorchFunction:
    """Compile generated source while retaining it for AST-based synthesis."""
    filename = f"<random-torch-synthesis-{seed}-{program_index}>"
    linecache.cache[filename] = (len(program.source), None, program.source.splitlines(keepends=True), filename)
    namespace: dict[str, object] = {"__name__": "__random_torch_synthesis__", "torch": torch}
    exec(compile(program.source, filename, "exec"), namespace)  # noqa: S102
    function = namespace.get("f_torch")
    if not callable(function):
        raise RuntimeError("generated source did not define f_torch")
    return cast(TorchFunction, function)


def _random_inputs(input_specs: InputSpecs, rng: random.Random) -> dict[str, torch.Tensor]:
    """Generate independent replayable FP32 inputs."""
    generator = torch.Generator().manual_seed(rng.randrange(1 << 63))
    return {
        name: torch.rand(shape, generator=generator, dtype=torch.float32) - 0.5
        for name, (shape, _dtype) in input_specs.items()
    }


@pytest.mark.parametrize(
    "program_index", [pytest.param(index, id=f"random_program_{index}") for index in range(RANDOM_PROGRAM_COUNT)]
)
def test_random_torch_functions_synthesize_and_match_fp32(program_index: int) -> None:
    """Synthesize one random Torch expression graph and compare fresh inputs."""
    seed = random.SystemRandom().randrange(1 << 63)
    rng = random.Random(seed)
    program = _random_program(seed)
    print(
        f"program={program_index} seed={seed} operations={','.join(program.operations)}\n{program.source}", flush=True
    )
    function = _compile_torch_function(program, seed, program_index)
    kernel = synthesize_torch_to_nkigym(function, program.input_specs, seed=rng.randrange(1 << 63))
    reference_inputs = _random_inputs(program.input_specs, rng)
    expected = kernel.adapt_output(function(**{name: value.clone() for name, value in reference_inputs.items()}))
    kernel_inputs = kernel.adapt_inputs({name: value.clone() for name, value in reference_inputs.items()})
    actual = kernel.function(**kernel_inputs)
    torch.testing.assert_close(
        torch.as_tensor(actual),
        torch.as_tensor(expected),
        atol=5e-3,
        rtol=5e-3,
        msg=f"program={program_index} seed={seed} operations={program.operations}",
    )
