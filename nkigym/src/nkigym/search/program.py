"""Durable program artifacts consumed by agentic tuning."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from nkigym.ops import nkigym_kernel as _nkigym_kernel
from nkigym.search.types import InputSpecs

_METADATA_FILENAME = "program.json"
_NKIGYM_FILENAME = "f_nkigym.py"


@dataclass(frozen=True)
class ProgramSpec:
    """Persisted nkigym program and its tuning workload."""

    name: str
    nkigym_source: str
    input_specs: InputSpecs
    workload_guidance: str
    neuronx_cc_args: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata without duplicating source artifacts."""
        input_specs = {
            name: {"shape": list(shape), "dtype": dtype} for name, (shape, dtype) in self.input_specs.items()
        }
        return {
            "name": self.name,
            "input_specs": input_specs,
            "workload_guidance": self.workload_guidance,
            "neuronx_cc_args": list(self.neuronx_cc_args),
        }


def _dependency_line(name: str, value: Any) -> str:
    """Serialize one referenced nkigym operation or literal constant."""
    module_name = getattr(value, "__module__", None)
    exported_name = getattr(value, "__name__", None)
    if isinstance(module_name, str) and module_name.startswith("nkigym.ops.") and isinstance(exported_name, str):
        alias = "" if name == exported_name else f" as {name}"
        line = f"from {module_name} import {exported_name}{alias}"
    else:
        rendered = repr(value)
        try:
            ast.literal_eval(rendered)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"f_nkigym global {name!r} must be an nkigym operation or a literal constant") from error
        line = f"{name} = {rendered}"
    return line


def _kernel_decorator_import(original: Callable[..., Any], function_source: str) -> str:
    """Preserve a plain-name alias used by the nkigym kernel decorator."""
    parsed = ast.parse(function_source)
    definitions = [
        node
        for node in parsed.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == original.__name__
    ]
    decorator_name = "nkigym_kernel"
    if len(definitions) == 1:
        namespace = getattr(original, "__globals__", {})
        aliases = [
            decorator.id
            for decorator in definitions[0].decorator_list
            if isinstance(decorator, ast.Name) and namespace.get(decorator.id) is _nkigym_kernel
        ]
        if len(aliases) == 1:
            decorator_name = aliases[0]
    alias = "" if decorator_name == "nkigym_kernel" else f" as {decorator_name}"
    return f"from nkigym.ops import nkigym_kernel{alias}"


def _nkigym_source(f_nkigym: Callable[..., Any]) -> str:
    """Build a standalone module for one source-backed nkigym function."""
    original = inspect.unwrap(f_nkigym)
    try:
        function_source = textwrap.dedent(inspect.getsource(original))
    except (OSError, TypeError) as error:
        raise ValueError("f_nkigym source is unavailable; define it in a Python module") from error
    closure = inspect.getclosurevars(original)
    if closure.unbound:
        names = ", ".join(sorted(closure.unbound))
        raise ValueError(f"f_nkigym has unresolved global names: {names}")
    dependencies = {**closure.globals, **closure.nonlocals}
    dependency_lines = [_dependency_line(name, dependencies[name]) for name in sorted(dependencies)]
    decorator_import = _kernel_decorator_import(original, function_source)
    preamble = ["from __future__ import annotations", "", decorator_import, *dependency_lines]
    return "\n".join(preamble) + "\n\n\n" + function_source.rstrip() + "\n"


def _validated_input_specs(f_nkigym: Callable[..., Any], input_specs: InputSpecs, lnc: int) -> InputSpecs:
    """Validate the nkigym marker, profile shapes, and parameter ordering."""
    if f_nkigym.__name__ != "f_nkigym":
        raise ValueError(f"agentic tuning entry point must be named f_nkigym, got {f_nkigym.__name__!r}")
    if not getattr(f_nkigym, "__nkigym_kernel__", False):
        raise ValueError("f_nkigym must be decorated with @nkigym_kernel")
    normalized = {name: (tuple(shape), dtype) for name, (shape, dtype) in input_specs.items()}
    if not normalized:
        raise ValueError("input_specs must not be empty")
    for name, (shape, dtype) in normalized.items():
        if not name.isidentifier():
            raise ValueError(f"input name must be a Python identifier: {name!r}")
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"input {name!r} must have a non-empty positive shape")
        if not dtype:
            raise ValueError(f"input {name!r} must have a dtype")
    if lnc not in {1, 2}:
        raise ValueError("profile lnc must be 1 or 2")
    parameters = list(inspect.signature(f_nkigym).parameters)
    if parameters != list(normalized):
        raise ValueError(f"f_nkigym parameters {parameters} do not match input_specs keys {list(normalized)}")
    return normalized


def _workload_guidance(name: str, input_specs: InputSpecs) -> str:
    """Describe the nkigym program and profile shape to the transform policy."""
    inputs = ", ".join(f"{input_name}={shape}:{dtype}" for input_name, (shape, dtype) in input_specs.items())
    return (
        f"Optimize nkigym function {name} with inputs {inputs}. "
        "Use measured MFU and compiler failures to apply up to three compatible atomic transforms or revisit an "
        "earlier branch point per reasoning step."
    )


def program_from_callable(
    f_nkigym: Callable[..., Any], input_specs: InputSpecs, neuronx_cc_args: tuple[str, ...], lnc: int
) -> ProgramSpec:
    """Build a durable agentic tuning program from a decorated callable."""
    normalized_specs = _validated_input_specs(f_nkigym, input_specs, lnc)
    return ProgramSpec(
        name=f_nkigym.__name__,
        nkigym_source=_nkigym_source(f_nkigym),
        input_specs=normalized_specs,
        workload_guidance=_workload_guidance(f_nkigym.__name__, normalized_specs),
        neuronx_cc_args=neuronx_cc_args,
    )


def _decode_input_specs(value: object) -> InputSpecs:
    """Decode input specifications from program metadata."""
    if not isinstance(value, dict) or not value:
        raise ValueError("program input_specs must be a non-empty object")
    input_specs: InputSpecs = {}
    for name, raw_spec in value.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"invalid program input name: {name!r}")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"program input spec for {name!r} must be an object")
        raw_shape = raw_spec.get("shape")
        dtype = raw_spec.get("dtype")
        if not isinstance(raw_shape, list) or not raw_shape:
            raise ValueError(f"program input {name!r} must have a non-empty shape")
        if any(
            not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1 for dimension in raw_shape
        ):
            raise ValueError(f"program input {name!r} shape must contain positive integers")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"program input {name!r} must have a dtype")
        input_specs[name] = (tuple(raw_shape), dtype)
    return input_specs


def write_program(program: ProgramSpec, directory: Path) -> None:
    """Persist nkigym source and structured metadata."""
    directory.mkdir(parents=True)
    (directory / _NKIGYM_FILENAME).write_text(program.nkigym_source, encoding="utf-8")
    (directory / _METADATA_FILENAME).write_text(json.dumps(program.as_dict(), indent=2) + "\n", encoding="utf-8")


def read_program(directory: Path) -> ProgramSpec:
    """Load a persisted program and reject malformed metadata."""
    metadata_path = directory / _METADATA_FILENAME
    decoded = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"program metadata is not an object: {metadata_path}")
    name = decoded.get("name")
    guidance = decoded.get("workload_guidance")
    raw_compiler_args = decoded.get("neuronx_cc_args")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"invalid program name: {name!r}")
    if not isinstance(guidance, str) or not guidance.strip():
        raise ValueError("program workload_guidance must be a non-empty string")
    if not isinstance(raw_compiler_args, list) or any(not isinstance(arg, str) for arg in raw_compiler_args):
        raise ValueError("program neuronx_cc_args must be a list of strings")
    program = ProgramSpec(
        name=name,
        nkigym_source=(directory / _NKIGYM_FILENAME).read_text(encoding="utf-8"),
        input_specs=_decode_input_specs(decoded.get("input_specs")),
        workload_guidance=guidance,
        neuronx_cc_args=tuple(raw_compiler_args),
    )
    return program


def nkigym_source_path(directory: Path) -> Path:
    """Return the generated module path inside a program artifact directory."""
    return directory / _NKIGYM_FILENAME


def _load_module(path: Path) -> ModuleType:
    """Import nkigym source from its durable file."""
    spec = importlib.util.spec_from_file_location("nkigym_search_program", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load generated nkigym module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_nkigym_program(directory: Path) -> tuple[ProgramSpec, Callable[..., Any]]:
    """Load and validate a persisted ``f_nkigym`` callable and its metadata."""
    program = read_program(directory)
    path = nkigym_source_path(directory)
    module = _load_module(path)
    candidate = getattr(module, "f_nkigym", None)
    if not callable(candidate):
        raise ValueError(f"program source does not define callable f_nkigym: {path}")
    kernel = cast(Callable[..., Any], candidate)
    if not getattr(kernel, "__nkigym_kernel__", False):
        raise ValueError("f_nkigym must be decorated with @nkigym_kernel")
    parameters = list(inspect.signature(kernel).parameters)
    if parameters != list(program.input_specs):
        raise ValueError(f"f_nkigym parameters {parameters} do not match input_specs keys {list(program.input_specs)}")
    return program, kernel


__all__ = [
    "ProgramSpec",
    "load_nkigym_program",
    "nkigym_source_path",
    "program_from_callable",
    "read_program",
    "write_program",
]
