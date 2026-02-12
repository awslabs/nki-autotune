"""Source code manipulation utilities for NKI Gym.

Provides source code round-tripping: extracting source from callables
and compiling source strings back into callable functions.
"""

import importlib.util
import inspect
import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np

import nkigym


def get_source(func: Callable) -> str:
    """Get source code for a function (dynamic or static).

    Args:
        func: A callable function, either statically defined or dynamically
              generated (with __source__ attribute).

    Returns:
        Source code string for the function.
    """
    source = getattr(func, "__source__", None)
    if source is None:
        source = textwrap.dedent(inspect.getsource(func))
    return source


def exec_source_to_func(source: str, func_name: str) -> Callable[..., np.ndarray]:
    """Execute source code and return the named function.

    Args:
        source: Python source code string containing a function definition.
        func_name: Name of the function to extract from executed namespace.

    Returns:
        Callable function from the executed source with __source__ attribute attached.

    Raises:
        ValueError: If the named function is not found in the executed source.
    """
    namespace: dict[str, object] = {"np": np, "nkigym": nkigym}
    exec(source, namespace)
    func = namespace.get(func_name)
    if func is None:
        raise ValueError(f"Function '{func_name}' not found in executed source")
    func.__source__ = source
    return func


def import_func(file_path: Path, func_name: str) -> Callable:
    """Import a function by name from a Python source file.

    Args:
        file_path: Path to the Python source file.
        func_name: Name of the function to import from the module.

    Returns:
        The imported callable.

    Raises:
        ImportError: If the module spec cannot be created or has no loader.
        AttributeError: If the function is not found in the module.
    """
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)
