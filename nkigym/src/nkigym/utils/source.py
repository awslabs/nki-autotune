"""Source code manipulation utilities for NKI Gym.

Provides source code round-tripping: extracting source from callables
and compiling source strings back into callable functions.
"""

import ast
import inspect
from collections.abc import Callable

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
        source = inspect.getsource(func)
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


def exec_tree_to_func(tree: ast.Module, func_name: str) -> Callable[..., np.ndarray]:
    """Compile an AST tree directly and return the named function.

    Avoids the string round-trip of ``exec(source_string)`` by compiling
    the AST directly. The unparsed source is still attached as
    ``__source__`` for deduplication and future re-parsing.

    Args:
        tree: An ``ast.Module`` node with locations already set.
        func_name: Name of the function to extract from executed namespace.

    Returns:
        Callable function with ``__source__`` attribute attached.

    Raises:
        ValueError: If the named function is not found in the executed code.
    """
    code = compile(tree, "<nkigym>", "exec")
    source = ast.unparse(tree)
    namespace: dict[str, object] = {"np": np, "nkigym": nkigym}
    exec(code, namespace)
    func = namespace.get(func_name)
    if func is None:
        raise ValueError(f"Function '{func_name}' not found in compiled tree")
    func.__source__ = source
    return func
