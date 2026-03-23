"""Source code utilities for extracting Python function source."""

import inspect
import textwrap
from collections.abc import Callable


def callable_to_source(func: Callable) -> str:
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
