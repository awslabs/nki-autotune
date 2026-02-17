"""Unit tests for nkigym.utils.source module."""

import pytest
from utils_golden import (
    CUSTOM_SOURCE,
    EXEC_BAD_SYNTAX_SOURCE,
    EXEC_GREET_SOURCE,
    EXEC_HAPPY_SOURCE,
    EXEC_MISSING_SOURCE,
    EXEC_NKIGYM_SOURCE,
    EXEC_NUMPY_SOURCE,
    STATIC_FUNCTION_RETURN_MARKER,
    STATIC_FUNCTION_SOURCE_MARKER,
)

import nkigym as nkigym_mod
from nkigym.utils.source import callable_to_source, source_to_callable


def _static_function(x: int) -> int:
    """A statically defined function for testing callable_to_source fallback."""
    return x + 1


class TestCallableToSource:
    """Tests for callable_to_source()."""

    def test_returns_dunder_source_when_present(self) -> None:
        """When __source__ is set, callable_to_source returns it directly."""

        def dummy() -> None:
            pass

        dummy.__source__ = "def dummy(): pass"
        result = callable_to_source(dummy)
        assert result == "def dummy(): pass"

    def test_falls_back_to_inspect_getsource(self) -> None:
        """When __source__ is absent, callable_to_source falls back to inspect.getsource."""
        result = callable_to_source(_static_function)
        assert STATIC_FUNCTION_SOURCE_MARKER in result
        assert STATIC_FUNCTION_RETURN_MARKER in result

    def test_prefers_dunder_source_over_inspect(self) -> None:
        """When __source__ is set, inspect.getsource is not used."""
        _static_function.__source__ = CUSTOM_SOURCE
        try:
            result = callable_to_source(_static_function)
            assert result == CUSTOM_SOURCE
        finally:
            del _static_function.__source__


class TestSourceToCallable:
    """Tests for source_to_callable()."""

    def test_happy_path(self) -> None:
        """Executes valid source and returns the named function."""
        func = source_to_callable(EXEC_HAPPY_SOURCE, "add")
        assert func(2, 3) == 5

    def test_attaches_source_attribute(self) -> None:
        """Returned function has __source__ set to the input source string."""
        func = source_to_callable(EXEC_GREET_SOURCE, "greet")
        assert func.__source__ == EXEC_GREET_SOURCE

    def test_raises_value_error_for_missing_function(self) -> None:
        """Raises ValueError when func_name is not defined in the source."""
        with pytest.raises(ValueError, match="'bar' not found"):
            source_to_callable(EXEC_MISSING_SOURCE, "bar")

    def test_raises_syntax_error_for_invalid_source(self) -> None:
        """Raises SyntaxError when source contains invalid Python."""
        with pytest.raises(SyntaxError):
            source_to_callable(EXEC_BAD_SYNTAX_SOURCE, "broken")

    def test_numpy_available_via_import(self) -> None:
        """Source code can use np (numpy) via its own import."""
        func = source_to_callable(EXEC_NUMPY_SOURCE, "zeros")
        result = func()
        assert result.shape == (3,)
        assert (result == 0).all()

    def test_nkigym_available_via_import(self) -> None:
        """Source code can reference nkigym via its own import."""
        func = source_to_callable(EXEC_NKIGYM_SOURCE, "get_mod")
        assert func() is nkigym_mod
