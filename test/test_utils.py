"""Unit tests for nkigym.utils.source module."""

import pytest

import nkigym as nkigym_mod
from nkigym.utils.source import exec_source_to_func, get_source


def _static_function(x: int) -> int:
    """A statically defined function for testing get_source fallback."""
    return x + 1


class TestGetSource:
    """Tests for get_source()."""

    def test_returns_dunder_source_when_present(self) -> None:
        """When __source__ is set, get_source returns it directly."""

        def dummy() -> None:
            pass

        dummy.__source__ = "def dummy(): pass"
        result = get_source(dummy)
        assert result == "def dummy(): pass"

    def test_falls_back_to_inspect_getsource(self) -> None:
        """When __source__ is absent, get_source falls back to inspect.getsource."""
        result = get_source(_static_function)
        assert "def _static_function" in result
        assert "return x + 1" in result

    def test_prefers_dunder_source_over_inspect(self) -> None:
        """When __source__ is set, inspect.getsource is not used."""
        custom_source = "def _static_function(): return 42"
        _static_function.__source__ = custom_source
        try:
            result = get_source(_static_function)
            assert result == custom_source
        finally:
            del _static_function.__source__


class TestExecSourceToFunc:
    """Tests for exec_source_to_func()."""

    def test_happy_path(self) -> None:
        """Executes valid source and returns the named function."""
        source = "def add(a, b):\n    return a + b\n"
        func = exec_source_to_func(source, "add")
        assert func(2, 3) == 5

    def test_attaches_source_attribute(self) -> None:
        """Returned function has __source__ set to the input source string."""
        source = "def greet():\n    return 'hello'\n"
        func = exec_source_to_func(source, "greet")
        assert func.__source__ == source

    def test_raises_value_error_for_missing_function(self) -> None:
        """Raises ValueError when func_name is not defined in the source."""
        source = "def foo():\n    return 1\n"
        with pytest.raises(ValueError, match="'bar' not found"):
            exec_source_to_func(source, "bar")

    def test_raises_syntax_error_for_invalid_source(self) -> None:
        """Raises SyntaxError when source contains invalid Python."""
        source = "def broken(:\n    return 1\n"
        with pytest.raises(SyntaxError):
            exec_source_to_func(source, "broken")

    def test_numpy_available_in_namespace(self) -> None:
        """Source code can use np (numpy) from the execution namespace."""
        source = "def zeros():\n    return np.zeros(3)\n"
        func = exec_source_to_func(source, "zeros")
        result = func()
        assert result.shape == (3,)
        assert (result == 0).all()

    def test_nkigym_available_in_namespace(self) -> None:
        """Source code can reference nkigym from the execution namespace."""
        source = "def get_mod():\n    return nkigym\n"
        func = exec_source_to_func(source, "get_mod")
        assert func() is nkigym_mod
