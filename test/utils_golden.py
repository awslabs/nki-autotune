"""Golden data for utils tests."""

STATIC_FUNCTION_SOURCE_MARKER = "def _static_function"
STATIC_FUNCTION_RETURN_MARKER = "return x + 1"
CUSTOM_SOURCE = "def _static_function(): return 42"

EXEC_HAPPY_SOURCE = "def add(a, b):\n    return a + b\n"
EXEC_GREET_SOURCE = "def greet():\n    return 'hello'\n"
EXEC_MISSING_SOURCE = "def foo():\n    return 1\n"
EXEC_BAD_SYNTAX_SOURCE = "def broken(:\n    return 1\n"
EXEC_NUMPY_SOURCE = "def zeros():\n    return np.zeros(3)\n"
EXEC_NKIGYM_SOURCE = "def get_mod():\n    return nkigym\n"
