"""Utility functions for the autotune core module."""

import os
import sys
import traceback


def capture_error_message(e) -> str:
    """
    Capture and format error message with full traceback.

    Args:
        e: The exception to capture

    Returns:
        str: Formatted error string with exception type, message, and traceback
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    error_string = f"{exc_type.__name__}: {str(e)}\n"
    error_string += "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    return error_string


def split_file_info(filepath: str) -> tuple[str, str, str]:
    """
    Split a file path into its directory, filename, and file type components.

    Args:
        filepath (str): The file path to split.

    Returns:
        tuple: A tuple containing:
            - directory (str): The directory path.
            - filename (str): The filename without extension.
            - file_type (str): The file type/extension (without the dot).

    Example:
        >>> split_file_info('/path/to/file.txt')
        ('/path/to', 'file', 'txt')
    """
    directory = os.path.dirname(filepath)
    full_filename = os.path.basename(filepath)
    filename, file_type = os.path.splitext(full_filename)
    file_type = file_type.lstrip(".")
    return directory, filename, file_type
