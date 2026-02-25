"""Logging utilities for NKI Gym.

Provides a multiline-aligned formatter and logging configuration helper.
"""

import logging

__all__ = ["setup_logging", "MultilineFormatter"]

_NOISY_LOGGERS = ("nki.compiler.backends.neuron.TraceKernel",)


class MultilineFormatter(logging.Formatter):
    """Formatter that aligns multiline messages with indentation.

    Attributes:
        msg_width: Width for message alignment.
        show_metadata: Whether to append timestamp/level/name metadata.
    """

    def __init__(self, msg_width: int, show_metadata: bool) -> None:
        """Initialize the formatter.

        Args:
            msg_width: Width for message alignment.
            show_metadata: Whether to append timestamp/level/name metadata.
        """
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.msg_width = msg_width
        self.show_metadata = show_metadata

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with multiline alignment.

        Args:
            record: The log record to format.

        Returns:
            Formatted log message string.
        """
        message = record.getMessage()
        lines = message.split("\n")

        first_line = lines[0]
        if self.show_metadata:
            metadata = f"{self.formatTime(record)} - {record.levelname} - {record.name}"
            first_line = f"{lines[0]:<{self.msg_width}}{metadata}"

        result = first_line
        if len(lines) > 1:
            continuation = "\n".join(lines[1:])
            result = f"{first_line}\n{continuation}"
        return result


def setup_logging(log_file: str, level: int, msg_width: int, show_metadata: bool) -> None:
    """Configure logging with multiline-aligned formatter.

    Args:
        log_file: Path to the log file.
        level: Logging level.
        msg_width: Width for message alignment.
        show_metadata: Whether to append timestamp/level/name metadata to log lines.
    """
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(MultilineFormatter(msg_width=msg_width, show_metadata=show_metadata))
    logging.root.addHandler(handler)
    logging.root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
