import logging

from nkigym.codegen import get_source

__all__ = ["get_source", "setup_logging", "MultilineFormatter"]


class MultilineFormatter(logging.Formatter):
    """Formatter that aligns multiline messages with indentation."""

    def __init__(self, msg_width: int, show_metadata: bool = True) -> None:
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

        if self.show_metadata:
            metadata = f"{self.formatTime(record)} - {record.levelname} - {record.name}"
            first_line = f"{lines[0]:<{self.msg_width}}{metadata}"
        else:
            first_line = lines[0]

        if len(lines) == 1:
            return first_line

        continuation = "\n".join(lines[1:])
        return f"{first_line}\n{continuation}"


def setup_logging(log_file: str, level: int = logging.DEBUG, msg_width: int = 300, show_metadata: bool = False) -> None:
    """Configure logging with multiline-aligned formatter.

    Args:
        log_file: Path to the log file.
        level: Logging level (default: DEBUG).
        msg_width: Width for message alignment (default: 300).
        show_metadata: Whether to append timestamp/level/name metadata to log lines.
    """
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(MultilineFormatter(msg_width=msg_width, show_metadata=show_metadata))
    logging.root.addHandler(handler)
    logging.root.setLevel(level)
