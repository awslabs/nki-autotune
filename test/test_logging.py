"""Unit tests for nkigym.utils.logging module.

Tests MultilineFormatter formatting and setup_logging configuration.

Run with: pytest test/test_logging.py -v
"""

import logging

import pytest

from nkigym.utils.logging import MultilineFormatter, setup_logging


class TestMultilineFormatterSingleLine:
    """Tests for MultilineFormatter with single-line messages."""

    def test_single_line_with_metadata(self) -> None:
        """Single-line message includes right-padded text and metadata suffix."""
        formatter = MultilineFormatter(msg_width=40, show_metadata=True)
        record = logging.LogRecord(
            name="test.logger", level=logging.INFO, pathname="", lineno=0, msg="hello world", args=(), exc_info=None
        )
        result = formatter.format(record)
        assert result.startswith("hello world")
        assert "INFO" in result
        assert "test.logger" in result

    def test_single_line_without_metadata(self) -> None:
        """Single-line message without metadata returns just the message."""
        formatter = MultilineFormatter(msg_width=40, show_metadata=False)
        record = logging.LogRecord(
            name="test.logger", level=logging.INFO, pathname="", lineno=0, msg="hello world", args=(), exc_info=None
        )
        result = formatter.format(record)
        assert result == "hello world"

    def test_single_line_padding(self) -> None:
        """Single-line message is padded to msg_width before metadata."""
        formatter = MultilineFormatter(msg_width=50, show_metadata=True)
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="", lineno=0, msg="short", args=(), exc_info=None
        )
        result = formatter.format(record)
        metadata_start = result.index("20")
        assert metadata_start >= 50


class TestMultilineFormatterMultiLine:
    """Tests for MultilineFormatter with multiline messages."""

    def test_multiline_with_metadata(self) -> None:
        """Multiline message: first line gets metadata, continuation lines are preserved."""
        formatter = MultilineFormatter(msg_width=40, show_metadata=True)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="line one\nline two\nline three",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "WARNING" in lines[0]
        assert "test.logger" in lines[0]
        assert lines[1] == "line two"
        assert lines[2] == "line three"

    def test_multiline_without_metadata(self) -> None:
        """Multiline message without metadata preserves all lines."""
        formatter = MultilineFormatter(msg_width=40, show_metadata=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="first\nsecond", args=(), exc_info=None
        )
        result = formatter.format(record)
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "first"
        assert lines[1] == "second"

    def test_multiline_first_line_padded(self) -> None:
        """First line of multiline message is padded to msg_width."""
        formatter = MultilineFormatter(msg_width=60, show_metadata=True)
        record = logging.LogRecord(
            name="x", level=logging.INFO, pathname="", lineno=0, msg="abc\ndef", args=(), exc_info=None
        )
        result = formatter.format(record)
        first_line = result.split("\n")[0]
        metadata_start = first_line.index("20")
        assert metadata_start >= 60


class TestMultilineFormatterInit:
    """Tests for MultilineFormatter initialization."""

    def test_default_show_metadata(self) -> None:
        """show_metadata defaults to True."""
        formatter = MultilineFormatter(msg_width=80)
        assert formatter.show_metadata is True

    def test_custom_show_metadata(self) -> None:
        """show_metadata can be set to False."""
        formatter = MultilineFormatter(msg_width=80, show_metadata=False)
        assert formatter.show_metadata is False

    def test_msg_width_stored(self) -> None:
        """msg_width is stored on the formatter."""
        formatter = MultilineFormatter(msg_width=120)
        assert formatter.msg_width == 120


class TestMultilineFormatterMetadataContent:
    """Tests for metadata content in formatted output."""

    def test_metadata_contains_timestamp(self) -> None:
        """Metadata includes a formatted timestamp."""
        formatter = MultilineFormatter(msg_width=20, show_metadata=True)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0, msg="msg", args=(), exc_info=None
        )
        result = formatter.format(record)
        assert "20" in result

    def test_metadata_contains_level_name(self) -> None:
        """Metadata includes the log level name."""
        formatter = MultilineFormatter(msg_width=20, show_metadata=True)
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0, msg="msg", args=(), exc_info=None
        )
        result = formatter.format(record)
        assert "ERROR" in result

    def test_metadata_contains_logger_name(self) -> None:
        """Metadata includes the logger name."""
        formatter = MultilineFormatter(msg_width=20, show_metadata=True)
        record = logging.LogRecord(
            name="my.custom.logger", level=logging.INFO, pathname="", lineno=0, msg="msg", args=(), exc_info=None
        )
        result = formatter.format(record)
        assert "my.custom.logger" in result


class TestSetupLogging:
    """Tests for setup_logging()."""

    def test_adds_file_handler(self, tmp_path: "pytest.TempPathFactory") -> None:
        """setup_logging adds a FileHandler to the root logger."""
        log_file = str(tmp_path / "test.log")
        initial_handler_count = len(logging.root.handlers)
        setup_logging(log_file)
        try:
            assert len(logging.root.handlers) == initial_handler_count + 1
            handler = logging.root.handlers[-1]
            assert isinstance(handler, logging.FileHandler)
        finally:
            logging.root.removeHandler(logging.root.handlers[-1])

    def test_sets_root_level(self, tmp_path: "pytest.TempPathFactory") -> None:
        """setup_logging sets the root logger level."""
        log_file = str(tmp_path / "test.log")
        setup_logging(log_file, level=logging.WARNING)
        try:
            assert logging.root.level == logging.WARNING
        finally:
            logging.root.removeHandler(logging.root.handlers[-1])
            logging.root.setLevel(logging.WARNING)

    def test_default_level_is_debug(self, tmp_path: "pytest.TempPathFactory") -> None:
        """setup_logging defaults to DEBUG level."""
        log_file = str(tmp_path / "test.log")
        setup_logging(log_file)
        try:
            assert logging.root.level == logging.DEBUG
        finally:
            logging.root.removeHandler(logging.root.handlers[-1])

    def test_handler_uses_multiline_formatter(self, tmp_path: "pytest.TempPathFactory") -> None:
        """setup_logging configures the handler with MultilineFormatter."""
        log_file = str(tmp_path / "test.log")
        setup_logging(log_file, msg_width=200, show_metadata=True)
        try:
            handler = logging.root.handlers[-1]
            formatter = handler.formatter
            assert isinstance(formatter, MultilineFormatter)
            assert formatter.msg_width == 200
            assert formatter.show_metadata is True
        finally:
            logging.root.removeHandler(logging.root.handlers[-1])

    def test_writes_to_log_file(self, tmp_path: "pytest.TempPathFactory") -> None:
        """setup_logging writes log messages to the specified file."""
        log_file = str(tmp_path / "test.log")
        setup_logging(log_file, level=logging.INFO, show_metadata=False)
        try:
            logger = logging.getLogger("test_writes")
            logger.info("test message")
            logging.root.handlers[-1].flush()
            with open(log_file) as f:
                content = f.read()
            assert "test message" in content
        finally:
            logging.root.removeHandler(logging.root.handlers[-1])

    def test_file_mode_is_write(self, tmp_path: "pytest.TempPathFactory") -> None:
        """setup_logging opens the log file in write mode, overwriting existing content."""
        log_file = str(tmp_path / "test.log")
        with open(log_file, "w") as f:
            f.write("old content\n")
        setup_logging(log_file, show_metadata=False)
        try:
            logger = logging.getLogger("test_mode")
            logger.info("new content")
            logging.root.handlers[-1].flush()
            with open(log_file) as f:
                content = f.read()
            assert "old content" not in content
            assert "new content" in content
        finally:
            logging.root.removeHandler(logging.root.handlers[-1])
