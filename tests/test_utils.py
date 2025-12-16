"""Unit tests for llm_behavior_adaptation.utils module."""

import logging
import sys

from llm_behavior_adaptation.utils import ColorfulFormatter, register_logger


class TestColorfulFormatter:
    """Test ColorfulFormatter class."""

    def test_formatter_initialization(self):
        """Test that ColorfulFormatter initializes correctly."""
        formatter = ColorfulFormatter()
        assert formatter is not None
        assert hasattr(formatter, "FORMATS")
        assert len(formatter.FORMATS) == 5

    def test_format_debug_message(self):
        """Test formatting a DEBUG level message."""
        formatter = ColorfulFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=10,
            msg="Debug message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"
        record.module = "test_module"
        record.process = 12345

        formatted = formatter.format(record)
        assert "Debug message" in formatted
        assert "DEBUG" in formatted
        assert "test.py:10" in formatted

    def test_format_info_message(self):
        """Test formatting an INFO level message."""
        formatter = ColorfulFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=20,
            msg="Info message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"
        record.module = "test_module"
        record.process = 12345

        formatted = formatter.format(record)
        assert "Info message" in formatted
        assert "INFO" in formatted

    def test_format_warning_message(self):
        """Test formatting a WARNING level message."""
        formatter = ColorfulFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=30,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"
        record.module = "test_module"
        record.process = 12345

        formatted = formatter.format(record)
        assert "Warning message" in formatted
        assert "WARNING" in formatted

    def test_format_error_message(self):
        """Test formatting an ERROR level message."""
        formatter = ColorfulFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=40,
            msg="Error message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"
        record.module = "test_module"
        record.process = 12345

        formatted = formatter.format(record)
        assert "Error message" in formatted
        assert "ERROR" in formatted

    def test_format_critical_message(self):
        """Test formatting a CRITICAL level message."""
        formatter = ColorfulFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.CRITICAL,
            pathname="test.py",
            lineno=50,
            msg="Critical message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_func"
        record.module = "test_module"
        record.process = 12345

        formatted = formatter.format(record)
        assert "Critical message" in formatted
        assert "CRITICAL" in formatted


class TestRegisterLogger:
    """Test register_logger function."""

    def test_register_logger_with_no_logger(self):
        """Test registering logger without passing a logger object."""
        # Create a fresh logger for testing
        test_logger = logging.getLogger("test_logger_1")
        # Clear any existing handlers and ensure propagate is False
        test_logger.handlers.clear()
        test_logger.propagate = False

        register_logger(test_logger)

        assert len(test_logger.handlers) >= 1
        assert any(isinstance(h, logging.StreamHandler) for h in test_logger.handlers)
        assert any(isinstance(h.formatter, ColorfulFormatter) for h in test_logger.handlers)
        assert test_logger.level == logging.DEBUG

        # Cleanup
        test_logger.handlers.clear()

    def test_register_logger_with_existing_logger(self):
        """Test registering an existing logger."""
        test_logger = logging.getLogger("test_logger_2")
        test_logger.handlers.clear()
        test_logger.propagate = False

        register_logger(test_logger)

        assert len(test_logger.handlers) >= 1
        assert test_logger.level == logging.DEBUG

        # Cleanup
        test_logger.handlers.clear()

    def test_register_logger_does_not_duplicate_handlers(self):
        """Test that register_logger doesn't add duplicate handlers."""
        test_logger = logging.getLogger("test_logger_3")
        test_logger.handlers.clear()
        test_logger.propagate = False

        # Register once
        register_logger(test_logger)
        initial_handler_count = len(test_logger.handlers)

        # Try to register again (should not add more handlers)
        register_logger(test_logger)
        final_handler_count = len(test_logger.handlers)

        # Should not add duplicate handlers
        assert initial_handler_count == final_handler_count
        assert initial_handler_count >= 1

        # Cleanup
        test_logger.handlers.clear()

    def test_register_logger_uses_stderr(self):
        """Test that register_logger uses sys.stderr."""
        test_logger = logging.getLogger("test_logger_4")
        test_logger.handlers.clear()
        test_logger.propagate = False

        register_logger(test_logger)

        assert len(test_logger.handlers) >= 1
        stream_handlers = [h for h in test_logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1
        assert stream_handlers[0].stream == sys.stderr

        # Cleanup
        test_logger.handlers.clear()

    def test_register_logger_default_logger(self):
        """Test register_logger with None (uses root logger)."""
        # Save original handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level

        # Clear handlers for testing
        root_logger.handlers.clear()

        try:
            register_logger(None)

            assert len(root_logger.handlers) >= 1
            # Check that at least one handler is a StreamHandler with ColorfulFormatter
            has_colorful_handler = any(
                isinstance(h, logging.StreamHandler) and isinstance(h.formatter, ColorfulFormatter)
                for h in root_logger.handlers
            )
            assert has_colorful_handler
            assert root_logger.level == logging.DEBUG

        finally:
            # Restore original state
            root_logger.handlers = original_handlers
            root_logger.level = original_level
