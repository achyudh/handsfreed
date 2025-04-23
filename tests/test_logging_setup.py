"""Tests for logging setup."""

import logging
import pytest

from handsfreed.logging_setup import setup_logging


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging configuration after each test."""
    yield
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)


def test_setup_logging_console_only(caplog, capsys):
    """Test basic logging setup without file output."""
    caplog.set_level(logging.DEBUG)
    setup_logging("DEBUG")
    root = logging.getLogger()

    # Check level is set
    assert root.level == logging.DEBUG

    # Should have one handler (stderr)
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)

    # Test logging works (check stderr output)
    logger = logging.getLogger("test")
    test_msg = "Test message"
    logger.info(test_msg)

    # Get stderr output
    captured = capsys.readouterr()
    assert test_msg in captured.err


def test_setup_logging_with_file(tmp_path):
    """Test logging setup with file output."""
    log_file = tmp_path / "test.log"
    setup_logging("INFO", log_file)
    root = logging.getLogger()

    # Should have two handlers (stderr and file)
    assert len(root.handlers) == 2
    assert any(isinstance(h, logging.FileHandler) for h in root.handlers)

    # Test file is created
    assert log_file.exists()

    # Test logging to file
    logger = logging.getLogger("test")
    test_msg = "Test error message"
    logger.error(test_msg)

    log_content = log_file.read_text()
    assert test_msg in log_content


def test_setup_logging_invalid_file(tmp_path):
    """Test logging setup with invalid file path."""
    # Create a directory where the log file should be
    invalid_dir = tmp_path / "test.log"
    invalid_dir.mkdir()

    # Try to use it as a log file (should fail)
    setup_logging("INFO", invalid_dir)
    root = logging.getLogger()

    # Should only have stderr handler as file handler should fail
    handlers = root.handlers
    assert len(handlers) == 1
    assert all(isinstance(h, logging.StreamHandler) for h in handlers)
    assert not any(isinstance(h, logging.FileHandler) for h in handlers)


def test_setup_logging_invalid_level():
    """Test logging setup with invalid level falls back to INFO."""
    setup_logging("INVALID")
    root = logging.getLogger()
    assert root.level == logging.INFO


def test_setup_logging_removes_existing_handlers(caplog):
    """Test that existing handlers are removed."""
    # Add a test handler
    root = logging.getLogger()
    test_handler = logging.StreamHandler()
    root.addHandler(test_handler)
    orig_handlers = len(root.handlers)

    # Setup logging
    setup_logging("INFO")

    # Should only have one handler (the new one)
    assert len(root.handlers) == 1
    assert test_handler not in root.handlers
    assert len(root.handlers) < orig_handlers
