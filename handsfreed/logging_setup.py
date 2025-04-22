"""Logging configuration for handsfreed daemon."""

import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure logging with console and optional file output.

    Args:
        log_level: The logging level to use (DEBUG, INFO, etc.)
        log_file: Optional path to a log file.
    """
    # Convert level string to actual level
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Remove any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Common format for all handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler if path provided
    if log_file:
        try:
            # Ensure parent directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
            root.info(f"Logging to file: {log_file}")
        except Exception as e:
            root.error(f"Failed to setup file logging to {log_file}: {e}")

    # Set the level after handlers are configured
    root.setLevel(level)
    root.info(f"Logging initialized at level {log_level}")
