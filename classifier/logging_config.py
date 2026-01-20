"""Logging configuration for the ticket classifier."""

import logging
import sys
from pathlib import Path


def setup_logging(
    output_dir: str | Path = "output",
    verbose: bool = False,
    log_filename: str = "run.log",
) -> logging.Logger:
    """
    Configure logging for both terminal and file output.

    Args:
        output_dir: Directory for log file output
        verbose: If True, terminal shows DEBUG level; otherwise INFO
        log_filename: Name of the log file

    Returns:
        Root logger instance
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create root logger
    logger = logging.getLogger("classifier")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Terminal handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(
        output_path / log_filename,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"classifier.{name}")
