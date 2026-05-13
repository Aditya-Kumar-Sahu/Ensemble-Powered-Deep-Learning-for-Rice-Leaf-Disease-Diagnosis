"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "rice_leaf_disease",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create and return a logger configured with a console handler and an optional file handler.

    Parameters:
        name (str): Logger name to create or retrieve.
        level (int): Logging level applied to the logger and its handlers.
        log_file (Optional[str]): Path to a file to write logs to; if provided, the file's
            parent directory will be created if it does not exist.

    Returns:
        logging.Logger: The configured logger instance with a StreamHandler writing to
            stdout and, when requested, a FileHandler writing to `log_file`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
