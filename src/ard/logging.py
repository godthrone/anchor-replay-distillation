"""ARD logging configuration.

Provides a ``get_logger`` helper for consistent log formatting across all modules.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger with ARD standard configuration.

    Logs to stdout with level INFO and a simple ``LEVEL: message`` format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger