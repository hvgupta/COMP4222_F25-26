# ...existing code...
import logging
import sys
from typing import Optional

def get_logger(name: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a logger configured to log to the terminal (stdout).
    Safe to call multiple times; it avoids adding duplicate handlers.
    """
    logger_name = name if name is not None else __name__
    logger = logging.getLogger(logger_name)

    # If handlers already attached, just set level and return
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False  # prevent double logging if root logger configured elsewhere
    return logger

# Convenience module-level logger
logger = get_logger(__name__, level=logging.DEBUG)

