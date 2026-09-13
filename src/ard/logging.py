"""ARD logging configuration.

Provides a ``get_logger`` helper for consistent log formatting across all
modules, and ``configure_file_logging`` for persisting logs to disk in the
output directory (called explicitly after the output directory is created).
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_FILE_LOGGING_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger with ARD standard configuration.

    Logs to stdout with level INFO and a simple ``LEVEL: message`` format.
    The StreamHandler level is explicitly set to INFO so that later changes
    to the logger's own level (e.g. by :func:`configure_file_logging`) do
    not accidentally promote DEBUG messages to stdout (§2.2).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)  # explicit — §2.2, Y1 fix
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def configure_file_logging(run_dir: Path) -> None:
    """Add file-based log handlers writing into *run_dir*/logs/.

    Must be called **after** the output directory has been created
    (i.e. after ``output_dir.mkdir()`` in the pipeline).  This is an
    explicit, no-magic entry point — it does not guess paths from
    environment variables or module-level globals (§2.2).

    **Idempotent:** repeated calls are no-ops (§2.1 契约即防呆, Y2 fix).

    Three log files are created per §13.3 (one log domain = one pipeline):

    * ``ard.log`` — human-readable (INFO+, timestamped)
    * ``ard_debug.log`` — machine-parseable (DEBUG+, with
      millisecond timestamps and source-location info)
    * ``ard_error.log`` — ERROR+ only (§13.3 common error log)

    All files use :class:`~logging.handlers.RotatingFileHandler`
    (10 MB × 3 backups) to prevent unbounded growth.

    Handlers are attached to the ``ard`` namespace logger (not root),
    so third-party library logs (httpx, urllib3, etc.) are excluded
    from the project's log files (W1 fix).

    **Existing stdout behaviour is preserved:** the ``StreamHandler``
    created by :func:`get_logger` has an explicit INFO level and is
    never modified.  Child logger levels are raised to DEBUG to enable
    propagation to the file handlers, but the INFO-level StreamHandler
    still suppresses DEBUG on stdout.
    """
    global _FILE_LOGGING_CONFIGURED
    if _FILE_LOGGING_CONFIGURED:
        return
    _FILE_LOGGING_CONFIGURED = True

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Attach handlers to the ``ard`` namespace logger, NOT root.
    # This keeps third-party logs (httpx, urllib3) out of our files.
    ard_logger = logging.getLogger("ard")
    ard_logger.setLevel(logging.DEBUG)

    # ── Human-readable log (INFO) ──────────────────────────────────────
    info_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "ard.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    ard_logger.addHandler(info_handler)

    # ── Machine-parseable log (DEBUG) ──────────────────────────────────
    debug_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "ard_debug.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    ard_logger.addHandler(debug_handler)

    # ── Common error log (ERROR+) — §13.3 ──────────────────────────────
    error_handler = logging.handlers.RotatingFileHandler(
        logs_dir / "ard_error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    ard_logger.addHandler(error_handler)

    # Allow DEBUG messages from ard.* child loggers to propagate to the
    # ``ard`` logger's file handlers.  StreamHandler levels are untouched
    # (explicitly INFO via get_logger), so stdout remains INFO+ only.
    #
    # Note: loggers NOT created via get_logger (e.g. ard.pipeline,
    # ard.backends.api_client) have level NOTSET and inherit DEBUG from
    # the ``ard`` parent — no explicit level change needed.
    for name in logging.root.manager.loggerDict:
        if name.startswith("ard") and name != "ard":
            logging.getLogger(name).setLevel(logging.DEBUG)