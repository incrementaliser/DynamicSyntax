"""Logging setup for induction runs (loguru sinks + stdlib intercept)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger

from dylan.induction.pipeline.config import LoggingConfig


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit one stdlib record through the loguru logger."""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_induction_logging(
    config: LoggingConfig,
    *,
    run_dir: Path | None = None,
) -> list[int]:
    """Configure loguru sinks from *config*; return handler ids for later removal.

    Also installs an intercept so stdlib ``dylan.*`` loggers use the same sinks.
    """
    logger.remove()
    handler_ids: list[int] = []
    level = config.level.upper()
    fmt = "<level>{level}</level> | {name}: {message}"

    if config.to_cli:
        handler_ids.append(logger.add(sys.stderr, level=level, format=fmt))

    if config.to_file:
        if run_dir is None:
            raise ValueError("run_dir is required when logging.to_file is true")
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / config.file_name
        handler_ids.append(
            logger.add(
                str(log_path),
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}: {message}",
                encoding="utf-8",
            ),
        )

    std_level = getattr(logging, level, logging.INFO)
    logging.root.handlers = [_InterceptHandler()]
    logging.root.setLevel(std_level)
    logging.getLogger("dylan").setLevel(std_level)
    logging.getLogger("dylan").propagate = True
    return handler_ids


# Backward-compatible alias
configure_experiment_logging = configure_induction_logging
