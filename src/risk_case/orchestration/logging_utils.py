from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from risk_case.settings import ensure_dir


def setup_run_logger(run_dir: Path, level: str = "INFO") -> logging.Logger:
    ensure_dir(run_dir)
    logger = logging.getLogger("risk_case.pipeline")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def close_run_logger(logger: logging.Logger) -> None:
    handlers = list(logger.handlers)
    for handler in handlers:
        try:
            handler.flush()
            handler.close()
        finally:
            logger.removeHandler(handler)


@contextmanager
def log_stage(logger: logging.Logger, stage: str):
    started = time.perf_counter()
    logger.info("START: %s", stage)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        logger.info("END: %s (%.2fs)", stage, elapsed)
