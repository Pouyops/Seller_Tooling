"""Structured (JSON-lines) logging on the stdlib, with per-request/job context.

    from st_common.logging import setup_logging, log_context, get_logger
    setup_logging("inference-api")
    with log_context(job_id=job_id):
        get_logger(__name__).info("job.done", extra={"latency_ms": 41.2})
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
import time
from typing import Any

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("st_log_context", default={})

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "service": self.service,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(_context.get())
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(service: str, level: str = "INFO", json_output: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JsonFormatter(service))
    else:
        handler.setFormatter(logging.Formatter(f"%(asctime)s %(levelname)s {service} %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "urllib3", "aiogram.event", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextlib.contextmanager
def log_context(**fields: Any):
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def bind_context(**fields: Any) -> None:
    """Bind fields for the rest of the current task (e.g. inside a request middleware)."""
    _context.set({**_context.get(), **fields})


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
