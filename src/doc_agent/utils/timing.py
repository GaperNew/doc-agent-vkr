from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class TimerResult:
    elapsed_ms: float = 0.0


@contextmanager
def measure_time():
    result = TimerResult()
    started_at = time.perf_counter()

    try:
        yield result
    finally:
        finished_at = time.perf_counter()
        result.elapsed_ms = round((finished_at - started_at) * 1000, 3)
