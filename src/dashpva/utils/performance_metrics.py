from collections import deque
from threading import Lock

import numpy as np


class BoundedLatencyWindow:
    """Thread-safe recent latency samples with deterministic summaries."""

    def __init__(self, max_samples: int = 1024):
        if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be a positive integer")
        self._samples = deque(maxlen=max_samples)
        self._lock = Lock()

    def observe(self, seconds: float) -> None:
        value = float(seconds)
        if not np.isfinite(value) or value < 0:
            raise ValueError("latency must be finite and non-negative")
        with self._lock:
            self._samples.append(value)

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    def snapshot(self) -> dict[str, float | int | None]:
        with self._lock:
            values = np.asarray(tuple(self._samples), dtype=np.float64)
        if values.size == 0:
            return {"samples": 0, "p50_seconds": None, "p95_seconds": None, "p99_seconds": None}
        p50, p95, p99 = np.percentile(values, (50, 95, 99))
        return {
            "samples": int(values.size),
            "p50_seconds": float(p50),
            "p95_seconds": float(p95),
            "p99_seconds": float(p99),
        }
