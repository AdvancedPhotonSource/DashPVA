from concurrent.futures import ThreadPoolExecutor

import pytest

from dashpva.utils.performance_metrics import BoundedLatencyWindow


def test_latency_window_reports_seconds_and_percentiles():
    window = BoundedLatencyWindow(max_samples=8)
    for value in (0.001, 0.002, 0.003, 0.004):
        window.observe(value)
    result = window.snapshot()
    assert result["samples"] == 4
    assert result["p50_seconds"] == pytest.approx(0.0025)
    assert result["p95_seconds"] == pytest.approx(0.00385)
    assert result["p99_seconds"] == pytest.approx(0.00397)


def test_latency_window_retains_only_configured_sample_count():
    window = BoundedLatencyWindow(max_samples=3)
    for value in range(10):
        window.observe(value)
    result = window.snapshot()
    assert result["samples"] == 3
    assert result["p50_seconds"] == 8.0


def test_latency_window_reset_and_input_validation():
    window = BoundedLatencyWindow()
    window.observe(1.0)
    window.reset()
    assert window.snapshot() == {
        "samples": 0,
        "p50_seconds": None,
        "p95_seconds": None,
        "p99_seconds": None,
    }
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            window.observe(value)


def test_latency_window_accepts_concurrent_observations_with_fixed_memory():
    window = BoundedLatencyWindow(max_samples=64)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(window.observe, (value / 1000 for value in range(1000))))
    result = window.snapshot()
    assert result["samples"] == 64
    assert 0.0 <= result["p50_seconds"] <= 1.0
