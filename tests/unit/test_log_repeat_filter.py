"""LogManager repeat suppression is opt-in per logger."""

import logging

from dashpva.utils import log_manager


def _manager(monkeypatch, tmp_path):
    monkeypatch.setattr(log_manager.settings, "LOG_PATH", str(tmp_path))
    monkeypatch.setattr(log_manager.sys, "excepthook", log_manager.sys.excepthook)
    return log_manager.LogManager()


def _records(logger):
    seen = []
    handler = logging.Handler()
    handler.emit = seen.append
    logger.addHandler(handler)
    return seen


def test_repeat_window_writes_a_repeated_message_once(monkeypatch, tmp_path):
    logger = _manager(monkeypatch, tmp_path).get_logger("repeat.on", drop_repeats=True)
    seen = _records(logger)
    for _ in range(5):
        logger.error("[HKL] setup failed: missing Energy:Value")
    logger.error("[HKL] setup failed: missing UB")
    assert [r.getMessage() for r in seen] == [
        "[HKL] setup failed: missing Energy:Value",
        "[HKL] setup failed: missing UB",
    ]


def test_repeat_is_written_again_after_the_window(monkeypatch, tmp_path):
    logger = _manager(monkeypatch, tmp_path).get_logger("repeat.window", drop_repeats=True)
    seen = _records(logger)
    times = iter([1000.0, 1030.0, 1061.0])
    monkeypatch.setattr(logging.time, "time", lambda: next(times))
    for _ in range(3):
        logger.error("same")
    assert len(seen) == 2


def test_loggers_without_a_window_write_every_message(monkeypatch, tmp_path):
    logger = _manager(monkeypatch, tmp_path).get_logger("repeat.off")
    seen = _records(logger)
    for _ in range(3):
        logger.error("same")
    assert len(seen) == 3
