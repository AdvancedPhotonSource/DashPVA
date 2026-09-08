# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
pytest.importorskip("pyqtgraph")
pytest.importorskip("qtawesome")
mask_viewer = importlib.import_module("dashpva.viewer.mask_viewer")
MaskViewerWindow = mask_viewer.MaskViewerWindow


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value.copy())


class _Event:
    def __init__(self):
        self.accepted = None

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def _parent(tmp_path, baseline=None):
    manager = SimpleNamespace(
        mask=None if baseline is None else baseline.copy(),
        mask_path=None,
        masks_dir=str(tmp_path),
        DEFAULT_MASK_FILENAME="active_mask.npy",
    )
    return SimpleNamespace(mask_manager=manager)


def _window(tmp_path, baseline=None, mask_path=None):
    mask = np.zeros((4, 4), dtype=bool) if baseline is None else baseline.copy()
    return SimpleNamespace(
        mask=mask,
        _original_mask=mask.copy(),
        mask_path=mask_path,
        _original_path=mask_path,
        parent_viewer=_parent(tmp_path, baseline),
        mask_updated=_Signal(),
        lbl_info=SimpleNamespace(setText=lambda *_: None),
        setWindowTitle=lambda *_: None,
        _info_text=lambda: "mask info",
        _pause_live_plotting=lambda *_: None,
    )


def test_save_is_the_only_operation_that_publishes(tmp_path):
    baseline = np.zeros((4, 4), dtype=bool)
    window = _window(tmp_path, baseline)
    window.mask[1, 1] = True
    assert window.mask_updated.values == []

    assert MaskViewerWindow._save_mask(window)
    assert len(window.mask_updated.values) == 1
    assert window.mask_updated.values[0][1, 1]
    assert np.load(tmp_path / "active_mask.npy")[1, 1]


def test_save_failure_keeps_original_baseline_and_emits_nothing(
    tmp_path, monkeypatch
):
    baseline = np.zeros((4, 4), dtype=bool)
    window = _window(tmp_path, baseline)
    window.mask[2, 2] = True
    monkeypatch.setattr(np, "save", lambda *_: (_ for _ in ()).throw(OSError("full")))
    monkeypatch.setattr(mask_viewer.QMessageBox, "critical", lambda *args: None)

    assert not MaskViewerWindow._save_mask(window)
    assert window.mask_updated.values == []
    assert not window._original_mask.any()


def test_save_failure_during_close_keeps_editor_open(tmp_path, monkeypatch):
    baseline = np.zeros((4, 4), dtype=bool)
    window = _window(tmp_path, baseline)
    window.mask[2, 2] = True
    monkeypatch.setattr(
        mask_viewer.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.Save,
    )
    window._save_mask = lambda: False
    event = _Event()

    MaskViewerWindow.closeEvent(window, event)
    assert event.accepted is False


@pytest.mark.parametrize(
    "answer, closes, saves",
    [
        (QtWidgets.QMessageBox.Cancel, False, False),
        (QtWidgets.QMessageBox.Discard, True, False),
        (QtWidgets.QMessageBox.Save, True, True),
    ],
)
def test_close_save_discard_cancel(tmp_path, monkeypatch, answer, closes, saves):
    baseline = np.zeros((4, 4), dtype=bool)
    window = _window(tmp_path, baseline)
    window.mask[0, 0] = True
    monkeypatch.setattr(mask_viewer.QMessageBox, "question", lambda *args: answer)
    saved = []
    window._save_mask = lambda: saved.append(True) or True
    event = _Event()

    MaskViewerWindow.closeEvent(window, event)
    assert event.accepted is closes
    assert bool(saved) is saves
    if answer == QtWidgets.QMessageBox.Discard:
        assert np.array_equal(window.mask, baseline)


def test_discard_preserves_no_mask_and_no_file(tmp_path, monkeypatch):
    window = _window(tmp_path)
    window.mask[0, 0] = True
    monkeypatch.setattr(
        mask_viewer.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.Discard,
    )
    event = _Event()

    MaskViewerWindow.closeEvent(window, event)
    assert event.accepted
    assert window._original_path is None
    assert not (tmp_path / "active_mask.npy").exists()


def test_discard_leaves_existing_file_unchanged(tmp_path, monkeypatch):
    baseline = np.zeros((4, 4), dtype=bool)
    path = tmp_path / "existing.npy"
    np.save(path, baseline)
    window = _window(tmp_path, baseline, str(path))
    window.mask[0, 0] = True
    monkeypatch.setattr(
        mask_viewer.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.Discard,
    )
    event = _Event()

    MaskViewerWindow.closeEvent(window, event)
    assert event.accepted
    np.testing.assert_array_equal(np.load(path), baseline)


def test_reopen_stops_when_existing_editor_rejects_close(monkeypatch):
    from dashpva.viewer.area_det.area_det_viewer import DiffractionImageWindow

    existing = SimpleNamespace(close=lambda: False)
    parent = SimpleNamespace(mask_viewer=existing)
    created = []
    monkeypatch.setattr(
        "dashpva.viewer.area_det.area_det_viewer.MaskViewerWindow",
        lambda *args, **kwargs: created.append(True),
    )

    DiffractionImageWindow._open_mask_viewer(
        parent, mask=np.zeros((2, 2), dtype=bool)
    )

    assert parent.mask_viewer is existing
    assert created == []


def test_parent_close_cancellation_happens_before_cleanup():
    from dashpva.viewer.area_det.area_det_viewer import DiffractionImageWindow

    calls = []
    event = SimpleNamespace(ignore=lambda: calls.append("ignored"))
    parent = SimpleNamespace(
        mask_viewer=SimpleNamespace(close=lambda: False),
        saveState=lambda *_: calls.append("saved"),
    )

    DiffractionImageWindow.closeEvent(parent, event)

    assert calls == ["ignored"]
