# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
pytest.importorskip("pyqtgraph")
pytest.importorskip("qtawesome")
MaskViewerWindow = importlib.import_module(
    "dashpva.viewer.mask_viewer"
).MaskViewerWindow


@pytest.fixture(scope="module")
def app():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


def _parent(tmp_path, baseline=None):
    parent = QtWidgets.QWidget()
    manager = SimpleNamespace(
        mask=None if baseline is None else baseline.copy(),
        mask_path=None,
        masks_dir=str(tmp_path),
        DEFAULT_MASK_FILENAME="active_mask.npy",
    )
    parent.mask_manager = manager
    parent.image_is_transposed = False
    parent.rot_num = 0
    parent.reader = None
    return parent


def test_save_is_the_only_operation_that_publishes(app, tmp_path):
    baseline = np.zeros((4, 4), dtype=bool)
    parent = _parent(tmp_path, baseline)
    window = MaskViewerWindow(baseline, parent=parent)
    published = []
    window.mask_updated.connect(lambda mask: published.append(mask.copy()))
    window.mask[1, 1] = True
    assert published == []

    assert window._save_mask()
    assert len(published) == 1
    assert published[0][1, 1]
    assert np.load(tmp_path / "active_mask.npy")[1, 1]
    window.deleteLater()


def test_save_failure_keeps_original_baseline_and_emits_nothing(
    app, tmp_path, monkeypatch
):
    baseline = np.zeros((4, 4), dtype=bool)
    window = MaskViewerWindow(baseline, parent=_parent(tmp_path, baseline))
    published = []
    window.mask_updated.connect(lambda mask: published.append(mask.copy()))
    window.mask[2, 2] = True
    monkeypatch.setattr(np, "save", lambda *_: (_ for _ in ()).throw(OSError("full")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: None)

    assert not window._save_mask()
    assert published == []
    assert not window._original_mask.any()
    window.deleteLater()


def test_save_failure_during_close_keeps_editor_open(app, tmp_path, monkeypatch):
    baseline = np.zeros((4, 4), dtype=bool)
    window = MaskViewerWindow(baseline, parent=_parent(tmp_path, baseline))
    window.mask[2, 2] = True
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.Save,
    )
    monkeypatch.setattr(window, "_save_mask", lambda: False)
    window.show()
    app.processEvents()

    assert not window.close()
    assert window.isVisible()
    window.hide()
    window.deleteLater()


@pytest.mark.parametrize(
    "answer, closes, saves",
    [
        (QtWidgets.QMessageBox.Cancel, False, False),
        (QtWidgets.QMessageBox.Discard, True, False),
        (QtWidgets.QMessageBox.Save, True, True),
    ],
)
def test_close_save_discard_cancel(app, tmp_path, monkeypatch, answer, closes, saves):
    baseline = np.zeros((4, 4), dtype=bool)
    window = MaskViewerWindow(baseline, parent=_parent(tmp_path, baseline))
    window.mask[0, 0] = True
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: answer)
    saved = []
    monkeypatch.setattr(window, "_save_mask", lambda: saved.append(True) or True)
    window.show()
    app.processEvents()

    assert window.close() is closes
    assert bool(saved) is saves
    if answer == QtWidgets.QMessageBox.Discard:
        assert np.array_equal(window.mask, baseline)
    window.deleteLater()


def test_discard_preserves_no_mask_and_no_file(app, tmp_path, monkeypatch):
    window = MaskViewerWindow(np.zeros((4, 4), dtype=bool), parent=_parent(tmp_path))
    window.mask[0, 0] = True
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.Discard,
    )
    window.show()
    app.processEvents()

    assert window.close()
    assert window._original_path is None
    assert not (tmp_path / "active_mask.npy").exists()
    window.deleteLater()


def test_discard_leaves_existing_file_unchanged(app, tmp_path, monkeypatch):
    baseline = np.zeros((4, 4), dtype=bool)
    path = tmp_path / "existing.npy"
    np.save(path, baseline)
    window = MaskViewerWindow(baseline, mask_path=str(path), parent=_parent(tmp_path))
    window.mask[0, 0] = True
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args: QtWidgets.QMessageBox.Discard,
    )
    window.show()
    app.processEvents()

    assert window.close()
    np.testing.assert_array_equal(np.load(path), baseline)
    window.deleteLater()


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
