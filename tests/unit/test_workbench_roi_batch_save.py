# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Batch-saving a Workbench ROI applies one geometry to compatible files."""

import sys
import types

import h5py
import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

sys.modules.setdefault("qtawesome", types.SimpleNamespace(icon=lambda *args, **kwargs: None))

from PyQt5.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from dashpva.viewer.workbench.managers.roi_manager import ROIManager  # noqa: E402
from dashpva.viewer.workbench.workbench import WorkbenchWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _image_file(path, shape=(3, 12, 16), dataset_path="/entry/data/data"):
    with h5py.File(path, "w") as h5f:
        group_path, dataset_name = dataset_path.rsplit("/", 1)
        group = h5f.require_group(group_path)
        group.create_dataset(dataset_name, data=np.arange(np.prod(shape)).reshape(shape))


def test_batch_save_writes_cropped_stack_and_geometry(tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)

    saved_path = ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (4, 3, 5, 4))

    assert saved_path == "/entry/data/rois/Batch_ROI"
    with h5py.File(path, "r") as h5f:
        source = h5f["/entry/data/data"]
        roi = h5f[saved_path]
        np.testing.assert_array_equal(roi[...], source[:, 3:7, 4:9])
        assert (roi.attrs["x"], roi.attrs["y"]) == (4, 3)
        assert (roi.attrs["w"], roi.attrs["h"]) == (5, 4)
        assert roi.attrs["source_path"] == "/entry/data/data"


def test_batch_save_suffixes_same_named_roi(tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)

    ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2))
    second_path = ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (2, 3, 4, 5))

    assert second_path == "/entry/data/rois/Batch_ROI(2)"
    with h5py.File(path, "r") as h5f:
        first_roi = h5f["/entry/data/rois/Batch_ROI"]
        second_roi = h5f[second_path]
        assert first_roi.shape == (3, 2, 2)
        assert second_roi.shape == (3, 5, 4)
        assert (second_roi.attrs["x"], second_roi.attrs["y"]) == (2, 3)


def test_batch_update_reuses_linked_dataset(tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)
    batch_files = [str(path)]

    saved_path = ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2),
        batch_id="batch-1", batch_files=batch_files)
    updated_path = ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (2, 3, 4, 5),
        batch_id="batch-1", batch_files=batch_files, update_batch=True)

    assert updated_path == saved_path
    with h5py.File(path, "r") as h5f:
        rois = h5f["/entry/data/rois"]
        datasets = [name for name, item in rois.items() if isinstance(item, h5py.Dataset)]
        assert datasets == ["Batch_ROI"]
        roi = rois["Batch_ROI"]
        assert roi.shape == (3, 5, 4)
        assert roi.attrs["batch_id"] == "batch-1"
        assert roi.attrs["batch_name"] == "Batch ROI"


def test_delete_batch_roi_removes_current_copy_from_batch(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    for path in (first, second):
        _image_file(path)
        ROIManager._save_roi_geometry_to_file(
            str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2),
            batch_id="batch-1", batch_files=[str(first), str(second)])

    class Main:
        def update_status(self, *args, **kwargs):
            pass

    roi = object()
    manager = ROIManager(Main())
    manager.roi_source_by_id[id(roi)] = {
        "batch_id": "batch-1",
        "batch_files": [str(first), str(second)],
        "file_path": str(first),
    }

    manager.delete_roi_from_disk(roi)

    with h5py.File(first, "r") as h5f:
        assert "/entry/data/rois/Batch_ROI" not in h5f
    with h5py.File(second, "r") as h5f:
        roi_dataset = h5f["/entry/data/rois/Batch_ROI"]
        assert roi_dataset.attrs["batch_files"] == f'["{second}"]'


def test_batch_manifest_keeps_only_successful_files(tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)
    ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2),
        batch_id="batch-1", batch_files=[str(path), "missing.h5"])

    ROIManager._update_batch_file_manifest(str(path), "batch-1", [str(path)])

    with h5py.File(path, "r") as h5f:
        manifest = h5f["/entry/data/rois/Batch_ROI"].attrs["batch_files"]
        assert manifest == f'["{path}"]'


def test_detach_keeps_roi_but_removes_batch_metadata(tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)
    ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2),
        batch_id="batch-1", batch_files=[str(path)], batch_color="#123456")

    remaining = ROIManager._detach_batch_file(str(path), "batch-1", [str(path)])

    assert remaining == []
    with h5py.File(path, "r") as h5f:
        roi = h5f["/entry/data/rois/Batch_ROI"]
        assert "batch_id" not in roi.attrs
        assert "batch_files" not in roi.attrs
        assert "batch_color" not in roi.attrs


def test_batch_color_is_shared_through_hdf5_metadata(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    for path in (first, second):
        _image_file(path)
        ROIManager._save_roi_geometry_to_file(
            str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2),
            batch_id="batch-1", batch_files=[str(first), str(second)],
            batch_color="#112233")
        ROIManager._set_batch_color(str(path), "batch-1", "#abcdef")

    for path in (first, second):
        with h5py.File(path, "r") as h5f:
            assert h5f["/entry/data/rois/Batch_ROI"].attrs["batch_color"] == "#abcdef"


def test_roi_dock_text_marks_batch_members():
    class Point:
        def __init__(self, x, y):
            self._x = x
            self._y = y

        def x(self):
            return self._x

        def y(self):
            return self._y

    class Roi:
        def pos(self):
            return Point(1, 2)

        def size(self):
            return Point(3, 4)

    roi = Roi()
    manager = ROIManager(object())
    manager.roi_names[id(roi)] = "Linked ROI"
    manager.roi_source_by_id[id(roi)] = {"batch_id": "batch-1"}

    assert manager.format_roi_text(roi) == "[Batch] Linked ROI: x=1, y=2, w=3, h=4"


def test_save_roi_does_not_write_batch_until_save_batch_is_used(monkeypatch):
    statuses = []

    class Main:
        def update_status(self, message, **kwargs):
            statuses.append((message, kwargs))

    roi = object()
    manager = ROIManager(Main())
    manager.roi_source_by_id[id(roi)] = {
        "batch_id": "batch-1",
        "batch_files": ["first.h5", "second.h5"],
    }
    updates = []
    monkeypatch.setattr(manager, "_update_batch_roi", lambda value: updates.append(value))

    manager.save_roi(roi)

    assert updates == []
    assert "use Save Batch" in statuses[0][0]


@pytest.mark.parametrize("is_batch", [False, True])
def test_workbench_ctrl_s_saves_active_roi_or_batch(is_batch):
    roi = object()
    calls = []

    class Manager:
        current_roi = roi

        def get_roi_source(self, value):
            assert value is roi
            return {"batch_id": "batch-1"} if is_batch else {}

        def save_roi(self, value):
            calls.append(("roi", value))

        def save_batch_roi(self, value):
            calls.append(("batch", value))

    class Window:
        roi_manager = Manager()
        current_roi = None

        def update_status(self, *args, **kwargs):
            pass

    WorkbenchWindow.handle_save_shortcut(Window())

    assert calls == [("batch" if is_batch else "roi", roi)]


def test_batch_save_rejects_missing_or_too_small_source(tmp_path):
    missing = tmp_path / "missing.h5"
    with h5py.File(missing, "w"):
        pass
    with pytest.raises(KeyError, match="was not found"):
        ROIManager._save_roi_geometry_to_file(
            str(missing), "/entry/data/data", "ROI", (1, 1, 2, 2))

    small = tmp_path / "small.h5"
    _image_file(small, shape=(2, 4, 4))
    with pytest.raises(ValueError, match="exceeds image size"):
        ROIManager._save_roi_geometry_to_file(
            str(small), "/entry/data/data", "ROI", (3, 3, 2, 2))


def test_folder_selection_finds_only_direct_hdf5_files(tmp_path):
    (tmp_path / "b.hdf5").touch()
    (tmp_path / "a.H5").touch()
    (tmp_path / "notes.txt").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "ignored.h5").touch()

    files = ROIManager._hdf5_files_in_folder(str(tmp_path))

    assert files == [str(tmp_path / "a.H5"), str(tmp_path / "b.hdf5")]


def test_save_to_always_includes_current_file_once(qapp, monkeypatch, tmp_path):
    current = tmp_path / "current.h5"
    other = tmp_path / "other.h5"
    _image_file(current)
    _image_file(other)

    class Point:
        def __init__(self, x, y):
            self._x = x
            self._y = y

        def x(self):
            return self._x

        def y(self):
            return self._y

    class Roi:
        def pos(self):
            return Point(1, 2)

        def size(self):
            return Point(3, 4)

    class Main:
        current_file_path = str(current)
        selected_dataset_path = "/entry/data/data"

        def update_status(self, *args, **kwargs):
            pass

    manager = ROIManager(Main())
    monkeypatch.setattr(manager, "get_roi_name", lambda roi: "Batch ROI")
    monkeypatch.setattr(
        QInputDialog, "getText", lambda *args, **kwargs: ("Exported ROI", True))
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Yes)
    monkeypatch.setattr(
        "dashpva.viewer.workbench.managers.roi_manager.QMessageBox.information",
        lambda *args, **kwargs: None,
    )

    manager._save_roi_to_files(Roi(), [str(other), str(current)])

    for path in (current, other):
        with h5py.File(path, "r") as h5f:
            assert "/entry/data/rois/Exported_ROI" in h5f


def test_confirmation_details_list_every_destination(qapp, monkeypatch, tmp_path):
    current = tmp_path / "current.h5"
    other = tmp_path / "other.h5"
    _image_file(current)
    _image_file(other)

    class Point:
        def x(self):
            return 1

        def y(self):
            return 2

    class Roi:
        pos = size = lambda self: Point()

    class Main:
        current_file_path = str(current)
        selected_dataset_path = "/entry/data/data"

        def update_status(self, *args, **kwargs):
            pass

    details = []
    manager = ROIManager(Main())
    monkeypatch.setattr(manager, "get_roi_name", lambda roi: "ROI")
    prompted_names = []
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *args, **kwargs: (prompted_names.append(args[-1]) or args[-1], True),
    )
    monkeypatch.setattr(QMessageBox, "setDetailedText", lambda self, text: details.append(text))
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Cancel)

    manager._save_roi_to_files(Roi(), [str(other)])

    assert str(current) in details[0]
    assert str(other) in details[0]
    assert prompted_names == ["ROI (current)"]
