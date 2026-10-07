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

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

from dashpva.viewer.workbench.managers.roi_manager import ROIManager  # noqa: E402


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


def test_batch_save_replaces_same_named_roi(tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)

    ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2))
    ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Batch ROI", (2, 3, 4, 5))

    with h5py.File(path, "r") as h5f:
        roi = h5f["/entry/data/rois/Batch_ROI"]
        assert roi.shape == (3, 5, 4)
        assert (roi.attrs["x"], roi.attrs["y"]) == (2, 3)


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
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Yes)
    monkeypatch.setattr(
        "dashpva.viewer.workbench.managers.roi_manager.QMessageBox.information",
        lambda *args, **kwargs: None,
    )

    manager._save_roi_to_files(Roi(), [str(other), str(current)])

    for path in (current, other):
        with h5py.File(path, "r") as h5f:
            assert "/entry/data/rois/Batch_ROI" in h5f


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
    monkeypatch.setattr(QMessageBox, "setDetailedText", lambda self, text: details.append(text))
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Cancel)

    manager._save_roi_to_files(Roi(), [str(other)])

    assert str(current) in details[0]
    assert str(other) in details[0]
