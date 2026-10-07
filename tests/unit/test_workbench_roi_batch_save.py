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

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication,
    QInputDialog,
    QListWidget,
    QMessageBox,
)

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
        np.testing.assert_array_equal(roi[...], source[:, 4:9, 3:7])
        assert (roi.attrs["x"], roi.attrs["y"]) == (4, 3)
        assert (roi.attrs["w"], roi.attrs["h"]) == (5, 4)
        assert roi.attrs["source_path"] == "/entry/data/data"
        frames = roi[...]
        totals = frames.sum(axis=(1, 2))
        expected_x = frames.sum(axis=2) @ np.arange(frames.shape[1]) / totals
        expected_y = frames.sum(axis=1) @ np.arange(frames.shape[2]) / totals
        np.testing.assert_allclose(
            h5f["/entry/data/rois/info/Batch_ROI/ComX"], expected_x)
        np.testing.assert_allclose(
            h5f["/entry/data/rois/info/Batch_ROI/ComY"], expected_y)
        result = h5f["/entry/data/rois/info/Batch_ROI"]
        assert result.attrs["NX_class"] == "NXcollection"
        assert result["ComX"].attrs["units"] == "pixel"
        assert result["ComY"].attrs["units"] == "pixel"
        assert h5f.get(f"{result.name}/roi", getlink=True).path == saved_path
        assert h5f.get(f"{result.name}/source", getlink=True).path == "/entry/data/data"
        assert roi.attrs["com_x"] == f"{result.name}/ComX"
        assert roi.attrs["com_y"] == f"{result.name}/ComY"
        assert roi.attrs["analysis"] == result.name


def test_saved_com_matches_the_spot_the_roi_was_drawn_on(tmp_path):
    path = tmp_path / "spot.h5"
    data = np.zeros((1, 12, 16))
    data[0, 7, 4] = 1.0
    with h5py.File(path, "w") as h5f:
        h5f["/entry/data/data"] = data

    ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Spot", (4, 3, 5, 4))

    with h5py.File(path, "r") as h5f:
        assert h5f["/entry/data/rois/info/Spot/ComX"][0] == 3.0
        assert h5f["/entry/data/rois/info/Spot/ComY"][0] == 1.0


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
        assert second_roi.shape == (3, 4, 5)
        assert (second_roi.attrs["x"], second_roi.attrs["y"]) == (2, 3)
        assert "/entry/data/rois/info/Batch_ROI/ComX" in h5f
        assert "/entry/data/rois/info/Batch_ROI(2)/ComX" in h5f


def test_deleting_an_roi_removes_its_results_and_keeps_the_others(tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)
    first = ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "First", (1, 1, 2, 2))
    second = ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "Second", (2, 2, 2, 2))

    with h5py.File(path, "a") as h5f:
        ROIManager._delete_roi_dataset(h5f, first)
        assert first not in h5f
        assert "/entry/data/rois/info/First" not in h5f
        assert "/entry/data/rois/info/Second/ComX" in h5f

        ROIManager._delete_roi_dataset(h5f, second)
        assert "/entry/data/rois" not in h5f
        assert "/entry/data/data" in h5f


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
        assert roi.shape == (3, 4, 5)
        assert rois["info/Batch_ROI/ComX"].shape == (3,)
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


def test_delete_entire_batch_removes_all_linked_copies(tmp_path):
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    batch_files = [str(first), str(second)]
    for path in (first, second):
        _image_file(path)
        ROIManager._save_roi_geometry_to_file(
            str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2),
            batch_id="batch-1", batch_files=batch_files)

    class Main:
        def update_status(self, *args, **kwargs):
            pass

    roi = object()
    manager = ROIManager(Main())
    manager.roi_source_by_id[id(roi)] = {
        "batch_id": "batch-1",
        "batch_files": batch_files,
        "file_path": str(first),
    }

    manager.delete_roi_from_disk(roi, delete_batch=True)

    for path in (first, second):
        with h5py.File(path, "r") as h5f:
            assert "/entry/data/rois/Batch_ROI" not in h5f


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


def test_save_batch_to_reuses_batch_id_and_replaces_manifest(qapp, monkeypatch, tmp_path):
    current = tmp_path / "current.h5"
    removed = tmp_path / "removed.h5"
    added = tmp_path / "added.h5"
    old_files = [str(current), str(removed)]
    for path in (current, removed):
        _image_file(path)
        ROIManager._save_roi_geometry_to_file(
            str(path), "/entry/data/data", "Batch ROI", (1, 1, 2, 2),
            batch_id="batch-1", batch_files=old_files)
    _image_file(added)

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
            return Point(2, 3)

        def size(self):
            return Point(4, 5)

    class Main:
        current_file_path = str(current)

        def update_status(self, *args, **kwargs):
            pass

    roi = Roi()
    manager = ROIManager(Main())
    manager.roi_source_by_id[id(roi)] = {
        "file_path": str(current),
        "dataset_path": "/entry/data/data",
        "batch_id": "batch-1",
        "batch_files": old_files,
        "batch_name": "Batch ROI",
    }
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: QMessageBox.Yes)

    manager._save_batch_to_files(roi, [str(added)])

    expected_manifest = f'["{current}", "{added}"]'
    for path in (current, added):
        with h5py.File(path, "r") as h5f:
            dataset = h5f["/entry/data/rois/Batch_ROI"]
            assert dataset.attrs["batch_id"] == "batch-1"
            assert dataset.attrs["batch_files"] == expected_manifest
    with h5py.File(removed, "r") as h5f:
        assert "/entry/data/rois/Batch_ROI" not in h5f
    assert manager.roi_source_by_id[id(roi)]["batch_files"] == [
        str(current), str(added)]


def test_roi_dock_groups_standalone_and_batch_rois(qapp):
    class Point:
        def __init__(self, x, y):
            self._x = x
            self._y = y

        def x(self):
            return self._x

        def y(self):
            return self._y

    class Roi:
        def __init__(self, x):
            self._x = x

        def pos(self):
            return Point(self._x, 2)

        def size(self):
            return Point(3, 4)

    class Main:
        roi_list = QListWidget()
        current_roi = None

    standalone = Roi(1)
    batch = Roi(5)
    manager = ROIManager(Main())
    manager.rois = [standalone, batch]
    manager.roi_names = {id(standalone): "Solo", id(batch): "Linked"}
    manager.roi_source_by_id[id(batch)] = {"batch_id": "batch-1"}

    manager._rebuild_roi_dock()

    texts = [Main.roi_list.item(row).text() for row in range(Main.roi_list.count())]
    assert texts == [
        "Solo: x=1, y=2, w=3, h=4",
        "Batches", "Linked: x=5, y=2, w=3, h=4",
    ]
    assert Main.roi_list.item(1).flags() == Qt.NoItemFlags


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

    assert manager.format_roi_text(roi) == "Linked ROI: x=1, y=2, w=3, h=4"


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

    roi = Roi()
    manager._save_roi_to_files(roi, [str(other), str(current)])

    assert manager.roi_source_by_id[id(roi)]["roi_dataset_path"] == "/entry/data/rois/Exported_ROI"
    assert manager.roi_source_by_id[id(roi)]["file_path"] == str(current)

    for path in (current, other):
        with h5py.File(path, "r") as h5f:
            assert "/entry/data/rois/Exported_ROI" in h5f
            assert "/entry/data/rois/info/Exported_ROI/ComX" in h5f
            assert "/entry/data/rois/info/Exported_ROI/ComY" in h5f


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


def test_save_roi_refuses_an_roi_that_was_never_saved(tmp_path):
    path = tmp_path / "scan.h5"
    data = np.zeros((2, 12, 16), dtype=np.float32)
    data[:, 7, 4] = 1.0
    with h5py.File(path, "w") as h5f:
        h5f["/entry/data/data"] = data

    class Point:
        def __init__(self, x, y):
            self._x, self._y = x, y

        def x(self):
            return self._x

        def y(self):
            return self._y

    class Roi:
        def pos(self):
            return Point(4, 3)

        def size(self):
            return Point(5, 4)

    class Main:
        current_file_path = str(path)
        current_2d_data = data
        selected_dataset_path = "/entry/data/data"

        def update_status(self, *args, **kwargs):
            pass

    manager = ROIManager(Main())
    manager.get_roi_name = lambda roi: "New ROI"
    manager.save_roi(Roi())

    with h5py.File(path, "r") as h5f:
        assert "/entry/data/rois" not in h5f


def _file_backed_manager(path, data):
    class Main:
        current_file_path = str(path)
        current_2d_data = data
        selected_dataset_path = "/entry/data/data"
        image_view = types.SimpleNamespace(addItem=lambda item: None, imageItem=None)
        rois = []

        def update_status(self, *args, **kwargs):
            pass

        def get_current_frame_data(self):
            return data[0]

    manager = ROIManager(Main())
    manager.render_rois_for_dataset(str(path), "/entry/data/data")
    return manager


def test_save_roi_overrides_the_existing_roi(qapp, tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)
    ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "R", (1, 1, 2, 2))
    with h5py.File(path, "r") as h5f:
        data = h5f["/entry/data/data"][...].astype(np.float32)
    manager = _file_backed_manager(path, data)

    manager.save_roi(manager.rois[0])

    with h5py.File(path, "r") as h5f:
        assert list(k for k in h5f["/entry/data/rois"] if k != "info") == ["R"]
        info = h5f["/entry/data/rois/info"]
        assert [k for k in info if isinstance(info[k], h5py.Group)] == ["R"]


def test_delete_from_disk_finds_a_renamed_file_backed_roi(qapp, tmp_path):
    path = tmp_path / "scan.h5"
    _image_file(path)
    ROIManager._save_roi_geometry_to_file(
        str(path), "/entry/data/data", "R", (1, 1, 2, 2))
    with h5py.File(path, "r") as h5f:
        data = h5f["/entry/data/data"][...].astype(np.float32)
    manager = _file_backed_manager(path, data)
    roi = manager.rois[0]
    manager.roi_names[id(roi)] = "Renamed"

    manager.delete_roi_from_disk(roi)

    with h5py.File(path, "r") as h5f:
        assert "/entry/data/rois" not in h5f
