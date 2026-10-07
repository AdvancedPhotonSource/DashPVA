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

from dashpva.viewer.workbench.managers.roi_manager import ROIManager  # noqa: E402


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
