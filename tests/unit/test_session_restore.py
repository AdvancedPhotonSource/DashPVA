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

"""Session restore: reopen the files a window had open and show what it was showing."""
from __future__ import annotations

import time

import pytest
from PyQt5.QtCore import QSettings, Qt, QThread
from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem

from dashpva.viewer.core.base_window import BaseWindow
from dashpva.viewer.workbench.workbench import WorkbenchWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path):
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(tmp_path / "qs"))
    yield


class _Session(BaseWindow):
    """Records what restore_session asks it to open and show."""

    def __init__(self, paths=(), view=None):
        super().__init__(viewer_name="SessionTest")
        self._paths, self._view = [list(p) for p in paths], view or {}
        self.opened, self.shown = [], []
        self.prepared_threads, self.applied_threads = [], []

    def session_paths(self):
        return self._paths

    def session_view(self):
        return self._view

    def load_file_content(self, path):
        self.opened.append(("file", path))

    def load_folder_content(self, path):
        self.opened.append(("folder", path))

    def prepare_session_entry(self, kind, path):
        self.prepared_threads.append(QThread.currentThread())
        return super().prepare_session_entry(kind, path)

    def apply_session_entry(self, prepared):
        self.applied_threads.append(QThread.currentThread())
        super().apply_session_entry(prepared)

    def restore_session_view(self, view):
        self.shown.append(view)


def test_close_then_start_reopens_files_then_restores_the_view(qapp, tmp_path):
    folder, scan = tmp_path / "scans", tmp_path / "scan.h5"
    folder.mkdir()
    scan.write_bytes(b"x")
    view = {"file": str(scan), "dataset": "/entry/data/data", "tab": "2D View", "frame": 7}
    _Session([["folder", str(folder)], ["file", str(scan)]], view).close()

    reopened = _Session()           # queues its own restore, as on a real launch
    assert reopened.opened == []   # nothing is loaded before the event loop runs (window paints first)

    deadline = time.monotonic() + 5
    while not reopened.shown and time.monotonic() < deadline:
        qapp.processEvents()

    assert reopened.opened == [("folder", str(folder)), ("file", str(scan))]
    assert reopened.shown == [view]
    assert all(thread is not qapp.thread() for thread in reopened.prepared_threads)
    assert all(thread is qapp.thread() for thread in reopened.applied_threads)


def test_missing_paths_are_dropped_and_no_view_is_shown_without_files(qapp, tmp_path):
    _Session([["file", str(tmp_path / "gone.h5")]], {"tab": "2D View"}).close()

    reopened = _Session()
    reopened._restore_on_start()

    assert reopened.opened == []
    assert reopened.shown == []


def test_restore_ignores_wrong_kinds_and_path_types(qapp, tmp_path, monkeypatch):
    folder, scan = tmp_path / "scans", tmp_path / "scan.h5"
    folder.mkdir()
    scan.write_bytes(b"x")
    paths = [
        ["folder", str(scan)],
        ["file", str(folder)],
        ["other", str(scan)],
        ["folder", str(folder)],
        ["file", str(scan)],
    ]
    _Session(paths).close()

    reopened = _Session()
    reopened.restore_session()
    deadline = time.monotonic() + 5
    while len(reopened.opened) < 2 and time.monotonic() < deadline:
        qapp.processEvents()

    assert reopened.opened == [("folder", str(folder)), ("file", str(scan))]


def _tree():
    tree = QTreeWidget()
    for kind, path in (("folder_section", "/data/scans"), ("file_root", "/data/a.h5")):
        item = QTreeWidgetItem([path])
        item.setData(0, Qt.UserRole + 2, kind)
        item.setData(0, Qt.UserRole + 1, path)
        tree.addTopLevelItem(item)
    dataset = QTreeWidgetItem(["data"])
    dataset.setData(0, 32, "/entry/data/data")
    tree.topLevelItem(1).addChild(dataset)
    return tree, dataset


def test_workbench_session_paths_come_from_the_tree(qapp):
    tree, _ = _tree()
    window = type("W", (), {"tree_data": tree})()
    assert WorkbenchWindow.session_paths(window) == [["folder", "/data/scans"], ["file", "/data/a.h5"]]


def test_workbench_finds_the_saved_dataset_only_under_its_own_file(qapp):
    tree, dataset = _tree()
    window = type("W", (), {"tree_data": tree})()
    assert WorkbenchWindow._find_dataset_item(window, "/data/a.h5", "/entry/data/data") is dataset
    assert WorkbenchWindow._find_dataset_item(window, "/data/other.h5", "/entry/data/data") is None


def test_workbench_always_load_all_bypasses_the_size_limit(tmp_path):
    scan = tmp_path / "large.h5"
    scan.write_bytes(b"123456")
    window = type("W", (), {"_always_load_all": lambda self: True})()

    assert WorkbenchWindow.should_restore_session_file(window, str(scan)) is True


def test_workbench_restores_saved_top_level_order(qapp):
    tree = QTreeWidget()
    for path in ("/data/b.h5", "/data/a.h5"):
        item = QTreeWidgetItem([path])
        item.setData(0, Qt.UserRole + 1, path)
        tree.addTopLevelItem(item)
    window = type("W", (), {"tree_data": tree})()

    WorkbenchWindow.restore_session_path_order(
        window, [["file", "/data/a.h5"], ["file", "/data/b.h5"]])

    assert [tree.topLevelItem(i).text(0) for i in range(2)] == ["/data/a.h5", "/data/b.h5"]


def test_workbench_restored_hdf5_tree_is_populated_lazily(qapp):
    tree = QTreeWidget()
    window = type("W", (), {
        "tree_data": tree,
        "_add_prepared_hdf5_nodes": WorkbenchWindow._add_prepared_hdf5_nodes,
    })()
    nodes = [{
        "text": "data (Dataset: (2, 2), float32)",
        "path": "/entry/data",
        "renderable": True,
        "children": [],
    }]

    WorkbenchWindow._add_prepared_hdf5(window, "/data/scan.h5", nodes=nodes)
    root = tree.topLevelItem(0)

    assert root.childCount() == 1
    assert root.child(0).text(0) == "Loading structure..."

    WorkbenchWindow._populate_prepared_hdf5_item(window, root)

    assert root.childCount() == 1
    assert root.child(0).data(0, 32) == "/entry/data"


def test_workbench_inspects_unprepared_structure_only_when_expanded(qapp):
    tree = QTreeWidget()
    inspected = []
    nodes = [{
        "text": "data (Dataset: (2, 2), float32)",
        "path": "/entry/data",
        "renderable": True,
        "children": [],
    }]
    window = type("W", (), {
        "tree_data": tree,
        "_inspect_hdf5_file": lambda self, path: inspected.append(path) or nodes,
        "_add_prepared_hdf5_nodes": WorkbenchWindow._add_prepared_hdf5_nodes,
        "_populate_prepared_hdf5_item": WorkbenchWindow._populate_prepared_hdf5_item,
    })()
    tree.itemExpanded.connect(window._populate_prepared_hdf5_item)

    WorkbenchWindow._add_prepared_hdf5(window, "/data/scan.h5")
    root = tree.topLevelItem(0)

    assert inspected == []

    root.setExpanded(True)
    qapp.processEvents()

    assert inspected == ["/data/scan.h5"]
    assert root.child(0).data(0, 32) == "/entry/data"
