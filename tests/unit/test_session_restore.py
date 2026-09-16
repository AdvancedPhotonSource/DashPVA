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
from PyQt5.QtCore import QSettings, Qt
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

    def session_paths(self):
        return self._paths

    def session_view(self):
        return self._view

    def load_file_content(self, path):
        self.opened.append(("file", path))

    def load_folder_content(self, path):
        self.opened.append(("folder", path))

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


def test_missing_paths_are_dropped_and_no_view_is_shown_without_files(qapp, tmp_path):
    _Session([["file", str(tmp_path / "gone.h5")]], {"tab": "2D View"}).close()

    reopened = _Session()
    reopened._restore_on_start()

    assert reopened.opened == []
    assert reopened.shown == []


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
