#!/usr/bin/env python3
# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Editable review table for a set of pending configuration changes."""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
)

from dashpva.gui.theme_colors import ERROR, SUCCESS, TEXT_MUTED, WARNING


class ChangeReviewDialog(QDialog):
    """Key / Old / New table for reviewing pending changes before they are saved.

    Each row is ``(kind, key, old, new)`` with *kind* one of ``change``, ``add``
    or ``remove``. The New cell is editable and each row can be dropped, so the
    operator confirms exactly what gets written rather than accepting the lot.

    Example:
        rows = [("change", "SAMPLE_ORIENTATION", "det", "sam")]
        result = ChangeReviewDialog.review(self, rows)
        if result is not None:
            kept, dropped = result
            print(kept)    # [('change', 'SAMPLE_ORIENTATION', 'det', 'sam')]
    """

    def __init__(self, parent, rows, is_editable=None, title="Review Changes"):
        super().__init__(parent)
        self.rows = list(rows)
        self._dropped = [False] * len(self.rows)
        self.setWindowTitle(title)
        self.setMinimumSize(720, 420)

        layout = QVBoxLayout(self)
        label = QLabel(
            f"{len(self.rows)} pending change(s). Edit a NEW value, or ✕ to drop a change."
        )
        layout.addWidget(label)
        layout.setAlignment(label, Qt.AlignLeft)

        self.table = QTableWidget(len(self.rows), 4, self)
        self.table.setHorizontalHeaderLabels(["Key", "Old", "New", "Action"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        for row, (kind, key, old, new) in enumerate(self.rows):
            writable = is_editable is None or is_editable(key)
            value_editable = kind in ("change", "add") and writable
            self._fill_row(
                row,
                kind,
                key,
                old,
                new,
                value_editable=value_editable,
                droppable=writable,
            )

        layout.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _fill_row(
        self,
        row,
        kind,
        key,
        old,
        new,
        *,
        value_editable,
        droppable,
    ):
        key_item = QTableWidgetItem(
            f"+ {key}" if kind == "add" else f"- {key}" if kind == "remove" else key
        )
        key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
        if kind == "add":
            key_item.setForeground(QColor(SUCCESS))
        elif kind == "remove":
            key_item.setForeground(QColor(ERROR))
        self.table.setItem(row, 0, key_item)

        old_item = QTableWidgetItem(old or "")
        old_item.setFlags(old_item.flags() & ~Qt.ItemIsEditable)
        if kind == "change":
            old_item.setForeground(QColor(WARNING))
        elif kind == "remove":
            old_item.setForeground(QColor(ERROR))
        self.table.setItem(row, 1, old_item)

        new_item = QTableWidgetItem(new or "")
        if value_editable:
            new_item.setFlags(new_item.flags() | Qt.ItemIsEditable)
            new_item.setForeground(QColor(SUCCESS))
        else:
            new_item.setFlags(new_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 2, new_item)

        if droppable:
            button = QToolButton()
            button.setText("✕")
            button.setToolTip("Drop this change")
            button.setAutoRaise(True)
            button.clicked.connect(lambda checked=False, idx=row: self._drop(idx))
            self.table.setCellWidget(row, 3, button)

    def _drop(self, index):
        if self._dropped[index]:
            return
        self._dropped[index] = True
        for column in range(3):
            cell = self.table.item(index, column)
            if cell is None:
                continue
            font = cell.font()
            font.setStrikeOut(True)
            cell.setFont(font)
            cell.setForeground(QColor(TEXT_MUTED))
            cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
        button = self.table.cellWidget(index, 3)
        if button is not None:
            button.setEnabled(False)

    def decisions(self):
        """``(kept, dropped)`` rows, kept carrying any edited New value."""
        kept, dropped = [], []
        for row, (kind, key, old, new) in enumerate(self.rows):
            if self._dropped[row]:
                dropped.append((kind, key, old, new))
            else:
                cell = self.table.item(row, 2)
                kept.append((kind, key, old, cell.text() if cell else new))
        return kept, dropped

    @classmethod
    def review(cls, parent, rows, is_editable=None, title="Review Changes"):
        """Show the dialog; return ``(kept, dropped)``, or None if cancelled."""
        dialog = cls(parent, rows, is_editable=is_editable, title=title)
        if dialog.exec_() != QDialog.Accepted:
            return None
        return dialog.decisions()
