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

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ProcessManagerDialog(QDialog):
    """Non-modal window listing the launcher's running modules.

    Each row shows the module name and PID with per-process actions to raise the
    module's window (``Show``) and stop it (``Stop``); the footer stops everything.
    It reads and controls the launcher through the parent passed at construction.

    Usage:
        # `launcher` is a LauncherDialog instance
        dlg = ProcessManagerDialog(launcher)
        dlg.refresh()
        dlg.show()
    """

    def __init__(self, launcher):
        super().__init__(launcher)
        self._launcher = launcher
        self._can_raise = launcher.bring_to_front_supported()
        self.setWindowTitle('Running Modules')
        self.resize(460, 300)

        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(['Module', 'PID', 'Actions'])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

        self._lbl_empty = QLabel('No modules running', self)
        self._lbl_empty.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_empty)

        bottom = QHBoxLayout()
        bottom.addStretch()
        self.btn_shutdown_all = QPushButton('Shutdown All', self)
        self.btn_shutdown_all.setProperty('role', 'error')
        self.btn_shutdown_all.setToolTip('Force stop all running modules')
        self.btn_shutdown_all.clicked.connect(self._shutdown_all)
        bottom.addWidget(self.btn_shutdown_all)
        btn_close = QPushButton('Close', self)
        btn_close.clicked.connect(self.close)
        bottom.addWidget(btn_close)
        layout.addLayout(bottom)

    def refresh(self):
        """Rebuild the table from the launcher's tracked processes."""
        procs = self._launcher.processes
        self.table.setRowCount(0)
        for proc_id, entry in procs.items():
            p = entry.get('popen')
            if p is None or p.poll() is not None:
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(entry.get('label', proc_id))))
            self.table.setItem(row, 1, QTableWidgetItem(str(p.pid)))
            self.table.setCellWidget(row, 2, self._action_cell(proc_id, p.pid))

        has_rows = self.table.rowCount() > 0
        self.table.setVisible(has_rows)
        self._lbl_empty.setVisible(not has_rows)
        self.btn_shutdown_all.setEnabled(has_rows)

    def _action_cell(self, proc_id, pid):
        cell = QWidget(self.table)
        row = QHBoxLayout(cell)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(4)
        if self._can_raise:
            btn_show = QPushButton('Show', cell)
            btn_show.setToolTip('Bring this module\'s window to the front')
            btn_show.clicked.connect(lambda _=False, pid=pid: self._launcher.bring_to_front(pid))
            row.addWidget(btn_show)
        btn_stop = QPushButton('Stop', cell)
        btn_stop.setProperty('role', 'error')
        btn_stop.setToolTip('Force stop this module')
        btn_stop.clicked.connect(lambda _=False, pid=proc_id: self._launcher.stop_process(pid))
        row.addWidget(btn_stop)
        return cell

    def _shutdown_all(self):
        self._launcher.shutdown_all()
