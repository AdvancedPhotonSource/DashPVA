#!/usr/bin/env python3
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


"""Named-input persistence, usable by a window that cannot inherit BaseWindow.

Kept out of BaseWindow so a plain QMainWindow (the Scan Monitor, the Bayesian
viewer) gets the same save/restore walk by mixing this in.

Example:
    class MyWindow(UiStateMixin, QMainWindow):
        persist_input_skip = {'lineedit_password'}
"""
import json

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
)


class UiStateMixin:
    """Walk a window's named widgets, saving and restoring their values."""

    #: objectNames never persisted (per class; add to it in subclasses).
    persist_input_skip: set = set()

    #: Widget type -> (getter, setter). Splitters are included, so splitter
    #: positions persist alongside the plain inputs.
    _INPUT_ACCESSORS = (
        (QLineEdit, lambda w: w.text(), lambda w, v: w.setText(str(v))),
        (QComboBox, lambda w: w.currentText(), None),          # setter needs a lookup
        (QDoubleSpinBox, lambda w: w.value(), lambda w, v: w.setValue(float(v))),
        (QSpinBox, lambda w: w.value(), lambda w, v: w.setValue(int(v))),
        (QCheckBox, lambda w: w.isChecked(), lambda w, v: w.setChecked(bool(v))),
        (QRadioButton, lambda w: w.isChecked(), lambda w, v: w.setChecked(bool(v))),
        (QSlider, lambda w: w.value(), lambda w, v: w.setValue(int(v))),
        (QTabWidget, lambda w: w.currentIndex(), lambda w, v: w.setCurrentIndex(int(v))),
        (QSplitter, lambda w: w.sizes(), lambda w, v: w.setSizes([int(x) for x in v])),
    )

    def _persisted_inputs(self):
        """Yield ``(name, widget, getter, setter)`` for every persistable input.

        Anything without a stable objectName is skipped: Qt's internal children
        are named ``qt_*``, and unnamed designer widgets have no key to store
        them under. ``seen`` guards against a widget matching twice.
        """
        seen = set()
        for cls, getter, setter in self._INPUT_ACCESSORS:
            for w in self.findChildren(cls):
                name = w.objectName()
                if (not name or name.startswith('qt_') or name in seen
                        or name in self.persist_input_skip):
                    continue
                seen.add(name)
                yield name, w, getter, setter

    def session_inputs(self) -> dict:
        """Current value of every named input, keyed by objectName."""
        values = {}
        for name, w, getter, _ in self._persisted_inputs():
            try:
                values[name] = getter(w)
            except Exception:
                pass
        return values

    def apply_session_inputs(self, values: dict) -> None:
        """Re-apply saved input values, skipping anything that no longer fits."""
        for name, w, _, setter in self._persisted_inputs():
            if name not in values:
                continue
            value = values[name]
            try:
                if isinstance(w, QComboBox):
                    # Items may be populated at runtime; only restore a choice
                    # that actually exists, never inject a new one.
                    idx = w.findText(str(value))
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                    elif w.isEditable():
                        w.setEditText(str(value))
                elif setter is not None:
                    setter(w, value)
            except Exception:
                pass

    #: Set False on a window that opts out of machine-local persistence.
    persist_state = True

    def _qsettings(self) -> QSettings:
        """QSettings scoped to this window's type (org "DashPVA").

        Lives here rather than in BaseWindow so the mixin stands on its own: a
        plain QMainWindow that mixes it in needs somewhere to read and write.
        """
        return QSettings("DashPVA", type(self).__name__)

    def save_inputs(self) -> None:
        """Write the named inputs' values to this window's store."""
        if not self.persist_state:
            return
        self._qsettings().setValue("inputs", json.dumps(self.session_inputs()))

    def restore_inputs(self) -> None:
        """Re-apply the input values saved by the previous session."""
        if not self.persist_state:
            return
        raw = self._qsettings().value("inputs")
        if not raw:
            return
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            return
        if isinstance(values, dict) and values:
            self.apply_session_inputs(values)

