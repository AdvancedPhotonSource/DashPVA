# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
"""Shared input-state persistence for windows that keep UI state in QSettings.

Used by :class:`~dashpva.viewer.core.base_window.BaseWindow` and by windows that
cannot inherit it (the Scan Monitor is a plain ``QMainWindow``)."""
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLineEdit,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
)


class UiStateMixin:
    """Save and restore every named input widget on a window.

        class MyWindow(UiStateMixin, QMainWindow):
            persist_input_skip = {"lineedit_channel"}

        w = MyWindow()
        values = w.session_inputs()        # {objectName: value}
        w.apply_session_inputs(values)
    """

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
        (QGroupBox, lambda w: w.isChecked(), lambda w, v: w.setChecked(bool(v))),
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
                # A plain QGroupBox has no check state worth storing.
                if isinstance(w, QGroupBox) and not w.isCheckable():
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
