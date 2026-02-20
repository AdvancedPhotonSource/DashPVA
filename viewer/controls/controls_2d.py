"""
2D Controls wiring for Workbench and other viewers.
Encapsulates signal connections for 2D-specific UI elements.
"""

from typing import Optional
from PyQt5.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QFormLayout,
    QWidget,
    QLayout,
)


class Controls2D:
    def __init__(self, main_window):
        self.main = main_window
        self._roi_stats_label_added = False
        self._vmin_vmax_hints_inserted = False

    def setup(self) -> None:
        """Wire up 2D controls to main window handlers."""
        try:
            # Colormap selection
            if hasattr(self.main, 'cbColorMapSelect_2d'):
                self.main.cbColorMapSelect_2d.currentTextChanged.connect(self.main.on_colormap_changed)

            # Auto levels checkbox
            if hasattr(self.main, 'cbAutoLevels'):
                self.main.cbAutoLevels.toggled.connect(self.main.on_auto_levels_toggled)

            # Frame navigation controls
            if hasattr(self.main, 'btn_prev_frame'):
                self.main.btn_prev_frame.clicked.connect(self.main.previous_frame)
            if hasattr(self.main, 'btn_next_frame'):
                self.main.btn_next_frame.clicked.connect(self.main.next_frame)
            if hasattr(self.main, 'frame_spinbox'):
                self.main.frame_spinbox.valueChanged.connect(self.main.on_frame_spinbox_changed)

            # Speckle analysis & intensity controls
            if hasattr(self.main, 'cbLogScale'):
                self.main.cbLogScale.toggled.connect(self.main.on_log_scale_toggled)
            if hasattr(self.main, 'sbVmin'):
                self.main.sbVmin.valueChanged.connect(self.main.on_vmin_changed)
                # Stabilize spinbox width to prevent constant shrinking/growing as ranges/values change
                try:
                    # Widen to avoid truncation and keep a stable width
                    self.main.sbVmin.setFixedWidth(120)
                except Exception:
                    pass
            if hasattr(self.main, 'sbVmax'):
                self.main.sbVmax.valueChanged.connect(self.main.on_vmax_changed)
                # Stabilize spinbox width to prevent constant shrinking/growing as ranges/values change
                try:
                    # Widen to avoid truncation and keep a stable width
                    self.main.sbVmax.setFixedWidth(120)
                except Exception:
                    pass

            # Create two hint labels for Vmin/Vmax (data-domain). Create once on main.
            try:
                # Prefer to reuse existing static UI labels if present
                ui_has_vlabels = hasattr(self.main, 'label_vmin') and hasattr(self.main, 'label_vmax')
                if ui_has_vlabels:
                    # Map hint references to the static UI labels so they are updated by _update_vmin_vmax_hints()
                    try:
                        self.main.lblVminHint = self.main.label_vmin
                        self.main.lblVmaxHint = self.main.label_vmax
                        # Initialize with placeholders to reflect data-domain hints
                        if self.main.lblVminHint is not None:
                            self.main.lblVminHint.setText("Vmin(...):")
                        if self.main.lblVmaxHint is not None:
                            self.main.lblVmaxHint.setText("Vmax(...):")
                        # Mark as inserted to skip runtime placement
                        self._vmin_vmax_hints_inserted = True
                    except Exception:
                        pass
                # Otherwise, create lightweight labels to insert next to spinboxes
                if not ui_has_vlabels:
                    if not hasattr(self.main, 'lblVminHint') or self.main.lblVminHint is None:
                        self.main.lblVminHint = QLabel("Vmin(...):")
                        self.main.lblVminHint.setStyleSheet("color: #6c757d; font-size: 10px; padding-right: 6px;")
                    if not hasattr(self.main, 'lblVmaxHint') or self.main.lblVmaxHint is None:
                        self.main.lblVmaxHint = QLabel("Vmax(...):")
                        self.main.lblVmaxHint.setStyleSheet("color: #6c757d; font-size: 10px; padding-right: 6px;")
            except Exception:
                # If creation fails, continue without hints
                pass

            # Attempt to place hints immediately before sbVmin/sbVmax using runtime layout introspection
            try:
                placed_vmin = False
                placed_vmax = False
                # Only attempt runtime placement if we did not reuse static UI labels
                if (not self._vmin_vmax_hints_inserted and hasattr(self.main, 'lblVminHint') and hasattr(self.main, 'sbVmin') and
                        self.main.lblVminHint is not None and self.main.sbVmin is not None and
                        not self._vmin_vmax_hints_inserted):
                    placed_vmin = self._insert_label_before_widget(self.main.lblVminHint, self.main.sbVmin)
                if (not self._vmin_vmax_hints_inserted and hasattr(self.main, 'lblVmaxHint') and hasattr(self.main, 'sbVmax') and
                        self.main.lblVmaxHint is not None and self.main.sbVmax is not None and
                        not self._vmin_vmax_hints_inserted):
                    placed_vmax = self._insert_label_before_widget(self.main.lblVmaxHint, self.main.sbVmax)

                # Fallback: if precise placement fails for one or both, add only the missing ones into layout_2d_controls_main
                if not (placed_vmin and placed_vmax):
                    try:
                        if hasattr(self.main, 'layout_2d_controls_main') and self.main.layout_2d_controls_main is not None:
                            hbox_hints = QHBoxLayout()
                            try:
                                hbox_hints.setContentsMargins(0, 0, 0, 0)
                                hbox_hints.setSpacing(4)
                            except Exception:
                                pass
                            # Add only widgets that are not already parented/placed
                            if not placed_vmin and hasattr(self.main, 'lblVminHint') and self.main.lblVminHint is not None:
                                try:
                                    if self.main.lblVminHint.parentWidget() is None:
                                        hbox_hints.addWidget(self.main.lblVminHint)
                                except Exception:
                                    pass
                            if not placed_vmax and hasattr(self.main, 'lblVmaxHint') and self.main.lblVmaxHint is not None:
                                try:
                                    if self.main.lblVmaxHint.parentWidget() is None:
                                        hbox_hints.addWidget(self.main.lblVmaxHint)
                                except Exception:
                                    pass
                            # Only add layout if we actually added any widgets
                            if hbox_hints.count() > 0:
                                # Prefer to insert just below the top intensity row (layout_2d_controls_top) if available
                                try:
                                    insert_index = None
                                    top_layout = getattr(self.main, 'layout_2d_controls_top', None)
                                    if top_layout is not None:
                                        for i in range(self.main.layout_2d_controls_main.count()):
                                            it = self.main.layout_2d_controls_main.itemAt(i)
                                            if it is not None and it.layout() is top_layout:
                                                insert_index = i + 1
                                                break
                                    if insert_index is not None:
                                        self.main.layout_2d_controls_main.insertLayout(insert_index, hbox_hints)
                                    else:
                                        # Fallback to appending at the end
                                        self.main.layout_2d_controls_main.addLayout(hbox_hints)
                                except Exception:
                                    # Fallback to appending at the end on any error
                                    self.main.layout_2d_controls_main.addLayout(hbox_hints)
                            # Mark as inserted if both are now parented somewhere
                            try:
                                vmin_ok = hasattr(self.main, 'lblVminHint') and self.main.lblVminHint is not None and self.main.lblVminHint.parentWidget() is not None
                                vmax_ok = hasattr(self.main, 'lblVmaxHint') and self.main.lblVmaxHint is not None and self.main.lblVmaxHint.parentWidget() is not None
                                self._vmin_vmax_hints_inserted = bool(vmin_ok and vmax_ok)
                            except Exception:
                                self._vmin_vmax_hints_inserted = self._vmin_vmax_hints_inserted or (placed_vmin and placed_vmax)
                    except Exception:
                        # As a last resort, do nothing; hints remain unattached
                        pass
                else:
                    self._vmin_vmax_hints_inserted = True
            except Exception:
                # Guard UI quirks
                pass

            # ROI drawing
            if hasattr(self.main, 'btnDrawROI'):
                self.main.btnDrawROI.clicked.connect(self.main.on_draw_roi_clicked)

            # Reference/Other frame selection for speckle compare
            if hasattr(self.main, 'sbRefFrame'):
                self.main.sbRefFrame.valueChanged.connect(self.main.on_ref_frame_changed)
            if hasattr(self.main, 'sbOtherFrame'):
                self.main.sbOtherFrame.valueChanged.connect(self.main.on_other_frame_changed)

            # Analyze speckle
            if hasattr(self.main, 'btnAnalyzeSpeckle'):
                self.main.btnAnalyzeSpeckle.clicked.connect(self.main.on_analyze_speckle_clicked)

            # Plot motor positions
            if hasattr(self.main, 'btnPlotMotorPositions'):
                self.main.btnPlotMotorPositions.clicked.connect(self.main.on_plot_motor_positions_clicked)

            # Playback controls wiring (ensure timer exists and connect buttons)
            try:
                from PyQt5.QtCore import QTimer
                if not hasattr(self.main, 'play_timer') or self.main.play_timer is None:
                    self.main.play_timer = QTimer(self.main)
                    try:
                        self.main.play_timer.timeout.connect(self.main._advance_frame_playback)
                        print("[PLAYBACK][Controls2D] Created play_timer and wired timeout")
                    except Exception as e:
                        print(f"[PLAYBACK][Controls2D] ERROR wiring timer: {e}")

                if hasattr(self.main, 'btn_play'):
                    try:
                        self.main.btn_play.clicked.connect(self.main.start_playback)
                        print("[PLAYBACK][Controls2D] Wired btn_play -> start_playback")
                    except Exception as e:
                        print(f"[PLAYBACK][Controls2D] ERROR wiring btn_play: {e}")
                else:
                    print("[PLAYBACK][Controls2D] btn_play not found on main")

                if hasattr(self.main, 'btn_pause'):
                    try:
                        self.main.btn_pause.clicked.connect(self.main.pause_playback)
                        print("[PLAYBACK][Controls2D] Wired btn_pause -> pause_playback")
                    except Exception as e:
                        print(f"[PLAYBACK][Controls2D] ERROR wiring btn_pause: {e}")
                else:
                    print("[PLAYBACK][Controls2D] btn_pause not found on main")

                if hasattr(self.main, 'sb_fps'):
                    try:
                        self.main.sb_fps.valueChanged.connect(self.main.on_fps_changed)
                        print("[PLAYBACK][Controls2D] Wired sb_fps -> on_fps_changed")
                    except Exception as e:
                        print(f"[PLAYBACK][Controls2D] ERROR wiring sb_fps: {e}")
                else:
                    print("[PLAYBACK][Controls2D] sb_fps not found on main")
            except Exception as e:
                try:
                    print(f"[PLAYBACK][Controls2D] ERROR in playback wiring block: {e}")
                except Exception:
                    pass

            # Add ROI Stats label at the bottom of 2D controls once
            if hasattr(self.main, 'layout_2d_controls_main') and not hasattr(self.main, 'roi_stats_label'):
                try:
                    self.main.roi_stats_label = QLabel("ROI Stats: -")
                    self.main.roi_stats_label.setStyleSheet("color: #2c3e50; font-size: 11px;")
                    hbox = QHBoxLayout()
                    hbox.addWidget(self.main.roi_stats_label)
                    self.main.layout_2d_controls_main.addLayout(hbox)
                    self._roi_stats_label_added = True
                except Exception:
                    pass
        except Exception as e:
            try:
                self.main.update_status(f"Error setting up 2D connections: {e}")
            except Exception:
                pass

    def _insert_label_before_widget(self, label: QLabel, widget: QWidget) -> bool:
        """Safely place a label immediately before a target widget within its parent layout.

        Supports QHBoxLayout/QVBoxLayout by iterating items to find index and insert.
        Supports QGridLayout via indexOf/widget position; tries (row, col-1) if empty.
        Supports QFormLayout via getWidgetPosition; uses LabelRole when available.
        Returns True on successful insertion; False otherwise.
        """
        try:
            if label is None or widget is None:
                return False
            parent = widget.parentWidget()
            if parent is None:
                return False
            root_layout = parent.layout() if hasattr(parent, 'layout') else None
            if root_layout is None:
                return False

            # Helper: attempt insertion within a given layout; recurse into nested child layouts
            def _attempt_in_layout(layout: QLayout) -> bool:
                try:
                    # Box layouts: look for the widget directly; if not found, search nested layouts
                    if isinstance(layout, (QHBoxLayout, QVBoxLayout)):
                        # Direct children (widgets)
                        for i in range(layout.count()):
                            it = layout.itemAt(i)
                            w = it.widget() if it is not None else None
                            if w is widget:
                                layout.insertWidget(i, label)
                                return True
                        # Nested layouts inside this box layout
                        for i in range(layout.count()):
                            it = layout.itemAt(i)
                            sub_layout = it.layout() if it is not None else None
                            if sub_layout is not None:
                                if _attempt_in_layout(sub_layout):
                                    return True
                        return False

                    # Grid layout: try previous column in same row; else search nested layouts
                    if isinstance(layout, QGridLayout):
                        idx = layout.indexOf(widget)
                        if idx >= 0:
                            row, col, rowSpan, colSpan = layout.getItemPosition(idx)
                            if col > 0:
                                existing = layout.itemAtPosition(row, col - 1)
                                if existing is None:
                                    layout.addWidget(label, row, col - 1)
                                    return True
                        # Search nested layouts contained within grid cells
                        for i in range(layout.count()):
                            it = layout.itemAt(i)
                            sub_layout = it.layout()
                            if sub_layout is not None and _attempt_in_layout(sub_layout):
                                return True
                        return False

                    # Form layout: put label in LabelRole when the widget is present; else search nested
                    if isinstance(layout, QFormLayout):
                        pos = layout.getWidgetPosition(widget)
                        if pos is not None:
                            row, role = pos
                            try:
                                existing = layout.itemAt(row, QFormLayout.LabelRole)
                                has_existing = existing is not None and existing.widget() is not None
                            except Exception:
                                has_existing = True
                            if not has_existing:
                                layout.setWidget(row, QFormLayout.LabelRole, label)
                                return True
                        # Recurse into any nested layouts contained in the form
                        try:
                            for i in range(layout.count()):
                                it = layout.itemAt(i)
                                sub_layout = it.layout()
                                if sub_layout is not None and _attempt_in_layout(sub_layout):
                                    return True
                        except Exception:
                            pass
                        return False

                    # Unknown layout type: best-effort recursion into child layouts
                    try:
                        for i in range(layout.count()):
                            it = layout.itemAt(i)
                            sub_layout = it.layout()
                            if sub_layout is not None and _attempt_in_layout(sub_layout):
                                return True
                    except Exception:
                        pass
                    return False
                except Exception:
                    return False

            return _attempt_in_layout(root_layout)
        except Exception:
            return False
