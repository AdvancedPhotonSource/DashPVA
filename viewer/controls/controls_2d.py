"""
2D Controls wiring for Workbench and other viewers.
Encapsulates signal connections for 2D-specific UI elements.
"""

from typing import Optional
from PyQt5.QtWidgets import QLabel, QHBoxLayout


class Controls2D:
    def __init__(self, main_window):
        self.main = main_window
        self._roi_stats_label_added = False

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
            if hasattr(self.main, 'sbVmax'):
                self.main.sbVmax.valueChanged.connect(self.main.on_vmax_changed)

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
