"""
Standalone VIT (ptychography stitching) viewer.

Displays 5 panels: Transmission, Diffraction, Beam Position, NN Prediction, NN Stitched.
Uses PVAReader for the PVA connection and VitStitcher for stitching.

Usage:
    python viewer/vit_viewer.py
    python viewer/vit_viewer.py --channel vit:1:CSSI
"""
import os
import sys
import argparse

# --- Stitcher environment (must be set before vit_stitch is imported) ---
_default_csv = "/home/beams/AILEENLUO/ptycho-vit/workspace/positions_10um.csv"
if 'VIT_STITCH_POSITIONS_CSV' not in os.environ and os.path.exists(_default_csv):
    os.environ['VIT_STITCH_POSITIONS_CSV'] = _default_csv
if 'VIT_STITCH_STEP' not in os.environ:
    os.environ['VIT_STITCH_STEP'] = "0.05"
if 'VIT_STITCH_PIXEL_SIZE' not in os.environ:
    os.environ['VIT_STITCH_PIXEL_SIZE'] = "6.89e-9"
if 'VIT_STITCH_ID_OFFSET' not in os.environ:
    os.environ['VIT_STITCH_ID_OFFSET'] = "621356"

print("[VIT Viewer] To change VIT stitch, set before running:")
print(f"  export VIT_STITCH_POSITIONS_CSV='{os.environ.get('VIT_STITCH_POSITIONS_CSV', '')}'")
print(f"  export VIT_STITCH_ID_OFFSET='{os.environ.get('VIT_STITCH_ID_OFFSET', '621356')}'")
print(f"  export VIT_STITCH_STEP='{os.environ.get('VIT_STITCH_STEP', '0.05')}'")
print(f"  export VIT_STITCH_PIXEL_SIZE='{os.environ.get('VIT_STITCH_PIXEL_SIZE', '6.89e-9')}'")

import numpy as np
import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QCheckBox, QPushButton)
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from utils import rotation_cycle
from utils import PVAReader


rot_gen = rotation_cycle(1, 5)


class VitViewerWindow(QMainWindow):

    def __init__(self, input_channel='vit:1:input_phase'):
        super(VitViewerWindow, self).__init__()
        uic.loadUi('gui/imageshow.ui', self)
        self.setWindowTitle('DashPVA — VIT Viewer')
        self.show()
        self.reader = None
        self.image = None
        self.call_id_plot = 0
        self.first_plot = True
        self.image_is_transposed = False
        self.rot_num = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self._input_channel = input_channel
        self.pv_prefix.setText(self._input_channel)

        self.timer_labels = QTimer()
        self.timer_plot = QTimer()
        self.timer_labels.timeout.connect(self.update_labels)
        self.timer_plot.timeout.connect(self.update_image)

        # 5-panel layout: Transmission (3x3 left), Diffraction/Beam/Prediction (middle), Stitched (3x3 right)
        plot = pg.PlotItem()
        self.image_view = pg.ImageView(view=plot)
        self.viewer_layout.addWidget(self.image_view, 0, 0, 3, 3)

        plot2 = pg.PlotItem()
        self.image_view_2 = pg.ImageView(view=plot2)
        self.image_view_2.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view_2.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
        try:
            self.image_view_2.getView().getViewBox().invertY(True)
        except Exception:
            pass
        self.viewer_layout.addWidget(self.image_view_2, 0, 3, 1, 1)

        plot3 = pg.PlotItem()
        self.image_view_3 = pg.ImageView(view=plot3)
        self.image_view_3.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view_3.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
        try:
            self.image_view_3.getView().getViewBox().invertY(True)
        except Exception:
            pass
        self.viewer_layout.addWidget(self.image_view_3, 1, 3, 1, 1)

        plot4 = pg.PlotItem()
        self.image_view_4 = pg.ImageView(view=plot4)
        self.image_view_4.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view_4.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
        try:
            self.image_view_4.getView().getViewBox().invertY(True)
        except Exception:
            pass
        self.viewer_layout.addWidget(self.image_view_4, 2, 3, 1, 1)

        plot5 = pg.PlotItem()
        self.image_view_5 = pg.ImageView(view=plot5)
        self.image_view_5.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view_5.view.getAxis('bottom').setLabel(text='SizeX [pixels]')
        try:
            self.image_view_5.getView().getViewBox().invertY(True)
        except Exception:
            pass
        self.viewer_layout.addWidget(self.image_view_5, 0, 4, 3, 3)
        self.viewer_layout.setRowStretch(0, 1)
        self.viewer_layout.setRowStretch(1, 1)
        self.viewer_layout.setRowStretch(2, 1)
        self.viewer_layout.setColumnStretch(0, 3)
        self.viewer_layout.setColumnStretch(3, 1)
        self.viewer_layout.setColumnStretch(4, 3)

        self.image_view.view.getAxis('left').setLabel(text='SizeY [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='SizeX [pixels]')

        # Log checkbox for diffraction panel only
        self.log_image_1 = QCheckBox('Log (diffraction)')
        img_layout = getattr(self, 'image_layout', None) or \
                     (getattr(self, 'formLayoutWidget', None) and self.formLayoutWidget.layout())
        if img_layout is not None:
            img_layout.insertRow(5, '', self.log_image_1)
        self.log_image_1.setChecked(True)
        self.log_image_1.stateChanged.connect(self.update_image)

        # Magma colormap for all five ImageViews
        self._vit_colormap = None
        try:
            self._vit_colormap = pg.colormap.get('magma')
        except Exception:
            try:
                self._vit_colormap = pg.colormap.get('magma', source='colorcet')
            except Exception:
                pass

        _vit_views = (self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5)
        if self._vit_colormap is not None:
            for view in _vit_views:
                if hasattr(view, 'setColorMap'):
                    view.setColorMap(self._vit_colormap)
                elif hasattr(view, 'imageItem'):
                    lut = self._vit_colormap.getLookupTable(nPts=256)
                    if lut is not None:
                        view.imageItem.setLookupTable(lut)

        # Hide non-VIT buttons
        self.btn_hkl_viewer.setVisible(False)
        self.btn_save_caches.setVisible(False)
        self.btn_analysis_window.setVisible(False)
        self.btn_Stats1.setVisible(False)
        self.btn_Stats2.setVisible(False)
        self.btn_Stats3.setVisible(False)
        self.btn_Stats4.setVisible(False)
        self.btn_Stats5.setVisible(False)
        self.display_rois.setVisible(False)

        # Scale bar state
        self._vit_scale_bar_line = None
        self._vit_scale_bar_text = None
        self._vit_scale_bar_added = False
        self._vit_last_autoscale_log_state = None

        # Signal connections
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.pv_prefix.textChanged.connect(self.update_pv_prefix)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.rbtn_C.clicked.connect(self.c_ordering_clicked)
        self.rbtn_F.clicked.connect(self.f_ordering_clicked)
        self.rotate90degCCW.clicked.connect(self.rotation_count)
        self.log_image.clicked.connect(self.update_image)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.chk_transpose.stateChanged.connect(self.transpose_image_checked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)

        btn_autoscale = getattr(self, "btn_autoscale", None)
        if btn_autoscale is None:
            btn_autoscale = QPushButton("Autoscale (5-95%)")
            btn_autoscale.clicked.connect(self.apply_autoscale)
            if hasattr(self, "formLayoutWidget_6") and self.formLayoutWidget_6.layout() is not None:
                self.formLayoutWidget_6.layout().addRow(btn_autoscale)
            else:
                parent = self.min_setting_val.parent()
                if parent is not None and parent.layout() is not None:
                    parent.layout().addWidget(btn_autoscale)
        else:
            btn_autoscale.clicked.connect(self.apply_autoscale)

        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)

    # ---- VIT helpers ----

    def _get_vit_view_levels(self, view) -> tuple:
        if view is None:
            return None
        try:
            if getattr(view, 'ui', None) is not None and hasattr(view.ui, 'histogram'):
                return view.ui.histogram.getLevels()
            if hasattr(view, 'imageItem') and view.imageItem is not None:
                lev = getattr(view.imageItem, 'levels', None)
                if lev is not None and len(lev) == 2:
                    return tuple(lev)
                if callable(getattr(view.imageItem, 'getLevels', None)):
                    return view.imageItem.getLevels()
        except Exception:
            pass
        return None

    def _apply_vit_colormap(self) -> None:
        if not getattr(self, '_vit_colormap', None):
            return
        for view in (self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5):
            if view is None:
                continue
            try:
                if hasattr(view, 'setColorMap'):
                    view.setColorMap(self._vit_colormap)
                elif hasattr(view, 'imageItem'):
                    lut = self._vit_colormap.getLookupTable(nPts=256)
                    if lut is not None:
                        view.imageItem.setLookupTable(lut)
            except Exception:
                pass

    def _add_vit_scale_bar(self, height: int, width: int, view=None) -> None:
        if getattr(self, "_vit_scale_bar_added", False):
            return
        target = view if view is not None else self.image_view_5
        if target is None:
            return
        try:
            pixel_size_m = float(os.environ.get("VIT_STITCH_PIXEL_SIZE", "6.89e-9"))
            bar_nm = 1000.0  # 1 um
            bar_px = (bar_nm * 1e-9) / pixel_size_m
            if bar_px > 0.45 * width:
                bar_nm = 500.0
                bar_px = (bar_nm * 1e-9) / pixel_size_m
            if bar_px < 20:
                bar_nm = 1000.0
                bar_px = (bar_nm * 1e-9) / pixel_size_m
            margin = 20
            y_bar = height - 1 - margin
            x0, x1 = margin, margin + int(bar_px)
            color = "#00ffff"
            pen = pg.mkPen(color, width=2)
            self._vit_scale_bar_line = pg.PlotDataItem(x=[x0, x1], y=[y_bar, y_bar], pen=pen)
            self._vit_scale_bar_line.setZValue(100)
            target.view.addItem(self._vit_scale_bar_line)
            bar_label = "1 µm" if bar_nm >= 1000 else f"{bar_nm:.0f} nm"
            self._vit_scale_bar_text = pg.TextItem(text=bar_label, color=color, anchor=(0, 1))
            self._vit_scale_bar_text.setZValue(100)
            self._vit_scale_bar_text.setPos(x0, y_bar - 4)
            target.view.addItem(self._vit_scale_bar_text)
            self._vit_scale_bar_added = True
        except Exception:
            self._vit_scale_bar_added = True

    # ---- Timers ----

    def start_timers(self) -> None:
        if self.reader is not None and self.reader.channel.isMonitorActive():
            self.timer_labels.start(int(1000 / 100))
            self.timer_plot.start(int(1000 / self.plotting_frequency.value()))

    def stop_timers(self) -> None:
        self.timer_plot.stop()
        self.timer_labels.stop()

    # ---- Pixel ordering ----

    def set_pixel_ordering(self) -> None:
        if self.reader is not None:
            if self.rbtn_C.isChecked():
                self.reader.pixel_ordering = 'C'
                self.reader.image_is_transposed = True
            elif self.rbtn_F.isChecked():
                self.reader.pixel_ordering = 'F'
                self.image_is_transposed = False
                self.reader.image_is_transposed = False

    def c_ordering_clicked(self) -> None:
        if self.reader is not None:
            self.reader.pixel_ordering = 'C'
            self.reader.image_is_transposed = True

    def f_ordering_clicked(self) -> None:
        if self.reader is not None:
            self.reader.pixel_ordering = 'F'
            self.image_is_transposed = False
            self.reader.image_is_transposed = False

    # ---- Connection ----

    def start_live_view_clicked(self) -> None:
        try:
            self.stop_timers()
            for view in (self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5):
                view.clear()
            self._vit_last_autoscale_log_state = None
            self._vit_scale_bar_added = False

            if self.reader is None:
                self.reader = PVAReader(input_channel=self._input_channel)
            else:
                if self.reader.channel.isMonitorActive():
                    self.reader.stop_channel_monitor()
                del self.reader
                self.reader = PVAReader(input_channel=self._input_channel)

            self.set_pixel_ordering()
            self.transpose_image_checked()
            self.reader.start_channel_monitor()
            self.start_timers()
        except Exception as e:
            print(f'Failed to Connect to {self._input_channel}: {e}')
            for view in (self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5):
                view.clear()
            if getattr(self, 'reader', None) is not None:
                del self.reader
            self.reader = None
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

    def stop_live_view_clicked(self) -> None:
        if self.reader is not None:
            if self.reader.channel.isMonitorActive():
                self.reader.stop_channel_monitor()
            self.stop_timers()
            self.provider_name.setText('N/A')
            self.is_connected.setText('Disconnected')

    # ---- UI state ----

    def freeze_image_checked(self) -> None:
        if self.reader is not None:
            if self.freeze_image.isChecked():
                self.stop_timers()
            else:
                self.start_timers()

    def transpose_image_checked(self) -> None:
        if self.chk_transpose.isChecked():
            self.image_is_transposed = True
        else:
            self.image_is_transposed = False

    def reset_first_plot(self) -> None:
        self.first_plot = True

    def rotation_count(self) -> None:
        self.rot_num = next(rot_gen)

    def update_pv_prefix(self) -> None:
        self._input_channel = self.pv_prefix.text()

    # ---- Mouse tracking ----

    def update_mouse_pos(self, pos) -> None:
        mouse_point = self.image_view.getView().mapSceneToView(pos)
        self.mouse_x = int(mouse_point.x())
        self.mouse_y = int(mouse_point.y())
        self.update_mouse_labels()

    def update_mouse_labels(self) -> None:
        self.mouse_pos_x.setText(str(self.mouse_x))
        self.mouse_pos_y.setText(str(self.mouse_y))
        if self.image is not None:
            if 0 <= self.mouse_x < self.image.shape[1] and 0 <= self.mouse_y < self.image.shape[0]:
                self.mouse_pos_val.setText(f"{self.image[self.mouse_y, self.mouse_x]:.4f}")

    # ---- Labels ----

    def update_labels(self) -> None:
        if self.reader is not None:
            self.provider_name.setText(self.reader.pva_prefix)
            self.is_connected.setText('Connected')
            self.pv_col.setText(str(self.reader.shape[1]) if len(self.reader.shape) >= 2 else 'N/A')
            self.pv_row.setText(str(self.reader.shape[0]) if len(self.reader.shape) >= 2 else 'N/A')
            self.pv_data_type.setText(str(self.reader.data_type) if self.reader.data_type else 'N/A')
            self.pv_timestamp.setText(str(self.reader.timestamp) if self.reader.timestamp else 'N/A')
            self.pv_frame_recv.setText(str(self.reader.frames_received))
            self.pv_frame_miss.setText(str(self.reader.frames_missed))

    # ---- Image update (5-panel VIT) ----

    def update_image(self) -> None:
        if self.reader is None:
            return
        self.call_id_plot += 1
        vit_panels = getattr(self.reader, 'vit_panels', None)

        if vit_panels is not None and len(vit_panels) == 5:
            panels = [np.asarray(p, dtype=np.float32).copy() for p in vit_panels]
            panels[1] = np.maximum(panels[1], 0.0)
            log_diffraction = self.log_image_1.isChecked() if self.log_image_1 else True
            views = [self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5]

            for i, p in enumerate(panels):
                if i == 1:
                    p = np.transpose(p)
                if i in (0, 2, 4):
                    p = np.transpose(p)
                p = np.transpose(p) if self.image_is_transposed else p
                p = np.rot90(p, k=self.rot_num)
                if i == 4:
                    p = p[:, ::-1].copy()
                if i == 1 and log_diffraction:
                    p = np.maximum(p, 0)
                    p = np.log10(p + 1e-10)
                    p = np.maximum(p, 0.0)
                panels[i] = p

            self._add_vit_scale_bar(panels[4].shape[0], panels[4].shape[1], view=self.image_view_5)

            last = getattr(self, "_vit_last_autoscale_log_state", None)
            run_for_panel = [True] * 5
            if last is not None and last != log_diffraction:
                run_for_panel[1] = True
            if last is None or last != log_diffraction:
                self._apply_autoscale_vit(panels, views, run_for_panel=run_for_panel)
                self._vit_last_autoscale_log_state = log_diffraction

            for i, p in enumerate(panels):
                view = views[i]
                pmin, pmax = float(np.min(p)), float(np.max(p))
                levels = self._get_vit_view_levels(view)
                if levels is not None and len(levels) == 2 and levels[1] > levels[0]:
                    view.setImage(p, autoRange=False, autoLevels=False, levels=levels, autoHistogramRange=False)
                elif pmin == pmax:
                    levels = (pmin, pmin + 1.0) if np.isfinite(pmin) else (0.0, 1.0)
                    view.setImage(p, autoRange=False, autoLevels=False, levels=levels, autoHistogramRange=False)
                else:
                    view.setImage(p, autoRange=False, autoLevels=False, levels=(pmin, pmax), autoHistogramRange=False)
                view.setVisible(True)

            for i, v in enumerate(views):
                try:
                    ar = panels[i].shape[1] / float(max(panels[i].shape[0], 1))
                    v.view.setAspectLocked(lock=True, ratio=ar)
                except Exception:
                    pass

            self.image = panels[1]
            mn, mx = np.min(panels[1]), np.max(panels[1])
            self.min_px_val.setText(f"{mn:.2f}")
            self.max_px_val.setText(f"{mx:.2f}")

    # ---- Autoscale ----

    def _percentile_levels(self, intensities: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> tuple:
        intensities = np.asarray(intensities).flatten()
        intensities = intensities[np.isfinite(intensities)]
        if len(intensities) == 0:
            return (0.0, 1.0)
        data_min = float(np.min(intensities))
        data_max = float(np.max(intensities))
        if data_min < -9.0:
            population = intensities[intensities > -9.0]
            if len(population) >= 100:
                intensities = population
        lo = float(np.percentile(intensities, p_lo))
        hi = float(np.percentile(intensities, p_hi))
        if data_min < lo:
            lo = data_min
        if hi < data_max:
            hi = data_max
        if lo >= hi:
            hi = lo + 1.0
        return (lo, hi)

    def _apply_autoscale_vit(self, panels: list, views: list, run_for_panel: list = None) -> None:
        if run_for_panel is None:
            run_for_panel = [True] * len(panels)
        for i, p in enumerate(panels):
            if i >= len(run_for_panel) or i >= len(views) or not run_for_panel[i]:
                continue
            intensities = p.flatten()
            intensities = intensities[np.isfinite(intensities)]
            if len(intensities) == 0:
                continue
            lo, hi = self._percentile_levels(intensities)
            views[i].setLevels(lo, hi)
        if len(panels) > 1 and len(run_for_panel) > 1 and run_for_panel[1]:
            intensities = panels[1].flatten()
            intensities = intensities[np.isfinite(intensities)]
            if len(intensities) > 0:
                lo, hi = self._percentile_levels(intensities)
                self.min_setting_val.blockSignals(True)
                self.max_setting_val.blockSignals(True)
                self.min_setting_val.setValue(lo)
                self.max_setting_val.setValue(hi)
                self.min_setting_val.blockSignals(False)
                self.max_setting_val.blockSignals(False)

    def apply_autoscale(self) -> None:
        vit_panels = getattr(self.reader, "vit_panels", None) if self.reader else None
        if vit_panels is not None and len(vit_panels) == 5:
            panels = [np.asarray(p, dtype=np.float32).copy() for p in vit_panels]
            panels[1] = np.maximum(panels[1], 0.0)
            log_diffraction = self.log_image_1.isChecked() if self.log_image_1 else True
            views = [self.image_view, self.image_view_2, self.image_view_3, self.image_view_4, self.image_view_5]
            for i, p in enumerate(panels):
                if i == 1:
                    p = np.transpose(p)
                if i in (0, 2, 4):
                    p = np.transpose(p)
                p = np.transpose(p) if self.image_is_transposed else p
                p = np.rot90(p, k=self.rot_num)
                if i == 4:
                    p = p[:, ::-1].copy()
                if i == 1 and log_diffraction:
                    p = np.maximum(p, 0)
                    p = np.log10(p + 1e-10)
                    p = np.maximum(p, 0.0)
                panels[i] = p
            self._apply_autoscale_vit(panels, views)
            self._vit_last_autoscale_log_state = log_diffraction

    def update_min_max_setting(self) -> None:
        min_ = self.min_setting_val.value()
        max_ = self.max_setting_val.value()
        self.image_view_2.setLevels(min_, max_)

    # ---- Cleanup ----

    def closeEvent(self, event):
        self.stop_timers()
        if self.reader is not None:
            try:
                if self.reader.channel.isMonitorActive():
                    self.reader.stop_channel_monitor()
            except Exception:
                pass
        super(VitViewerWindow, self).closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description='DashPVA VIT Viewer — Ptychography Stitching')
    parser.add_argument('--channel', default='vit:1:input_phase',
                        help='PVA channel name (default: vit:1:input_phase, 9ID uses vit:1:CSSI)')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = VitViewerWindow(input_channel=args.channel)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
