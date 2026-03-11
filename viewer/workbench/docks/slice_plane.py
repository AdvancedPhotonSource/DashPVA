from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget

from viewer.workbench.docks.base_dock import BaseDock


class SlicePlaneDock(BaseDock):
    """
    Slice Controls dock for manipulating the 3D slice plane and camera.
    Loads its UI from gui/workbench/docks/slice_plane.ui and wires signals
    into Workspace3D (tab_3d) methods.
    """
    def __init__(self, title: str = "Slice Controls", main_window=None, segment_name: str = "3d", dock_area: Qt.DockWidgetArea = Qt.LeftDockWidgetArea, show: bool = False):
        super().__init__(title=title, main_window=main_window, segment_name=segment_name, dock_area=dock_area, show=show)
        self._widget = None
        self._build()
        self._wire()

    def setup(self):
        # BaseDock handles docking and Windows->segment toggle registration
        super().setup()

    def _build(self):
        try:
            self._widget = QWidget(self)
            uic.loadUi('gui/workbench/docks/slice_plane.ui', self._widget)
            self.setWidget(self._widget)
        except Exception as e:
            # If UI fails to load, keep an empty widget to avoid crashing
            self._widget = QWidget(self)
            self.setWidget(self._widget)
            try:
                if hasattr(self.main_window, 'update_status'):
                    self.main_window.update_status(f"SlicePlaneDock UI load failed: {e}")
            except Exception:
                pass

    def _wire(self):
        mw = self.main_window
        if mw is None:
            return
        tab = getattr(mw, 'tab_3d', None)
        if tab is None:
            return
        w = self._widget
        try:
            # Steps
            if hasattr(w, 'sb_slice_translate_step'):
                w.sb_slice_translate_step.setValue(0.01)
                w.sb_slice_translate_step.valueChanged.connect(lambda v: setattr(tab, '_slice_translate_step', float(v)))
            if hasattr(w, 'sb_slice_rotate_step_deg'):
                w.sb_slice_rotate_step_deg.setValue(1.0)
                w.sb_slice_rotate_step_deg.valueChanged.connect(lambda v: setattr(tab, '_slice_rotate_step_deg', float(v)))

            # Orientation preset
            if hasattr(w, 'cb_slice_orientation'):
                w.cb_slice_orientation.currentTextChanged.connect(lambda txt: tab.set_plane_preset(str(txt)))

            # Custom normal spinboxes
            def _apply_custom_normal():
                try:
                    h = float(w.sb_norm_h.value()) if hasattr(w, 'sb_norm_h') else 0.0
                    k = float(w.sb_norm_k.value()) if hasattr(w, 'sb_norm_k') else 0.0
                    l = float(w.sb_norm_l.value()) if hasattr(w, 'sb_norm_l') else 1.0
                    if hasattr(tab, 'set_custom_normal'):
                        tab.set_custom_normal([h, k, l])
                    # If Custom preset selected, apply immediately
                    cur = str(w.cb_slice_orientation.currentText()) if hasattr(w, 'cb_slice_orientation') else ''
                    if cur.lower().startswith('custom'):
                        tab.set_plane_preset('Custom')
                except Exception:
                    pass
            for name in ('sb_norm_h', 'sb_norm_k', 'sb_norm_l'):
                spin = getattr(w, name, None)
                if spin is not None:
                    try:
                        spin.editingFinished.connect(_apply_custom_normal)
                    except Exception:
                        pass

            # Translate buttons
            if hasattr(w, 'btn_up_normal'):
                w.btn_up_normal.clicked.connect(lambda: tab.nudge_along_normal(+1))
            if hasattr(w, 'btn_down_normal'):
                w.btn_down_normal.clicked.connect(lambda: tab.nudge_along_normal(-1))
            if hasattr(w, 'btn_pos_h'):
                w.btn_pos_h.clicked.connect(lambda: tab.nudge_along_axis('H', +1))
            if hasattr(w, 'btn_neg_h'):
                w.btn_neg_h.clicked.connect(lambda: tab.nudge_along_axis('H', -1))
            if hasattr(w, 'btn_pos_k'):
                w.btn_pos_k.clicked.connect(lambda: tab.nudge_along_axis('K', +1))
            if hasattr(w, 'btn_neg_k'):
                w.btn_neg_k.clicked.connect(lambda: tab.nudge_along_axis('K', -1))
            if hasattr(w, 'btn_pos_l'):
                w.btn_pos_l.clicked.connect(lambda: tab.nudge_along_axis('L', +1))
            if hasattr(w, 'btn_neg_l'):
                w.btn_neg_l.clicked.connect(lambda: tab.nudge_along_axis('L', -1))

            # Rotate buttons use current rotate-step from tab
            if hasattr(w, 'btn_rot_plus_h'):
                w.btn_rot_plus_h.clicked.connect(lambda: tab.rotate_about_axis('H', +getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_minus_h'):
                w.btn_rot_minus_h.clicked.connect(lambda: tab.rotate_about_axis('H', -getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_plus_k'):
                w.btn_rot_plus_k.clicked.connect(lambda: tab.rotate_about_axis('K', +getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_minus_k'):
                w.btn_rot_minus_k.clicked.connect(lambda: tab.rotate_about_axis('K', -getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_plus_l'):
                w.btn_rot_plus_l.clicked.connect(lambda: tab.rotate_about_axis('L', +getattr(tab, '_slice_rotate_step_deg', 1.0)))
            if hasattr(w, 'btn_rot_minus_l'):
                w.btn_rot_minus_l.clicked.connect(lambda: tab.rotate_about_axis('L', -getattr(tab, '_slice_rotate_step_deg', 1.0)))

            # Reset
            if hasattr(w, 'btn_reset_slice'):
                w.btn_reset_slice.clicked.connect(tab.reset_slice)

            # Visibility
            if hasattr(w, 'cb_show_slice'):
                w.cb_show_slice.toggled.connect(lambda checked: tab.toggle_3d_slice(bool(checked)))
            # Show Points (main cloud)
            if hasattr(w, 'cb_show_points'):
                w.cb_show_points.toggled.connect(lambda checked: tab.toggle_3d_points(bool(checked)))

            # Camera
            if hasattr(w, 'cb_cam_preset'):
                w.cb_cam_preset.currentTextChanged.connect(lambda txt: tab.set_camera_position(str(txt)))
            if hasattr(w, 'btn_zoom_in'):
                w.btn_zoom_in.clicked.connect(tab.zoom_in)
            if hasattr(w, 'btn_zoom_out'):
                w.btn_zoom_out.clicked.connect(tab.zoom_out)
            if hasattr(w, 'btn_reset_camera'):
                w.btn_reset_camera.clicked.connect(tab.reset_camera)
            if hasattr(w, 'btn_view_slice_normal'):
                w.btn_view_slice_normal.clicked.connect(tab.view_slice_normal)

            # Initialize defaults
            try:
                if hasattr(w, 'cb_slice_orientation'):
                    w.cb_slice_orientation.setCurrentText('HK (xy)')
                if hasattr(w, 'cb_show_slice'):
                    # Mirror current Tab state if available by checking actor existence
                    w.cb_show_slice.setChecked(True)
                if hasattr(w, 'cb_show_points'):
                    w.cb_show_points.setChecked(True)
            except Exception:
                pass
        except Exception:
            pass
