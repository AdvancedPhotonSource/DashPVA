from PyQt5 import uic
from PyQt5.QtWidgets import QDialog


class HKLControlsDialog(QDialog):
    """
    Modeless dialog that encapsulates Slice and Camera controls for the HKL 3D viewer.
    Wires UI signals to methods on the main HKL3DSliceWindow instance and updates small state
    variables (e.g., _zoom_step, _cam_pos_selection) without the main window directly reading
    dialog widgets.
    """
    def __init__(self, main):
        super().__init__(parent=main)
        self.main = main
        uic.loadUi('gui/controls/hkl_controls_dialog.ui', self)
        # Initialize slice orientation and custom normal from main
        try:
            if hasattr(self.main, '_slice_orientation_selection') and self.main._slice_orientation_selection:
                self.cbSliceOrientation.setCurrentText(str(self.main._slice_orientation_selection))
        except Exception:
            pass
        try:
            cn = getattr(self.main, '_custom_normal', [0.0, 0.0, 1.0])
            self.sbNormH.setValue(float(cn[0]))
            self.sbNormK.setValue(float(cn[1]))
            self.sbNormL.setValue(float(cn[2]))
        except Exception:
            pass

        # Camera controls wiring
        try:
            self.btnZoomIn.clicked.connect(self.main.zoom_in)
        except Exception:
            pass
        try:
            self.btnZoomOut.clicked.connect(self.main.zoom_out)
        except Exception:
            pass
        try:
            self.btnResetCamera.clicked.connect(self.main.reset_camera)
        except Exception:
            pass
        try:
            # Keep local state in main; avoid main reading this widget directly
            self.sbZoomStep.valueChanged.connect(self._on_zoom_step_changed)
            # Initialize spinbox with main's current zoom step if available
            if hasattr(self.main, '_zoom_step'):
                try:
                    self.sbZoomStep.setValue(float(self.main._zoom_step))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            # Camera preset selection: update main's state; execution triggered by Set button
            self.cbSetCamPos.currentTextChanged.connect(self._on_cam_pos_changed)
            self.btnSetCamPos.clicked.connect(self.main.set_camera_position)
        except Exception:
            pass
        try:
            self.btnHKView.clicked.connect(lambda: self.main._apply_cam_preset_button('HK(xy)'))
        except Exception:
            pass
        try:
            self.btnKLView.clicked.connect(lambda: self.main._apply_cam_preset_button('KL(yz)'))
        except Exception:
            pass
        try:
            self.btnHLView.clicked.connect(lambda: self.main._apply_cam_preset_button('HL(xz)'))
        except Exception:
            pass
        try:
            self.btnViewSliceNormal.clicked.connect(self.main.view_slice_normal)
        except Exception:
            pass

        # Slice controls wiring
        try:
            self.sbSliceTranslateStep.valueChanged.connect(self.main._on_translate_step_changed)
        except Exception:
            pass
        try:
            self.sbSliceRotateStep.valueChanged.connect(self.main._on_rotate_step_changed)
        except Exception:
            pass
        try:
            self.cbSliceOrientation.currentIndexChanged.connect(self._on_slice_orientation_changed)
        except Exception:
            pass
        try:
            self.sbNormH.editingFinished.connect(self._on_custom_normal_spinboxes_changed)
            self.sbNormK.editingFinished.connect(self._on_custom_normal_spinboxes_changed)
            self.sbNormL.editingFinished.connect(self._on_custom_normal_spinboxes_changed)
        except Exception:
            pass

        # Translate buttons
        try:
            self.btnSliceUpNormal.clicked.connect(lambda: self.main.nudge_along_normal(+1))
        except Exception:
            pass
        try:
            self.btnSliceDownNormal.clicked.connect(lambda: self.main.nudge_along_normal(-1))
        except Exception:
            pass
        try:
            self.btnSlicePosH.clicked.connect(lambda: self.main.nudge_along_axis('H', +1))
            self.btnSliceNegH.clicked.connect(lambda: self.main.nudge_along_axis('H', -1))
        except Exception:
            pass
        try:
            self.btnSlicePosK.clicked.connect(lambda: self.main.nudge_along_axis('K', +1))
            self.btnSliceNegK.clicked.connect(lambda: self.main.nudge_along_axis('K', -1))
        except Exception:
            pass
        try:
            self.btnSlicePosL.clicked.connect(lambda: self.main.nudge_along_axis('L', +1))
            self.btnSliceNegL.clicked.connect(lambda: self.main.nudge_along_axis('L', -1))
        except Exception:
            pass

        # Rotate buttons
        try:
            self.btnRotPlusH.clicked.connect(lambda: self.main.rotate_about_axis('H', +float(getattr(self.main, '_slice_rotate_step_deg', 1.0))))
            self.btnRotMinusH.clicked.connect(lambda: self.main.rotate_about_axis('H', -float(getattr(self.main, '_slice_rotate_step_deg', 1.0))))
        except Exception:
            pass
        try:
            self.btnRotPlusK.clicked.connect(lambda: self.main.rotate_about_axis('K', +float(getattr(self.main, '_slice_rotate_step_deg', 1.0))))
            self.btnRotMinusK.clicked.connect(lambda: self.main.rotate_about_axis('K', -float(getattr(self.main, '_slice_rotate_step_deg', 1.0))))
        except Exception:
            pass
        try:
            self.btnRotPlusL.clicked.connect(lambda: self.main.rotate_about_axis('L', +float(getattr(self.main, '_slice_rotate_step_deg', 1.0))))
            self.btnRotMinusL.clicked.connect(lambda: self.main.rotate_about_axis('L', -float(getattr(self.main, '_slice_rotate_step_deg', 1.0))))
        except Exception:
            pass
        try:
            self.btnResetSlice.clicked.connect(self.main._on_reset_slice)
        except Exception:
            pass

        # Dialog properties
        try:
            # Modeless by default; caller decides modality if needed
            self.setModal(False)
        except Exception:
            pass

    # ---------- Dialog-side slots updating main window state ----------
    def _on_zoom_step_changed(self, val: float):
        try:
            self.main._zoom_step = float(val)
        except Exception:
            self.main._zoom_step = 1.5

    def _on_cam_pos_changed(self, text: str):
        try:
            self.main._cam_pos_selection = str(text)
        except Exception:
            self.main._cam_pos_selection = None

    def _on_slice_orientation_changed(self, idx: int):
        # Update main state with current orientation selection then delegate
        try:
            text = self.cbSliceOrientation.currentText()
        except Exception:
            text = 'HK(xy)'
        try:
            self.main._slice_orientation_selection = text
        except Exception:
            pass
        try:
            self.main._on_orientation_changed(idx)
        except Exception:
            pass

    def _on_custom_normal_spinboxes_changed(self):
        # Update main state with current custom normal then delegate
        try:
            h = float(self.sbNormH.value())
            k = float(self.sbNormK.value())
            l = float(self.sbNormL.value())
            self.main._custom_normal = [h, k, l]
        except Exception:
            self.main._custom_normal = [0.0, 0.0, 1.0]
        try:
            self.main._on_custom_normal_changed()
        except Exception:
            pass

    # ---------- Focus helpers ----------
    def focus_camera_section(self):
        try:
            # Focus movements group; optionally scroll if a scroll area is added later
            self.gbCamMovements.setFocus()
        except Exception:
            pass

    def focus_slice_section(self):
        try:
            # Focus steps group for convenience
            self.gbSteps.setFocus()
        except Exception:
            pass
