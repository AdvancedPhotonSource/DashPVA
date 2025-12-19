"""
3D Controls wiring for Workbench and other viewers.
Encapsulates signal connections for 3D-specific UI elements.
"""

from typing import Optional


class Controls3D:
    def __init__(self, main_window):
        self.main = main_window

    def setup(self) -> None:
        """Wire up 3D controls to main window handlers."""
        try:
            # Load data button
            if hasattr(self.main, 'btn_load_3d_data'):
                self.main.btn_load_3d_data.clicked.connect(self.main.load_3d_data)

            # Colormap selection
            if hasattr(self.main, 'cb_colormap_3d'):
                self.main.cb_colormap_3d.currentTextChanged.connect(self.main.on_3d_colormap_changed)

            # Visibility checkboxes
            if hasattr(self.main, 'cb_show_volume'):
                self.main.cb_show_volume.toggled.connect(self.main.toggle_3d_volume)
            if hasattr(self.main, 'cb_show_slice'):
                self.main.cb_show_slice.toggled.connect(self.main.toggle_3d_slice)
            if hasattr(self.main, 'cb_show_pointer'):
                self.main.cb_show_pointer.toggled.connect(self.main.toggle_3d_pointer)

            # Intensity spinboxes
            if hasattr(self.main, 'sb_min_intensity_3d'):
                self.main.sb_min_intensity_3d.editingFinished.connect(self.main.update_3d_intensity)
            if hasattr(self.main, 'sb_max_intensity_3d'):
                self.main.sb_max_intensity_3d.editingFinished.connect(self.main.update_3d_intensity)

            # Slice controls
            if hasattr(self.main, 'cb_slice_orientation'):
                self.main.cb_slice_orientation.currentTextChanged.connect(self.main.change_slice_orientation)
            if hasattr(self.main, 'btn_reset_slice'):
                self.main.btn_reset_slice.clicked.connect(self.main.reset_3d_slice)
        except Exception as e:
            try:
                self.main.update_status(f"Error setting up 3D connections: {e}")
            except Exception:
                pass
