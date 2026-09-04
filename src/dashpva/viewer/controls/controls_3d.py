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

"""
3D Controls wiring for Workbench and other viewers.
Encapsulates signal connections for 3D-specific UI elements.
"""



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
