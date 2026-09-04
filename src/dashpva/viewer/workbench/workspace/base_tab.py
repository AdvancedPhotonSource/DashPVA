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

from PyQt5 import uic
from PyQt5.QtWidgets import QWidget


class BaseTab(QWidget):
    """
    Base class for all tabs in the Workbench.
    Provides common functionality and a consistent interface.
    """
    def __init__(self, ui_file, parent=None, main_window=None, title=""):
        super().__init__(parent)
        self.main_window = main_window
        self.title = title
        uic.loadUi(ui_file, self)
        self.setObjectName(self.__class__.__name__) # Set object name for easier identification
        self.setup()

    def setup(self):
        self.main_window.tabWidget_analysis.addTab(self, self.title)

    def on_tab_selected(self):
        """
        Called when this tab is selected.
        Can be overridden by subclasses to perform tab-specific actions.
        """
        pass

    def on_tab_deselected(self):
        """
        Called when this tab is deselected.
        Can be overridden by subclasses to perform tab-specific actions.
        """
        pass

    def update_data(self, data_path: str):
        """
        Called when new data is loaded or the selected dataset changes.
        Subclasses should implement this to update their display.
        """
        raise NotImplementedError

    def clear_data(self):
        """
        Called when the HDF5 file is closed or cleared.
        Subclasses should implement this to clear their display.
        """
        raise NotImplementedError

    def get_tab_name(self) -> str:
        """
        Returns the display name for the tab.
        """
        return self.__class__.__name__.replace('Tab', '') # Default to class name without 'Tab'
