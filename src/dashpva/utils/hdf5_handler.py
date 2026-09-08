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
HDF5 Handler — Qt signal wrapper around HDF5Writer.
All write logic lives in HDF5Writer; this class exists as a connectable QObject slot.
"""
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from dashpva.utils.log_manager import LogMixin
from dashpva.utils.pva_reader import PVAReader


class HDF5Handler(QObject, LogMixin):
    hdf5_writer_finished = pyqtSignal(str)

    def __init__(self, file_path: str = "", pva_reader: PVAReader = None):
        super(HDF5Handler, self).__init__()
        self.pva_reader = pva_reader
        self.file_path = file_path
        try:
            self.set_log_manager()
        except Exception:
            pass

    def load_data(self):
        raise NotImplementedError

    @pyqtSlot(bool, bool, bool, str)
    def save_to_h5(self, clear_caches: bool = True, write_temp: bool = True, write_output: bool = True, output_override: str = '') -> None:
        """Gateway slot — delegates to HDF5Writer.save_caches_to_h5. No h5py logic here.

        HDF5Writer is a plain class so it can be used from a pvaccess process
        without importing Qt; this re-wraps its completion callback as a signal.
        """
        from dashpva.utils.hdf5_writer import HDF5Writer
        try:
            writer = HDF5Writer(
                file_path="",
                pva_reader=self.pva_reader,
                on_finished=self.hdf5_writer_finished.emit,
            )
            writer.save_caches_to_h5(clear_caches, write_temp, write_output, output_override)
        except Exception as e:
            self.hdf5_writer_finished.emit(f"Failed to save: {e}")

    @pyqtSlot(bool, bool, bool, str)
    def save_caches_to_h5(self, clear_caches: bool = True, write_temp: bool = True, write_output: bool = True, output_override: str = '') -> None:
        """Alias matching HDF5Writer's method name, for callers that moved here."""
        self.save_to_h5(clear_caches, write_temp, write_output, output_override)

    def get_file_info(self):
        raise NotImplementedError
