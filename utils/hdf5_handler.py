"""
HDF5 Handler — Qt signal wrapper around HDF5Writer.
All write logic lives in HDF5Writer; this class exists as a connectable QObject slot.
"""
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from utils.pva_reader import PVAReader
from utils.log_manager import LogMixin


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
        """Gateway slot — delegates to HDF5Writer.save_caches_to_h5. No h5py logic here."""
        from utils.hdf5_writer import HDF5Writer
        try:
            writer = HDF5Writer(file_path="", pva_reader=self.pva_reader)
            writer.hdf5_writer_finished.connect(self.hdf5_writer_finished)
            writer.save_caches_to_h5(clear_caches, write_temp, write_output, output_override)
        except Exception as e:
            self.hdf5_writer_finished.emit(f"Failed to save: {e}")

    def get_file_info(self):
        raise NotImplementedError
