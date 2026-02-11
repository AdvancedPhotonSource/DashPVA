from typing import Optional, Dict
import os
import h5py
from PyQt5.QtWidgets import QTreeWidgetItem
from viewer.workbench.workspace.base_tab import BaseTab

class WorkspaceInfo(BaseTab):
    """
    DataInfo tab/controller that encapsulates all file/dataset info functionality
    previously implemented directly on the WorkbenchWindow.

    This class does not construct new UI; instead it writes into the existing
    QTextEdit widgets exposed on the main_window (file_info_text, dataset_info_text)
    and uses main_window state (current_file_path, update_status, etc.).
    """
    def __init__(self, main_window):
        # BaseTab signature: (parent=None, main_window=None, title="")
        super().__init__(parent=None, main_window=main_window, title="Data Info")
        self.main_window = main_window

    # === Lifecycle ===
    def initialize(self):
        """Initialize the info panels with default messages."""
        try:
            self.main_window.file_info_text.setPlainText("Load an HDF5 file to view file information.")
            self.main_window.dataset_info_text.setPlainText("Select a dataset from the tree to view detailed information.")
        except Exception:
            pass

    def clear_data(self):
        """Clear info panels to defaults when the file is closed/cleared."""
        try:
            self.main_window.file_info_text.setPlainText("Load an HDF5 file to view file information.")
            self.main_window.dataset_info_text.setPlainText("Select a dataset from the tree to view detailed information.")
        except Exception:
            pass

    def update_data(self, data_path: str):
        """Called when selection changes; update dataset info panel.
        Delegates to update_file_info_with_dataset.
        """
        try:
            self.update_file_info_with_dataset(data_path)
        except Exception:
            pass

    # === File & Dataset Info ===
    def update_file_info_display(self, file_path: str, additional_info: Optional[Dict] = None):
        """
        Update the file information display with details about the current file.

        Args:
            file_path (str): Path to the current file or status message
            additional_info (dict): Additional information to display
        """
        if not hasattr(self.main_window, 'file_info_text'):
            return

        try:
            if file_path == "No file loaded" or not os.path.exists(file_path):
                # Show default message
                self.main_window.file_info_text.setPlainText("Load an HDF5 file to view file information.")
                return

            # Get file information
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            file_modified = os.path.getmtime(file_path)

            # Format file size
            if file_size < 1024:
                size_str = f"{file_size} bytes"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            elif file_size < 1024 * 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{file_size / (1024 * 1024 * 1024):.1f} GB"

            # Format modification time
            import datetime
            mod_time = datetime.datetime.fromtimestamp(file_modified).strftime("%Y-%m-%d %H:%M:%S")

            # Build info lines
            info_lines = []
            info_lines.append(f"File: {os.path.basename(file_path)}")
            info_lines.append(f"Path: {os.path.dirname(file_path)}")
            info_lines.append(f"Size: {size_str}")
            info_lines.append(f"Modified: {mod_time}")

            # HDF5 structure summary
            try:
                with h5py.File(file_path, 'r') as h5file:
                    # Count groups and datasets
                    def count_items(group, counts):
                        for key in group.keys():
                            item = group[key]
                            if isinstance(item, h5py.Group):
                                counts['groups'] += 1
                                count_items(item, counts)
                            elif isinstance(item, h5py.Dataset):
                                counts['datasets'] += 1

                    counts = {'groups': 0, 'datasets': 0}
                    count_items(h5file, counts)

                    info_lines.append("")
                    info_lines.append("HDF5 Structure:")
                    info_lines.append(f"Groups: {counts['groups']}")
                    info_lines.append(f"Datasets: {counts['datasets']}")

                    # File attributes if any
                    if h5file.attrs:
                        info_lines.append(f"File Attributes: {len(h5file.attrs)}")

            except Exception as e:
                info_lines.append("")
                info_lines.append(f"HDF5 Error: {str(e)}")

            # Additional info if provided
            if additional_info:
                info_lines.append("")
                info_lines.append("Selection Details:")
                for key, value in additional_info.items():
                    info_lines.append(f"{key}: {value}")

            # Update the file info tab
            self.main_window.file_info_text.setPlainText("\n".join(info_lines))

        except Exception as e:
            error_text = f"Error reading file information: {str(e)}"
            self.main_window.file_info_text.setPlainText(error_text)

    def update_file_info_with_dataset(self, dataset_path: str):
        """
        Update the file information display with details about the selected dataset.

        Args:
            dataset_path (str): Path to the selected dataset within the HDF5 file
        """
        file_path = getattr(self.main_window, 'current_file_path', None)
        if not file_path or not dataset_path:
            return

        try:
            with h5py.File(file_path, 'r') as h5file:
                if dataset_path in h5file:
                    item = h5file[dataset_path]

                    additional_info: Dict[str, str] = {}

                    if isinstance(item, h5py.Dataset):
                        # Dataset information
                        additional_info['Selected Dataset'] = dataset_path
                        additional_info['Shape'] = str(item.shape) if item.shape else "scalar"
                        additional_info['Data Type'] = str(item.dtype)
                        additional_info['Size'] = f"{item.size:,} elements" if item.size > 0 else "0 elements"

                        # Memory size estimation
                        if item.size > 0:
                            mem_size = item.size * item.dtype.itemsize
                            if mem_size < 1024:
                                mem_str = f"{mem_size} bytes"
                            elif mem_size < 1024 * 1024:
                                mem_str = f"{mem_size / 1024:.1f} KB"
                            elif mem_size < 1024 * 1024 * 1024:
                                mem_str = f"{mem_size / (1024 * 1024):.1f} MB"
                            else:
                                mem_str = f"{mem_size / (1024 * 1024 * 1024):.1f} GB"
                            additional_info['Memory Size'] = mem_str

                        # Dataset attributes
                        if item.attrs:
                            additional_info['Dataset Attributes'] = f"{len(item.attrs)} attributes"

                        # Compression info
                        if item.compression:
                            additional_info['Compression'] = item.compression
                            if item.compression_opts:
                                additional_info['Compression Level'] = str(item.compression_opts)

                    elif isinstance(item, h5py.Group):
                        # Group information
                        additional_info['Selected Group'] = dataset_path
                        additional_info['Contains'] = f"{len(item)} items"

                        # Count subgroups and datasets
                        subgroups = sum(1 for key in item.keys() if isinstance(item[key], h5py.Group))
                        subdatasets = sum(1 for key in item.keys() if isinstance(item[key], h5py.Dataset))

                        if subgroups > 0:
                            additional_info['Subgroups'] = str(subgroups)
                        if subdatasets > 0:
                            additional_info['Subdatasets'] = str(subdatasets)

                        # Group attributes
                        if item.attrs:
                            additional_info['Group Attributes'] = f"{len(item.attrs)} attributes"

                    # Update the display with additional dataset/group info
                    self.update_file_info_display(file_path, additional_info)

        except Exception as e:
            # Fall back to basic file info if dataset reading fails
            self.update_file_info_display(file_path, {'Error': f"Could not read dataset info: {str(e)}"})

    def show_dataset_info(self, item: QTreeWidgetItem):
        """
        Show information about the selected dataset without loading it.

        Args:
            item: QTreeWidgetItem representing the dataset
        """
        try:
            # Retrieve full dataset path from item
            full_path = item.data(0, 32)  # Qt.UserRole = 32
            file_path = getattr(self.main_window, 'current_file_path', None)
            if not full_path or not file_path:
                return

            # Get dataset information
            with h5py.File(file_path, 'r') as h5file:
                if full_path in h5file:
                    dataset = h5file[full_path]

                    if isinstance(dataset, h5py.Dataset):
                        # Show dataset information
                        shape_str = f"{dataset.shape}" if dataset.shape else "scalar"
                        dtype_str = str(dataset.dtype)
                        size_str = f"{dataset.size:,}" if dataset.size > 0 else "0"

                        info_text = (
                            f"Dataset: {full_path}\n"
                            f"Shape: {shape_str}\n"
                            f"Data Type: {dtype_str}\n"
                            f"Size: {size_str} elements\n\n"
                            f"Double-click to load into workspace"
                        )

                        if hasattr(self.main_window, 'dataset_info_text') and self.main_window.dataset_info_text is not None:
                            self.main_window.dataset_info_text.setPlainText(info_text)
                    else:
                        # It's a group
                        group_info = (
                            f"Group: {full_path}\n\nContains {len(dataset)} items\n\n"
                            f"Double-click to expand/collapse"
                        )
                        if hasattr(self.main_window, 'dataset_info_text') and self.main_window.dataset_info_text is not None:
                            self.main_window.dataset_info_text.setPlainText(group_info)
        except Exception as e:
            error_msg = f"Error reading dataset info: {str(e)}"
            if hasattr(self.main_window, 'dataset_info_text') and self.main_window.dataset_info_text is not None:
                self.main_window.dataset_info_text.setPlainText(error_msg)
