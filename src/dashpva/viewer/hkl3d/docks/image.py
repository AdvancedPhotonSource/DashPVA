from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from dashpva.viewer.core.docks.base_dock import BaseDock


class ImageDock(BaseDock):

    def __init__(self, main_window=None, show: bool = True):
        super().__init__(title="Image", main_window=main_window,
                         segment_name="hkl", dock_area=Qt.RightDockWidgetArea, show=show)
        self._build()

    def _build(self):
        container = QWidget()
        container.setMaximumWidth(380)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Pixel order row
        order_row = QHBoxLayout()
        order_row.addWidget(QLabel("Image Pixel Order:"))
        self.rbtn_C = QRadioButton("C")
        self.rbtn_F = QRadioButton("Fortran")
        self.rbtn_F.setChecked(True)
        self._pixel_order_group = QButtonGroup(container)
        self._pixel_order_group.addButton(self.rbtn_C)
        self._pixel_order_group.addButton(self.rbtn_F)
        order_row.addWidget(self.rbtn_C)
        order_row.addWidget(self.rbtn_F)
        order_row.addStretch()
        layout.addLayout(order_row)

        # Log image + reset camera row
        tools_row = QHBoxLayout()
        self.log_image = QCheckBox("Log Image")
        self.btn_reset_camera = QPushButton("Reset Camera")
        tools_row.addWidget(self.log_image)
        tools_row.addWidget(self.btn_reset_camera)
        layout.addLayout(tools_row)

        # Action buttons
        self.btn_3d_slice_window = QPushButton("Open Slice 3D Window")

        self.btn_plot_cache = QPushButton("Plot Cache")
        self.btn_plot_cache.setObjectName("btn_plot_cache")

        self.btn_save_h5 = QPushButton("Save Cache")
        self.btn_save_h5.setObjectName("btn_save_h5")
        self.btn_save_h5.setMinimumHeight(50)

        layout.addWidget(self.btn_3d_slice_window)
        layout.addWidget(self.btn_plot_cache)
        layout.addWidget(self.btn_save_h5)
        layout.addStretch()

        self.setWidget(container)
