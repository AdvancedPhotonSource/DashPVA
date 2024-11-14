import json
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFileDialog, QSizePolicy, QLabel, QFormLayout, QWidget, QFrame


class PVSetupDialog(QDialog):
    def __init__(self, parent, file_mode, path=None):
        super(PVSetupDialog,self).__init__(parent)
        uic.loadUi('gui/edit_add_config_dialog.ui',self)
        self.config_dict = {}
        self.path = path
        self.file_mode = file_mode

        self.form_widget = QWidget()
        self.config_layout = QFormLayout(parent=self.form_widget)
        self.config_layout.setLabelAlignment(Qt.AlignRight)
        self.scroll_area.setWidget(self.form_widget)
        self.scroll_area.setWidgetResizable(True)

        self.load_config()
        self.show()

    def save_file_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save File', 'pv_configs', '.json (*.json)')
        return path
    
    def load_config(self):
        if self.file_mode == 'w':
            return
        with open(self.path, "r") as config_json:
            self.config_dict: dict = json.load(config_json)
            for key, value in self.config_dict.items():
                # set the label part of the form widget
                label = QLabel(key + ':')
                label.setMinimumHeight(35)
                label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                # set the field part of the form widget
                field = QLabel(value)
                field.setMinimumHeight(35)
                field.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                field.setFrameShape(QFrame.Shape.Box)
                field.setFrameShadow(QFrame.Shadow.Sunken)
                
                self.config_layout.addRow(label, field)