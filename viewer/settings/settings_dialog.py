import sys
from PyQt5 import QtWidgets, uic


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        # Load the placeholder UI
        uic.loadUi('gui/settings/settings_dialog.ui', self)

        # Wire dialog buttons if present
        button_box = getattr(self, 'buttonBox', None)
        if button_box is not None:
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dlg = SettingsDialog()
    dlg.show()
    sys.exit(app.exec_())
