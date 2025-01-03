from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5 import uic

class ScanPlanDialog(QDialog):
    def __init__(self, ):
        super(ScanPlanDialog, self).__init__()
        uic.loadUi('gui/scan_plan_upload_dialog.ui', self)
        self.setWindowTitle('Scan Plan Window')
        self.x_positions = None
        self.y_positions = None
        self.download_loc = None

        self.btn_xpos.clicked.connect(self.open_x_positions)
        self.btn_ypos.clicked.connect(self.open_y_positions)
        self.btn_download.clicked.connect(self.find_download_loc)
        self.btn_accept_reject.accepted.connect(self.dialog_accepted)    

    def open_x_positions(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Get X Positions', '', '*.npy (*.npy)')
        self.le_xpos_path.setText(path)
    
    def open_y_positions(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Get Y Positions', '', '*.npy (*.npy)')
        self.le_ypos_path.setText(path)

    def find_download_loc(self):
        path = QFileDialog.getExistingDirectory(self, 'Analysis Data Save Location', '')
        self.le_download.setText(path)

    def dialog_accepted(self):
        self.x_positions = self.le_xpos_path.text()
        self.y_positions = self.le_ypos_path.text()
        self.download_loc = self.le_download.text()

        if self.x_positions != '' and self.y_positions != '' and self.download_loc != '':
            self.accept()
        else:
            self.reject()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = ScanPlanDialog()
    window.show()
    sys.exit(app.exec_())

        

    