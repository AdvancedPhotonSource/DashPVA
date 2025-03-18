from PyQt5.QtWidgets import QDialog
from PyQt5 import uic


class RoiStatsDialog(QDialog):
    def __init__(self, parent: 'QMainWindow', stats_text: str, timer: 'QTimer'): # type: ignore # needed so annotations don't show warnings
        super(RoiStatsDialog, self).__init__()
        """
        Pop up QDialog for additional stats within an ROI.

        KeyWord Args:
        parent (QObject) -- The main window that opened the dialog has data it grabs from
        stats_text (str) -- Text sent from the button pressed to get the right data from parent
        timer (QTimer) -- QTimer passed by parent(area detector) so that there aren't multiple timers being ran
        """
        uic.loadUi("gui/roi_stats_dialog.ui", self)
        self.setWindowTitle(f"{stats_text} Info")
        self.stats_text = stats_text
        self.parent = parent
        self.prefix = self.parent.reader.pva_prefix
        # Setting up clock for updating QDialog Labels
        self.timer_labels = timer
        self.timer_labels.timeout.connect(self.update_stats_labels)
        # self.timer_labels.start(1000/100)

    def update_stats_labels(self):
        """Uses dict.get(key, default_return) method in case a PV isn't monitored."""
        self.stats_total_value.setText(f"{self.parent.stats_data.get(f'{self.prefix}:{self.stats_text}:Total_RBV', 0.0)}")
        self.stats_min_value.setText(f"{self.parent.stats_data.get(f'{self.prefix}:{self.stats_text}:MinValue_RBV', 0.0)}")
        self.stats_max_value.setText(f"{self.parent.stats_data.get(f'{self.prefix}:{self.stats_text}:MaxValue_RBV', 0.0)}")
        self.stats_sigma_value.setText(f"{self.parent.stats_data.get(f'{self.prefix}:{self.stats_text}:Sigma_RBV', 0.0):.4f}")
        self.stats_mean_value.setText(f"{self.parent.stats_data.get(f'{self.prefix}:{self.stats_text}:MeanValue_RBV', 0.0):.4f}")

    def closeEvent(self, event):
        """
        An altered closeEvent so pop up is removed from memory when closed.
        
        Keyword Args:
        event -- closing event sent by dialog window
        """
        self.parent.stats_dialog[self.stats_text] = None
        super(RoiStatsDialog,self).closeEvent(event)