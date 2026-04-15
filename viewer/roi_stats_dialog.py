from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import Qt
from PyQt5 import uic


# Stats-to-ROI color mapping: Stats1=Red, Stats2=Blue, Stats3=Green, Stats4=Pink
ROI_COLORS = {
    'Stats1': '#ff0000',
    'Stats2': '#0000ff',
    'Stats3': '#4CBB17',
    'Stats4': '#ff00ff',
}


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

        # Apply ROI color to window title and value labels
        color = ROI_COLORS.get(stats_text)
        if color:
            self.setStyleSheet(f'QDialog {{ color: {color}; }}')
            for label in [self.stats_total_value, self.stats_min_value,
                          self.stats_max_value, self.stats_sigma_value,
                          self.stats_mean_value]:
                label.setStyleSheet(f'color: {color};')
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        else:
            for label in [self.stats_total_value, self.stats_min_value,
                          self.stats_max_value, self.stats_sigma_value,
                          self.stats_mean_value]:
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

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
        self.parent.stats_dialogs[self.stats_text] = None
        super(RoiStatsDialog,self).closeEvent(event)