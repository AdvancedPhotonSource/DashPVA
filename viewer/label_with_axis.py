import sys
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QPointF
import numpy as np

class MyLabel(QLabel):
    def __init__(self, parent=None, shape=(0,0)):
        super(MyLabel, self).__init__(parent)
        self.m_pixmap = None
        self.left_margin = 10
        self.bottom_margin = 20
        self.shape = shape
    
    def setPixmap(self, pixmap):
    
        self.m_pixmap = pixmap
        super(MyLabel, self).setPixmap(pixmap)

    def paintEvent(self, event):
        super(MyLabel, self).paintEvent(event)
        painter = QPainter(self)
        if self.m_pixmap is not None:
            painter.drawPixmap(0, 0, self.width(), self.height(), self.m_pixmap)
            painter.setPen(QPen(Qt.white))
            # draw vertical line
            painter.drawLine(QPointF(10, 5), QPointF(10, self.height()-self.left_margin))
            # draw horizontal line
            painter.drawLine(QPointF(10, self.height() - self.bottom_margin), QPointF(self.width()-self.bottom_margin, self.height() - self.bottom_margin))

            # Draw axis ticks and numbers
            painter.setPen(QPen(Qt.white))
            painter.setFont(QFont('Arial', 10))

            num_ticks = 10
            
            percent = np.linspace(start=0, stop=1, num=num_ticks+1)

            # Draw vertical axis ticks and numbers
            for i in range(num_ticks + 1):
                y = i * (self.height() - 25) / num_ticks
                painter.drawLine(5, y+5, 15, y+5)
                painter.drawText(20, y + self.left_margin, f'{(self.shape[1]* (1 - percent[i])):.0f}')

            # Draw horizontal axis ticks and numbers
            for i in range(num_ticks + 1):
                x = i * (self.width() - 30)/ num_ticks + 10
                painter.drawLine(x, self.height() - 15, x, self.height() - 25)
                num_to_draw = f'{(self.shape[0] * percent[i]):.0f}'

                if len(num_to_draw) >=4:
                    painter.drawText(x - (8+len(num_to_draw)//2), self.height()-3, num_to_draw)
                else:
                    painter.drawText(x - 5, self.height() - 3, num_to_draw)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    label = MyLabel()
    label.setPixmap(QPixmap.fromImage(QImage(800, 800, QImage.Format_RGB888)))
    label.show()
    sys.exit(app.exec_())