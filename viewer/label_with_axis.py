import sys
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont
from PyQt5.QtCore import Qt, QPointF, QRect
import numpy as np

class MyLabel(QLabel):
    def __init__(self, parent=None, side_length=0, img_resolution=(0,0), location=None):
        super(MyLabel, self).__init__(parent)
        self.m_pixmap = None
        # self.padding = 20
        self.location = location
        self.side_length = side_length
        self.w, self.h= img_resolution
        try:
            if not(self.location == 'left' or self.location == 'bottom'):
                raise ValueError('location must be either \"left\" or \"bottom\"')
        except ValueError as ve:
            print(ve)
    
    # def setPixmap(self, pixmap : QPixmap):
    #     self.m_pixmap = pixmap
    #     self.lbl_height = self.m_pixmap.height()
    #     self.lbl_width = self.m_pixmap.width()
    #     super(MyLabel, self).setPixmap(pixmap)

    def paintEvent(self, event):
        super(MyLabel, self).paintEvent(event)
        painter = QPainter(self)
        # if self.m_pixmap is not None:
            # painter.drawPixmap(0, 0, self.width(), self.height(), self.m_pixmap)
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont('Arial', 10))
        num_ticks = 4 # starts at 0, then shows 3 more ticks 
        percent = np.linspace(start=0, stop=1, num=num_ticks+1)

        if self.location == "left":
            # self.setGeometry(ax=50, ay=self.side_length)
            self.setFixedWidth(50)
            self.setFixedHeight(self.side_length)
            painter.fillRect(QRect(0,0,70, self.side_length), Qt.black)
            # draw vertical line
            painter.drawLine(QPointF(40, 0), QPointF(40, self.side_length))
            for i in range(num_ticks + 1):
                if i == num_ticks:
                    y = i * (self.side_length-1) / num_ticks
                else:
                    y = i * (self.side_length) / num_ticks
                # draw Tick   
                painter.drawLine(35, y, 45, y)
                # draw Text
                if i == 0:
                    painter.drawText(QPointF(15, y+9),  f'{(self.h * (1 - percent[i])):.0f}')
                elif 0 < i < num_ticks:
                    painter.drawText(QPointF(15, y+5),  f'{(self.h * (1 - percent[i])):.0f}')
                else:
                    painter.drawText(QPointF(15, y),  f'{(self.h * (1 - percent[i])):.0f}')
        elif self.location == 'bottom':
            #  self.setGeometry(ax=50, ay=self.side_length)
            self.setFixedWidth(self.side_length)
            self.setFixedHeight(50)
            painter.fillRect(QRect(0,0,self.side_length, 50), Qt.black)
            # draw horizontal line
            painter.drawLine(QPointF(0, 10), QPointF(self.side_length, 10))
            for i in range(num_ticks + 1):
                if i == num_ticks:
                    x = i * (self.side_length-1) / num_ticks
                else:
                    x = i * (self.side_length) / num_ticks
                # draw Tick   
                painter.drawLine(x, 5, x, 15)
                num_to_draw = f'{(self.h * (percent[i])):.0f}'
                # draw Text
                if i == 0:
                    painter.drawText(QPointF(x, 30), num_to_draw)
                elif 0 < i < num_ticks:
                    painter.drawText(QPointF(x-len(num_to_draw)*3, 30),  num_to_draw)
                else:
                    painter.drawText(QPointF(x-len(num_to_draw)*8, 30), num_to_draw)

        # painter.setPen(QPen(Qt.white))
        # draw vertical line
        # painter.drawLine(QPointF(10, 5), QPointF(10, self.height()-self.left_margin))
        # draw horizontal line
        # painter.drawLine(QPointF(10, self.height() - self.bottom_margin), QPointF(self.width()-self.bottom_margin, self.height() - self.bottom_margin))

        # # Draw axis ticks and numbers
        # painter.setPen(QPen(Qt.white))
        # painter.setFont(QFont('Arial', 10))

        # num_ticks = 4 # starts at 0, then shows 3 more ticks 
        
        # percent = np.linspace(start=0, stop=1, num=num_ticks+1)

        # Draw vertical axis ticks and numbers
        # for i in range(num_ticks + 1):
        #     y = i * (self.height() - 25) / num_ticks
        #     painter.drawLine(5, y+5, 15, y+5)
        #     painter.drawText(20, y + self.left_margin, f'{(self.shape[1]* (1 - percent[i])):.0f}')

        # # Draw horizontal axis ticks and numbers
        # for i in range(num_ticks + 1):
        #     x = i * (self.width() - 30)/ num_ticks + 10
        #     painter.drawLine(x, self.height() - 15, x, self.height() - 25)
        #     num_to_draw = f'{(self.shape[0] * percent[i]):.0f}'

        #     if len(num_to_draw) >=4:
        #         painter.drawText(x - (8+len(num_to_draw)//2), self.height()-3, num_to_draw)
        #     else:
        #         painter.drawText(x - 5, self.height() - 3, num_to_draw)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    label = QLabel()
    label.setPixmap(QPixmap.fromImage(QImage(800, 800, QImage.Format_RGB888)),)
    label.show()
    my_lbl = MyLabel(location='bottom')
    my_lbl.side_length = label.width()
    my_lbl.w, my_lbl.h = (25,25)
    my_lbl.show()
    sys.exit(app.exec_())