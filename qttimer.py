import sys
import time
import pvaccess as pva
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPlainTextEdit
from PyQt5 import uic

class PVA_Reader:

    def __init__(self, provider=pva.PVA, pva_name="dp-ADSim:Pva1:Image"):
        
        """variables needed for monitoring a connection"""
        self.provider = provider
        self.pva_name = pva_name
        self.channel = pva.Channel(pva_name, provider)

        """variables that will store pva data"""
        self.pva_object = None
        self.image = None
        self.attributes = {}
        self.timestamp = None
        self.pva_cache = []
        self.__last_array_id = None
        self.frames_missed = 0

    def callbackSuccess(self, pv):
        self.pva_object = pv
        self.pva_cache.append(pv)

    def callbackError(self, code):
        print('error %s' % code)

    def asyncGet(self):
        self.channel.asyncGet(self.callbackSuccess, self.callbackError)

    def calcFramesMissed(self, data):
        if data is not None:
            current_array_id = data['uniqueId']
            if self.__last_array_id is not None: #and zoomUpdate == False:
                id_diff = current_array_id - self.__last_array_id - 1
                self.frames_missed += id_diff if (id_diff > 0) else 0
            self.__last_array_id = current_array_id

    def getFramesMissed(self):
        return self.frames_missed
    
    def parsePvaNdattributes(self):
        if self.pva_object:
            obj_dict = self.pva_object.get()
        else:
            return

        attributes = {
            attr["name"]: [val for val in attr.get("value", "")] for attr in obj_dict.get("attribute", {})
        }

        for value in ["codec", "uniqueId", "uncompressedSize"]:
            if value in self.pva_object:
                attributes[value] = self.pva_object[value]
        
        self.attributes = attributes


    def pvaToImage(self):
        if self.pva_object is not None:
            if "dimension" in self.pva_object:
                shape = tuple([dim["size"] for dim in self.pva_object["dimension"]])
                image = np.array(self.pva_object["value"][0]["byteValue"])
                image = np.reshape(image, shape)
            else:
                image = None
            
            self.image = image
        else:
            print('pvaObject is none')


    def startChannelMonitor(self):
        self.channel.startMonitor()

    def stopChannelMonitor(self):
        self.channel.stopMonitor()

    def getPvaObjects(self):
        return self.pva_cache

    def getLastPvaObject(self):
        return self.pva_cache[-1]

    def getPvaImage(self):
        return self.image
    
    def getAttributesDict(self):
        return self.attributes


class ImageWindow(QMainWindow):
    def __init__(self): 
        super(ImageWindow, self).__init__()
        uic.loadUi('/home/beams0/JULIO.RODRIGUEZ/Desktop/Lab Software/channel_reader/imageshow.ui', self)
        self.setWindowTitle("Image Viewer")
        self.show()
        
        self.reader = PVA_Reader(pva.PVA, self.pv_prefix.text())
        self.last_unique_id = None
        self.call_id_poll = 0
        self.call_id_plot = 0
        self.total_frames_received = 0
        
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)

        self.timer_poll = QTimer()
        self.timer_poll.timeout.connect(self.async_get_and_process)
        self.timer_poll.start(int(1000/float(self.polling_frequency.text())))
        
        self.timer_plot = QTimer()
        self.timer_plot.timeout.connect(self.update_image)
        self.timer_plot.start(int(1000/float(self.plotting_frequency.text())))
        
        self.first_plot = True


    def start_live_view_clicked(self):
        self.timer_poll.start(int(1000/float(self.polling_frequency.text())))
        self.timer_plot.start(int(1000/float(self.plotting_frequency.text())))

    def stop_live_view_clicked(self):
        self.timer_poll.stop()
        self.timer_plot.stop()
        
    def async_get_and_process(self):
        self.reader.startChannelMonitor()
        time.sleep(0.1)
        log_text = f"\n{self.reader.provider} Channel Name = {self.reader.channel.getName()} Channel is connected = {self.reader.channel.isConnected()}"
        self.log_plain_text_edit.appendPlainText(log_text)
        self.reader.asyncGet()
        self.reader.stopChannelMonitor()
        self.call_id_poll +=1 
        self.total_frames_received += 1
        self.log_plain_text_edit.appendPlainText(f"Call id for Poll  :  {self.call_id_poll:d}")

    def update_image(self):
        self.call_id_plot +=1
        # pva_object = self.reader.pva_object 
        pva_object = self.reader.getLastPvaObject() #caching 
        if pva_object is not None:
            self.reader.parsePvaNdattributes()
            unique_id = self.reader.getAttributesDict().get("uniqueId")
            if unique_id != self.last_unique_id:
                self.reader.calcFramesMissed(pva_object)
                self.last_unique_id = unique_id

                self.reader.pvaToImage()
                image = self.reader.getPvaImage()

                if image is not None:
                    
                    if len(image.shape) == 2:
                        if self.first_plot:
                            min_level, max_level = np.min(image), np.max(image)
                            self.image_view.setImage(image, autoRange=False, autoLevels=False, levels=(min_level, max_level))
                            self.first_plot = False
                        else:
                            self.image_view.setImage(image, autoRange=False, autoLevels=False)
                        self.log_plain_text_edit.appendPlainText(f"Total Frames Received: {self.total_frames_received:d}")
                        self.log_plain_text_edit.appendPlainText(f"Total Frames Missed  :  {self.reader.frames_missed:d}")
                        self.log_plain_text_edit.appendPlainText(f"Call id for Plot  :  {self.call_id_plot:d}")
        
              
if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    window = ImageWindow()

    sys.exit(app.exec_())