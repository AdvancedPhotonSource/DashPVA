import sys
import time
import pvaccess as pva
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

#from pyqtgraph import QtCore #provides timer important for polling

#TODO: implement polling of live data, ask Peco for assistance in setting up timer

#TODO: Have polling start and stop automatically based on timer

#TODO: have polling rate change with a variable


class PVA_Reader:

    def __init__(self, provider=pva.PVA, pva_name="dp-ADSim:Pva1:Image", timer=None):
        
        """variables needed for monitoring a connection"""
        self.provider = provider
        self.pva_name = pva_name
        self.channel = pva.Channel(pva_name, provider)
        #self.timer = timer

        """variables that will store pva data"""
        self.pva_object = None
        self.data = {}
        self.image = None
        self.attributes = {}
        self.timestamp = None
        self.pva_cache = {}

    def callbackSuccess(self, pv):
        self.pva_object = pv
        #print(self.pva_object)

    def callbackError(self, code):
        print('error %s' % code)

    def asyncGet(self):
        self.channel.asyncGet(self.callbackSuccess, self.callbackError)
    
    # def get(self):
    #     return self.channel.get()     

    # def readPvObject(self):
    #     self.pva_object = self.asyncGet()
    

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

    def getPvaObject(self):
        return self.pva_object

    def getPvaImage(self):
        return self.image
    
    def getPvaAttributes(self):
        return self.attributes
    
class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")

        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        self.reader = PVA_Reader(PROVIDER_TYPE, PVA_PV)
        self.update_image()

        self.show()

    def update_image(self):
        self.reader.startChannelMonitor()
        time.sleep(0.1)
        print(f"{self.reader.provider} Channel Name = {self.reader.channel.getName()} Channel is connected = {self.reader.channel.isConnected()}")

        self.reader.asyncGet()
        self.reader.stopChannelMonitor()

        self.reader.parsePvaNdattributes()
        print(self.reader.getPvaAttributes())

        self.reader.pvaToImage()
        image = self.reader.getPvaImage()

        if image is not None:
            print(
                f"Shape: \t{image.shape}\n"
                f"DataType: \t{image.dtype}\n"
                f"Min: \t{image.min()}\n"
                f"Max: \t{image.max()}"
            )

            # Convert numpy array to QImage
            if len(image.shape) == 2:  # Grayscale image
                height, width = image.shape
                qImg = QImage(image.data, width, height, QImage.Format_Grayscale8)
            else:  # RGB image
                height, width, channel = image.shape
                bytesPerLine = 3 * width
                qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

            self.label.setPixmap(QPixmap.fromImage(qImg))
              
if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    PVA_PV = "dp-ADSim:Pva1:Image"  # Name of the detector providing the images
    PROVIDER_TYPE = pva.PVA  # Protocol type provided

    window = ImageWindow()

    # Setup QTimer to periodically update the image
    timer = QTimer()
    timer.timeout.connect(window.update_image)
    timer.start(500)  # Timer interval in milliseconds (500 ms = 0.5 seconds)

    sys.exit(app.exec_())
 