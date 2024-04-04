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
        self.pva_cache = {}
        self.__last_array_id = None
        self.frames_missed = 0

    def callbackSuccess(self, pv):
        self.pva_object = pv
        self.pva_cache[self.pva_object['uniqueId']] = self.pva_object.get()
        #print(self.pva_object)

    def callbackError(self, code):
        print('error %s' % code)

    def asyncGet(self):
        self.channel.asyncGet(self.callbackSuccess, self.callbackError)
    
    def get(self):
        return self.channel.get()     

    # def readPvObject(self):
    #     self.pva_object = self.asyncGet()

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

    def getPvaObject(self):
        return self.pva_object

    def getPvaImage(self):
        return self.image
    
    def getAttributesDict(self):
        return self.attributes
    
class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")

        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        self.reader = PVA_Reader(PROVIDER_TYPE, PVA_PV)
        self.update_image()

        self.deltaTime = time.time()
        self.fr = 0.5

        self.show()

    def printNumFrames(self):
        print('Number of Frames detected: %s'%len(self.reader.pva_cache.keys()))



    def update_image(self):
        self.reader.startChannelMonitor()
        time.sleep(0.1)
        print(f"{self.reader.provider} Channel Name = {self.reader.channel.getName()} Channel is connected = {self.reader.channel.isConnected()}")

        self.reader.asyncGet()
        self.reader.stopChannelMonitor()

        self.reader.parsePvaNdattributes()
        print('PV UniqueID: %s' % self.reader.getAttributesDict().get('uniqueId'))

        self.reader.calcFramesMissed(self.reader.pva_object)
        print('Frames Missed: %s' % self.reader.getFramesMissed())

        self.reader.pvaToImage()
        image = self.reader.getPvaImage()

        if image is not None:
            print(
                f"Shape: \t{image.shape}\n"
                f"DataType: \t{image.dtype}\n"
                f"Min: \t{image.min()}\n"
                f"Max: \t{image.max()}"
            )

            if (time.time() - self.deltaTime) >= self.fr:
                # Convert numpy array to QImage
                if len(image.shape) == 2:  # Grayscale image
                    height, width = image.shape
                    qImg = QImage(image.data, width, height, QImage.Format_Grayscale8)
                else:  # RGB image
                    height, width = image.shape
                    bytesPerLine = 3 * width
                    qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

                self.label.setPixmap(QPixmap.fromImage(qImg))
                self.deltaTime = time.time()

     
              
if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    PVA_PV = "dp-ADSim:Pva1:Image"  # Name of the detector providing the images
    PROVIDER_TYPE = pva.PVA  # Protocol type provided

    window = ImageWindow()

    # Setup QTimer to periodically update the image
    timer = QTimer()
    timer.timeout.connect(window.update_image)
    timer.start(100)  # Timer interval in milliseconds (100 ms = 0.1 seconds)

    countdown = QTimer()
    countdown.singleShot(10000, timer.stop)

    pr = QTimer()
    pr.singleShot(11000, window.printNumFrames)

    sys.exit(app.exec_())
