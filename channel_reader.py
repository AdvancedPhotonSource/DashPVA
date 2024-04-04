import datetime
import enum
import time
import pvaccess as pva
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
        self.data = None
        self.image = None
        self.attributes = {}
        self.timestamp = None
        self.pva_cache = {}

    def setData(self, pv_object):
        self.data = pv_object.get()
    
    def get(self):
        return self.channel.get('')     

    def readPvObject(self):
        self.pva_object = self.channel.get()
    

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
        if "dimension" in self.pva_object:
            shape = tuple([dim["size"] for dim in self.pva_object["dimension"]])
            image = np.array(self.pva_object["value"][0]["byteValue"])
            image = np.reshape(image, shape)
        else:
            image = None
        
        self.image = image

    def timedMonitor(self, timer):
        pass

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
    

PVA_PV = "dp-ADSim:Pva1:Image" #name of the detector provide the images
PROVIDER_TYPE = pva.PVA   #protocol type provided
        
if __name__ == "__main__":

    """establish connection and monitor channel"""
    
    reader = PVA_Reader(PROVIDER_TYPE, PVA_PV)

    reader.startChannelMonitor()
    time.sleep(0.1)
    print(f"{reader.provider} Channel Name = {reader.channel.getName()} Channel is connected = {reader.channel.isConnected()}")

    """retrieve and store a pva object from a channel"""
    reader.readPvObject()
    #TODO: poll channel and store multiple pvaObjects
    reader.stopChannelMonitor()

    print(reader.pva_object)

    """parsing and printing the pva's attribute's"""
    reader.parsePvaNdattributes()
    print(reader.getPvaAttributes())

    """create an image based on pva object byte data"""
    reader.pvaToImage()
    image = reader.getPvaImage()

    if image is not None:
        print(
            f"Shape: \t{image.shape}\n"
            f"DataType: \t{image.dtype}\n"
            f"Min: \t{image.min()}\n"
            f"Max: \t{image.max()}"
        )

    #TODO: Once polling is set up, find a way to show new image, may have to switch to pyqtgraph for speed
    plt.imshow(image, norm=LogNorm(), cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Log-Normal Distribution')
    plt.show()

