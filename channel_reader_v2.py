import datetime
import enum
import time
import pvaccess as pva
from pvaccess import PvaException
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pyqtgraph import QtCore #provides timer important for polling

#TODO: implement polling of live data, ask Peco for assistance in setting up timer

#TODO: Have polling start and stop automatically based on timer

#TODO: have polling rate change with a variable


class ConnectionState(enum.Enum):
    """
    Connection State of the PV channel
    """
    CONNECTED = 1
    CONNECTING = 2
    DISCONNECTING = 3
    DISCONNECTED = 4
    FAILED_TO_CONNECT = 5

    def __str__(self):
        string_lookup = {
            1 : 'Connected',
            2 : 'Connecting',
            3 : 'Disconnecting',
            4 : 'Disconnected',
            5 : 'Failed to connect'
        }

        return string_lookup[int(self.value)]
    
######################################################################################################################

class MonitorStrategy:
    """
    Implementation of strategy interface used in Channel

    Uses pvmonitor to get data from PV channel
    """
    def __init__(self, context):
        """
        context:  PV_Reader object
        """
        self.ctx = context
        
    def _data_callback(self, data):
        #
        # Data callback called by pvmonitor
        #
        # data:  data from PV channel
        #
        self.ctx.setData(data)

    def _connection_callback(self, is_connected):
        #
        # Callback called by pvmonitor when there is a connection change
        # Changes the Channel state or notify error
        #
        #  is_connected:  True if connected to PV channel.  Otherwise False
        #
        if is_connected:
            self.ctx.set_state(ConnectionState.CONNECTED)
        elif self.ctx.is_running():
            self.ctx.notify_error()            
        
    def pollStart(self):
        """
        Starts the PV monitor
        """
        try:
            self.ctx.channel.setConnectionCallback(self._connection_callback)
            self.ctx.channel.subscribe('monitorCallback', self._data_callback)
            self.ctx.channel.startMonitor('')
        except PvaException as e:
            self.ctx.notify_error(str(e))
        
    def stop(self):
        """
        Stops the PV monitor
        """
        try:
            self.ctx.channel.setConnectionCallback(None)
            self.ctx.channel.stopMonitor()
            self.ctx.channel.unsubscribe('monitorCallback')
        except PvaException:
            pass


##################################################################################################
class PollStrategy:
    """
    Implementation of strategy interface used in Channel
    
    Collects data from the PV Channel by periodically polling
    the channel.
    """
    
    def __init__(self, context, timer):
        """
        context: the PVA_Reader #channel pva
        timer: Timer object 
        """
        self.ctx = context
        self.timer = timer
        self.timer.timeout.connect(self.poll)
        
    def _data_callback(self, pv_object):
        #
        # Callback called with returned data from the PV channel
        #
        # data: data from PV channel
        #
        # Set state to CONNECTING when data returned
        self.ctx.set_state(ConnectionState.CONNECTING)
        self.ctx.setData(pv_object)

    def _err_callback(self, msg):
        #
        # Callback called if there was an error attempting to get data
        # Notifies channel that error occured
        #
        # msg: error message
        #
        self.ctx.notify_error(msg)
        
    def poll(self):
        """
        Asynchronously calls pvget on PV channel.  
        Called by the timer object.
        Returned data and any errors are passed in via callbacks
        """
        try:
            self.ctx.channel.asyncGet(self._data_callback, self._err_callback, '')
        except pva.PvaException:
            #error because too many asyncGet calls at once.  Ignore
            pass
        
    def pollStart(self):
        """
        Starts timer to poll data. 
        """
        self.timer.start(int(1000/self.ctx.rate))

    def pollStop(self):
        """
        Stops polling
        """
        self.timer.stop()

#######################################################################################################
class PVA_Reader:

    def __init__(self, provider=pva.PVA, pva_name="dp-ADSim:Pva1:Image", timer=None):
        
        """variables needed for monitoring a connection"""
        self.provider = provider
        self.pva_name = pva_name
        self.channel = pva.Channel(pva_name, provider)
        self.rate = None
        self.state = None
        self.monitor_strategy = MonitorStrategy(self)
        #self.poll_strategy = PollStrategy(self, timer) if timer else None
        self.status_callback = None
        #self.timer = timer

        """variables that will store pva data"""
        self.pva_object = None
        self.data = None
        self.image = None
        self.attributes = {}
        self.timestamp = None
        self.pva_cache = {}


    def set_state(self, state, msg=None):
        """
        Set connection state.  

        state: ConnectionState object
        msg: (str) optional message
        """
        if state != self.state:
            self.state = state
            if self.status_callback:
                self.status_callback(str(state), msg)

    def notify_error(self, msg=None):
        """
        Notify that unable to connect to PV channel
        """
        #Only update status for the current PV that is running to prevent status overrides during error callbacks
        if self.is_running():
            self.set_state(ConnectionState.FAILED_TO_CONNECT, msg)

    def is_running(self):
        """
        Returns True if Channel is running
        """
        return self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]

    def setData(self, pv_object):
        print(pv_object)
        self.data = pv_object.get()
    
    def get(self):
        pv_obj = self.channel.get('')
        self.pva_cache[pv_obj.name] = pv_obj
        return pv_obj   

    def asyncGet(self, success_callback=None, error_callback=None):
        """
        Get data asynchronously from the PV channel.

        :param success_callback: Callback for successful data retrieval
        :param error_callback: Callback for error handling
        """
        def success_callback_wrapper(data):
            self.pva_cache[self.name] = data  # Cache the retrieved data
            if success_callback:
                success_callback(data)

        self.channel.asyncGet(success_callback=success_callback_wrapper, error_callback=error_callback)

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

    def startChannelMonitor(self, routine=None, rate=None, status_callback=None):
        """
        Start monitoring data
        
        routine: PV data callback
        rate: Rate to poll data.  If set, will poll to get data.  Otherwise
             will use a PV monitor
        status_callback: status callback.  Called whenever there is a state change.
        """
        self.data_callback = routine
        self.rate = rate
        if not self.monitor_strategy:
            raise Exception("Can't poll data unless timer is configured")
        if status_callback:
            self.status_callback = status_callback
        self.set_state(ConnectionState.CONNECTING)
        self.monitor_strategy.pollStart()

    def stopChannelMonitor(self):
        self.channel.stopMonitor()

    def getPvaObject(self):
        return self.pva_object

    def getPvaImage(self):
        return self.image
    
    def getPvaAttributes(self):
        return self.attributes

########################################################################################################################

PVA_PV = "dp-ADSim:Pva1:Image" #name of the detector provide the images
PROVIDER_TYPE = pva.PVA   #protocol type provided
        
if __name__ == "__main__":

    """establish connection and monitor channel"""
    timer = QtCore.QTimer()
    reader = PVA_Reader(PROVIDER_TYPE, PVA_PV, timer)

    reader.startChannelMonitor(rate=3)
    time.sleep(0.1)
    #print(f"{reader.provider} Channel Name = {reader.channel.getName()} Channel is connected = {reader.channel.isConnected()}")
    
    """retrieve and store a pva object from a channel"""
    #reader.readPvObject()
    #TODO: poll channel and store multiple pvaObjects
    

    """parsing and printing the pva's attribute's"""
    reader.parsePvaNdattributes()
    print(reader.getPvaAttributes())

    time.sleep(10)

    #reader.stopChannelMonitor()

    """create an image based on pva object byte data"""
    # reader.pvaToImage()
    # image = reader.getPvaImage()

    # if image is not None:
    #     print(
    #         f"Shape: \t{image.shape}\n"
    #         f"DataType: \t{image.dtype}\n"
    #         f"Min: \t{image.min()}\n"
    #         f"Max: \t{image.max()}"
    #     )

    # #TODO: Once polling is set up, find a way to show new image, may have to switch to pyqtgraph for speed
    # plt.imshow(image, norm=LogNorm(), cmap='viridis')
    # plt.colorbar(label='Value')
    # plt.title('Log-Normal Distribution')
    # plt.show()

