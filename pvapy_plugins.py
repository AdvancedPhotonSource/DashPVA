# -*- coding: utf-8 -*-

"""
C2 DATA VIEWER is distributed subject to a Software License Agreement found
in the file LICENSE that is included with this distribution.
SPDX-License-Identifier: EPICS

Copyright 2018 UChicago Argonne LLC
 as operator of Argonne National Laboratory

PVA object viewer utilities using pvaPy as pvAccess binding

@author: Guobao Shen <gshen@anl.gov>
"""
import time
from threading import Lock
from collections import deque
from statistics import mean
import enum

import pvaccess as pva
from pvaccess import PvaException

def make_protocol(proto):
    """
    Returns the pvapy enum for the given protocol string
    
    proto:  "ca" or "pva"
    """
    if isinstance(proto, pva.ProviderType):
        return proto
    
    proto = proto.lower()
    if proto == 'ca':
        return pva.ProviderType.CA
    elif proto == 'pva':
        return pva.ProviderType.PVA
    else:
        raise Exception('Invalid protocol: ' + proto)

def parse_pvname(pvname, default_proto):
    """
    Parses the pvname string provided from user.  Returns
    (PV, protocol)

    pvname:  string in format proto://PV or PV
    default_proto:  Default protocol if one is not given
    """
    proto = default_proto
    if "://" in pvname:
        proto, pvname = pvname.split('://')
        proto = make_protocol(proto.lower().strip())

    return pvname, proto

class PollStrategy:
    """
    Implementation of strategy interface used in Channel
    
    Collects data from the PV Channel by periodically polling
    the channel.
    """
    
    def __init__(self, context, timer):
        """
        context: the Channel object #channel pva
        timer: QT Timer object 
        """
        self.ctx = context
        self.timer = timer
        self.timer.timeout.connect(self.poll)
        
    def _data_callback(self, data):
        #
        # Callback called with returned data from the PV channel
        #
        # data: data from PV channel
        #
        # Set state to CONNECTING when data returned
        self.ctx.set_state(ConnectionState.CONNECTING)
        self.ctx.data_callback_wrapper(data)

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
        
    def start(self):
        """
        Starts timer to poll data. 
        """
        self.timer.start(int(1000/self.ctx.rate))

    def stop(self):
        """
        Stops polling
        """
        self.timer.stop()
    
class MonitorStrategy:
    """
    Implementation of strategy interface used in Channel

    Uses pvmonitor to get data from PV channel
    """
    def __init__(self, context):
        """
        context:  Channel object
        """
        self.ctx = context
        
    def _data_callback(self, data):
        #
        # Data callback called by pvmonitor
        #
        # data:  data from PV channel
        #
        self.ctx.data_callback_wrapper(data)

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
        
    def start(self):
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

class Channel:
    """
    PV Channel object.  Manages getting the PV data and monitoring channel connection
    status
    """
    def __init__(self, name, timer=None, provider=pva.PVA, status_callback=None):
        """
        name : PV name
        timer: Timer to use if polling data
        provider: PV protocol
        status_callback:  Callback called if connection state changed
        """
        self.channel = pva.Channel(name, provider)
        self.provider = provider
        self.name = name
        self.rate = None
        self.data_callback = None
        self.data = None
        self.monitor_strategy = MonitorStrategy(self)
        self.poll_strategy = PollStrategy(self, timer) if timer else None
        self.strategy = None
        self.rate = None
        self.status_callback = status_callback
        self.state = ConnectionState.DISCONNECTED

    def data_callback_wrapper(self, data):
        """
        Data callback called when received data from the PV channel
        
        data: PV data
        """
        if not self.is_running():
            return
        
        self.set_state(ConnectionState.CONNECTED)
        if self.data_callback:
            self.data_callback(data)
        else:
            self.data = data.get()

    def notify_error(self, msg=None):
        """
        Notify that unable to connect to PV channel
        """
        #Only update status for the current PV that is running to prevent status overrides during error callbacks
        if self.is_running():
            self.set_state(ConnectionState.FAILED_TO_CONNECT, msg)
        
    def start(self, routine=None, rate=None, status_callback=None):
        """
        Start monitoring data
        
        routine: PV data callback
        rate: Rate to poll data.  If set, will poll to get data.  Otherwise
             will use a PV monitor
        status_callback: status callback.  Called whenever there is a state change.
        """
        self.data_callback = routine
        self.rate = rate
        self.strategy = self.poll_strategy if self.rate else self.monitor_strategy
        if not self.strategy:
            raise Exception("Can't poll data unless DataSource timer is configured")
        if status_callback:
            self.status_callback = status_callback
        self.set_state(ConnectionState.CONNECTING)
        self.strategy.start()

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
                
    def stop(self):
        """
        Stop monitoring data
        """
        self.set_state(ConnectionState.DISCONNECTING)
        if self.strategy:
            self.strategy.stop()
        self.set_state(ConnectionState.DISCONNECTED)

    def is_running(self):
        """
        Returns True if Channel is running
        """
        return self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]
    
    def get(self):
        """
        Get data from PV channel.  blocking call.  Returns data if successful.
        """
        try:
            return self.channel.get('')
        except PvaException as e:
            self.notify_error(str(e))
            raise

    def async_get(self, data_callback, error_callback):
        """
        Asynchronously get data from PV channel. Nonblocking call.
        
        data_callback: callback called when received data
        error_callback: callback called if error occured
        """
        try:
            self.channel.asyncGet(data_callback, error_callback, '')
        except pva.PvaException:
            #error because too many asyncGet calls at once.  Ignore
            pass
        
class DataSource:
    def __init__(self, timer_factory=None, default=None):
        """

        :param default:
        """
        # local cache of latest data
        self.data = None
        # EPICS7 channel
        self.channel = None
        self.channel_cache = {}
        self.fps = None
        self.timer_factory = timer_factory if timer_factory else lambda: None
            
        self.data_callback = None
        self.status_callback = None
        
        # Default PV name
        self.device = None
        if default is not None:
            if type(default) in [list, tuple]:
                self.__init_connection(default[0])
            elif type(default) == dict:
                self.__init_connection(list(default.values())[0])
            elif type(default) == str:
                self.__init_connection(default)
            else:
                raise RuntimeError("Unknown data type of default parameter: ", default)

        # trigger support from external EPICS3/7 channel
        self.trigger = None
        self.trigger_chan = None

    def create_connection(self, name, provider):
        if name in self.channel_cache and self.channel_cache[name].provider == provider:
            return self.channel_cache[name]
        else:
            chan = Channel(name, self.timer_factory(), provider=provider)
            self.channel_cache[name] = chan
            return chan
    
    def __init_connection(self, name):
        """
        Create initial channel connection with given PV name

        :param name: EPICS7 PV name
        :return:
        """

        name, proto = parse_pvname(name, pva.ProviderType.PVA)
        self.device = name
        
        self.channel = Channel(self.device, self.timer_factory(), proto)
        self.channel_cache[name] = self.channel
            
    def get(self):
        """
        Get data from current PV channel
        :return:
        """
        if self.channel is None:
            return None

        if not self.channel.status_callback:
            self.channel.status_callback = self.status_callback
        
        return self.channel.get()

    def async_get(self, success_callback=None, error_callback=None):
        """
        Get data from current PV channel
        :return:
        """
        if self.channel is None:
            return

        if not self.channel.status_callback:
            self.channel.status_callback = self.status_callback

        if not success_callback:
            success_callback = lambda x: None

        if not error_callback:
            error_callback = lambda x: None
            
        self.channel.async_get(success_callback, error_callback)

    def update_framerate(self, fps):
        self.fps = fps
        
    def update_device(self, name, restart=False, test_connection=True):
        """
        Update device, EPICS PV name, and test its connectivity

        :param name: device name
        :param restart: flag to restart or not
        :param test_connection: flag to test channel connectivity or not
        :return:
        :raise PvaException: raise pvaccess exception when channel cannot be connected.
        """
        if self.channel is not None:
            self.stop()

        if name != "":
            chan = None
            if name in self.channel_cache:
                chan = self.channel_cache[name]
            else:
                name, proto = parse_pvname(name, pva.ProviderType.PVA)
                chan = Channel(name, self.timer_factory(), provider=proto, status_callback=self.status_callback)
                self.channel_cache[name] = chan
                
            self.channel = chan
            self.device = name

            # test channel connectivity
            if test_connection:
                chan.get()

            if restart:
                self.start()
        else:
            self.channel = None

    def start(self, routine=None, status_callback=None):
        """
        Start a EPICS7 monitor for current device.

        :return:
        :raise PvaException: raise pvaccess exception when channel cannot be connected.
        """
        if self.channel is None:
            return

        if routine:
            self.data_callback = routine

        if status_callback:
            self.status_callback = status_callback
            
        self.channel.start(routine=self.data_callback, rate=self.fps, status_callback=self.status_callback)

    def stop(self):
        """
        Stop monitor EPICS7 channel for current device.

        :return:
        :raise PvaException: raise pvaccess exception when channel fails to disconnect.
        """
        if self.channel is None:
            return

        self.channel.stop()

    def update_trigger(self, name, proto='ca'):
        """
        Update trigger PV name.  If failed to connect, trigger is set to None

        :param name: PV name for external trigger
        :param proto: protocol, should be 'pva' or 'ca'. 'ca' by default
        :return: trigger fields
        """
        if self.trigger_chan is not None:
            self.stop_trigger()

        self.trigger = None
        
        if name == self.trigger:
            # nothing change
            return
        
        if proto.lower() == "ca":
            # use channel access protocol, which is default
            trigger_chan = pva.Channel(name, pva.CA)
        elif proto.lower() == "pva":
            # use pvAccess protocol
            trigger_chan = pva.Channel(name)
        else:
            raise RuntimeError("Unknown EPICS communication protocol {}".format(proto))

        # test connectivity, which might cause for example timeout exception
        pv = trigger_chan.get("")

        # update trigger name & channel
        self.trigger = name
        self.trigger_proto = proto.lower()
        self.trigger_chan = trigger_chan
        pv_structure = pv.getStructureDict()
        keys =[ k for k,v in pv_structure.items() ]
        
        return keys

    def start_trigger(self, routine):
        """

        :return:
        """
        try:
            self.trigger_chan.subscribe('triggerMonitorCallback', routine)
            req = ''
            if self.trigger_proto == 'ca':
                #need to explicitly ask for timeStamp 
                req = 'field(timeStamp,value)'
            self.trigger_chan.startMonitor(req)
        except PvaException:
            raise RuntimeError("Cannot connect to PV {}".format(self.trigger))

    def stop_trigger(self):
        """

        :return:
        """
        if not self.trigger_chan:
            return
        
        try:
            self.trigger_chan.stopMonitor()
            self.trigger_chan.unsubscribe('triggerMonitorCallback')
        except PvaException:
            # raise RuntimeError("Fail to disconnect")
            pass
