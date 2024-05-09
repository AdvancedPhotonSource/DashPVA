import sys
import pvaccess as pva
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
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
        self.shape = (0,0)
        self.attributes = {}
        self.timestamp = None
        self.pva_cache = []
        self.__last_array_id = None
        self.frames_missed = 0
        self.poll_id = 0
        self.frames_received = 0
        self.data_type = None

    def callbackSuccess(self, pv):
        self.pva_object = pv
        if len(self.pva_cache) < 1000 : 
            self.pva_cache.append(pv)
            self.poll_id +=1
            self.frames_received += 1
            self.parseImageDataType()
            self.calcFramesMissed()
        else:
            self.pva_cache = self.pva_cache[1:]
            self.pva_cache.append(pv)
            self.poll_id +=1
            self.frames_received += 1
            self.parseImageDataType()
            self.calcFramesMissed()

    def callbackError(self, code):
        self.poll_id += 1
        print('error %s' % code)

    def parseImageDataType(self):
        if self.pva_object is not None:
            self.data_type = list(self.pva_object['value'][0].keys())[0]

    def calcFramesMissed(self):
        data = self.pva_cache[-1]
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
                self.shape = tuple([dim["size"] for dim in self.pva_object["dimension"]])
                image = np.array(self.pva_object["value"][0][self.data_type])
                image = np.reshape(image, self.shape)
            else:
                image = None
            
            self.image = image
        else:
            print('pvaObject is none')

    def startChannelMonitor(self):
        self.channel.subscribe('callback success', self.callbackSuccess)
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
        uic.loadUi('gui/imageshow.ui', self)
        self.setWindowTitle("Image Viewer with PVAaccess")
        self.show()

        self.reader = PVA_Reader(pva.PVA, self.pv_prefix.text())
        self.reader.startChannelMonitor() #start monitor once window is active
        self.call_id_plot = 0
        self.min_px = 0
        self.max_px = 0


        #TODO: Find Way to make Image View take a Plot Item for axes labels to appear
        self.image_vb = self.image_view.getView()

        self.x_axis = pg.AxisItem(orientation='bottom')
        self.y_axis = pg.AxisItem(orientation='left')

        # Add axis items to the view box of the image view
        
        



        self.image_vb.scene().sigMouseMoved.connect(self.update_mouse_pos)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.log_image.clicked.connect(self.update_image)

        self.timer_poll = QTimer()
        self.timer_poll.timeout.connect(self.update_labels)
        self.timer_poll.start(int(1000/float(self.update_frequency.text())))
        
        self.timer_plot = QTimer()
        self.timer_plot.timeout.connect(self.update_image)
        self.timer_plot.start(int(1000/float(self.plotting_frequency.text())))
        
        self.first_plot = True


    def reset_first_plot(self):
        self.first_plot = True

    def start_live_view_clicked(self):
        self.timer_poll.start(int(1000/float(self.update_frequency.text())))
        self.timer_plot.start(int(1000/float(self.plotting_frequency.text())))

    def stop_live_view_clicked(self):
        self.timer_poll.stop()
        self.timer_plot.stop()

            
    #Not exactly what we want as it gets looks at each pixel without scaling
    def update_mouse_pos(self, pos):
        if pos is not None:
            img = self.image_view.getImageItem()
            q_pointer = img.mapFromScene(pos)
            x, y = q_pointer.x(), q_pointer.y()

            self.mouse_x_value.setText(f"{x:.7f}")
            self.mouse_y_value.setText(f"{y:.7f}")


    
    #changed this to update labels as most processing will be done in the monitor call    
    def update_labels(self):
        provider_name = f"{self.reader.provider}"
        channel_name = self.reader.channel.getName()
        is_connected = "Connected" if self.reader.channel.isConnected() else "Disconnected"
        self.provider_name.setText(provider_name)
        self.name_val.setText(channel_name)
        self.is_connected.setText(is_connected)
        self.missed_frames.setText(f"{self.reader.frames_missed:d}")
        self.frames_received_val.setText(f"{self.reader.frames_received:d}")
        self.poll_call_id.setText(f"{self.reader.poll_id:d}")
        self.plot_call_id.setText(f"{self.call_id_plot:d}")
        self.size_x_value.setText(f"{self.reader.shape[0]}")
        self.size_y_value.setText(f"{self.reader.shape[1]}")
        self.min_px_val.setText(f"{self.min_px:.2f}")
        self.max_px_val.setText(f"{self.max_px:.2f}")
        self.data_type_val.setText(self.reader.data_type)

    def update_image(self):
        self.call_id_plot +=1
        self.reader.pvaToImage()
        image = self.reader.getPvaImage()

        if image is not None:
            if len(image.shape) == 2:
                min_level, max_level = np.min(image), np.max(image)
                if self.log_image.isChecked():
                        image = np.log(image + 1)
                        min_level = np.log(min_level + 1)
                        max_level = np.log(max_level + 1)
                if self.first_plot:
                    self.image_view.setImage(image, autoRange=False, autoLevels=False, levels=(min_level, max_level))
                    self.first_plot = False
                else:
                    self.image_view.setImage(image, autoRange=False, autoLevels=False)
            
                self.min_px = self.image_view.quickMinMax(image)[0][0]
                self.max_px = self.image_view.quickMinMax(image)[0][1]
                


if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    window = ImageWindow()

    sys.exit(app.exec_())