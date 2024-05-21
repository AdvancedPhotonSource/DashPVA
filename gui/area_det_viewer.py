import sys
import pvaccess as pva
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from PyQt5 import uic, QtWidgets

def cycle_1234():
            #generator for the rotation
            while True:
                for i in range(1,5):
                    yield i
gen = cycle_1234()
    
class PVA_Reader:

    def __init__(self, provider=pva.PVA, pva_prefix="dp-ADSim:Pva1:Image"):
        
        """variables needed for monitoring a connection"""
        self.provider = provider
        self.pva_prefix = pva_prefix
        self.channel = pva.Channel(self.pva_prefix, self.provider)

        """variables that will store pva data"""
        self.pva_object = None
        self.image = None
        self.shape = (0,0)
        self.attributes = {}
        self.timestamp = None
        self.pva_cache = []
        self.__last_array_id = None
        self.frames_missed = 0
        self.frames_received = 0
        self.data_type = None
        
    def callbackSuccess(self, pv):
        self.pva_object = pv
        if len(self.pva_cache) < 1000: 
            self.pva_cache.append(pv)
            self.pvaToImage()
            self.parseImageDataType()
        else:
            self.pva_cache = self.pva_cache[1:]
            self.pva_cache.append(pv)
            self.pvaToImage()
            self.parseImageDataType()
            
    def parseImageDataType(self):
        if self.pva_object is not None:
            self.data_type = list(self.pva_object['value'][0].keys())[0]
    
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
        try:
            if self.pva_object is not None:
                self.frames_received += 1
                if "dimension" in self.pva_object:
                    self.shape = tuple([dim["size"] for dim in self.pva_object["dimension"]])
                    image = np.array(self.pva_object["value"][0][self.data_type])
                    image = np.reshape(image, self.shape)
                else:
                    image = None
                
                self.image = image

                data = self.pva_cache[-1]
                if data is not None:
                    current_array_id = data['uniqueId']
                    if self.__last_array_id is not None: 
                        id_diff = current_array_id - self.__last_array_id - 1
                        self.frames_missed += id_diff if (id_diff > 0) else 0
                    self.__last_array_id = current_array_id
        except:
            self.frames_missed += 1
            
    def startChannelMonitor(self):
        self.channel.subscribe('callback success', self.callbackSuccess)
        self.channel.startMonitor()

    def stopChannelMonitor(self):
        self.channel.unsubscribe('callback success')
        self.channel.stopMonitor()

    def getPvaObjects(self):
        return self.pva_cache

    def getLastPvaObject(self):
        return self.pva_cache[-1]

    def getFramesMissed(self):
        return self.frames_missed

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

        #Initializing important variables
        self.reader = None
        self.call_id_plot = 0
        self.first_plot = True
        self.rot_num = 0 #original rotation count
        self.timer_poll = QTimer()
        self.timer_avgs = QTimer()
        self.timer_plot = QTimer()

        self.timer_poll.timeout.connect(self.update_labels)
        self.timer_avgs.timeout.connect(self.update_horizontal_vertical_plots)            
        self.timer_plot.timeout.connect(self.update_image)

        # Making image_view a plot to show axes
        plot = pg.PlotItem()        
        self.image_view = pg.ImageView(view=plot)
        #self.image_view.ui.histogram.vb.setYRange(0,max=300)
        self.viewer_layout.addWidget(self.image_view,1,1)

        self.image_view.view.getAxis('left').setLabel(text='Row [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='Columns [pixels]')
        
        #TODO: Adjust so that location, width, and height are read in from the detector
        # roi = pg.ROI([512, 512], [100, 100],pen=pg.mkPen("r"),movable=False)
        # self.image_view.addItem(roi)
        
        #Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.connect_pva_reader)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.log_image.clicked.connect(self.update_image)
        self.rotate90degCCW.clicked.connect(self.rotation_count)
        self.max_setting_val.valueChanged.connect(self.update_max_setting)

        self.horizontal_avg_plot = pg.PlotWidget()
        self.horizontal_avg_plot.invertY(True)
        self.horizontal_avg_plot.setMaximumWidth(175)
        self.horizontal_avg_plot.setYLink(self.image_view.getView())
        self.viewer_layout.addWidget(self.horizontal_avg_plot, 1,0)


    def connect_pva_reader(self):
        try:
            prefix = self.pv_prefix.text()

            if self.reader is None:
                self.reader = PVA_Reader(pva_prefix=prefix)
                self.reader.startChannelMonitor()

                if self.reader.channel.get():
                    self.start_timers()
            else:
                self.stop_timers()
                self.reader.stopChannelMonitor()
                del self.reader
                self.reader = PVA_Reader(pva_prefix=prefix)
                self.reader.startChannelMonitor()

                if self.reader.channel.get():
                    self.start_timers()
        except:
            self.image_view.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            del self.reader
            self.reader = None
            self.provider_name.setText("N/A")
            self.name_val.setText("none")
            self.is_connected.setText("Disconnected")

    def start_timers(self):
        self.timer_poll.start(1000/100)
        self.timer_avgs.start(int(1000/float(self.plotting_frequency.text())))
        self.timer_plot.start(int(1000/float(self.plotting_frequency.text())))

    def stop_timers(self):
        self.timer_plot.stop()
        self.timer_avgs.stop()
        self.timer_poll.stop()

    def update_horizontal_vertical_plots(self):
        image = self.reader.getPvaImage()
        if image is not None:
            image = np.rot90(image, k = self.rot_num)
            self.horizontal_avg_plot.plot(x=np.mean(image, axis=0), y=np.arange(0,self.reader.shape[1]), clear=True)

    def reset_first_plot(self):
        self.first_plot = True

    def start_live_view_clicked(self):
        if self.reader is not None:
            if self.reader.channel.isMonitorActive():
                self.start_timers()
                
            elif self.reader.channel:
                self.reader.startChannelMonitor()
                self.start_timers()


    def stop_live_view_clicked(self):
        if self.reader is not None:
            self.timer_plot.stop()
            self.reader.stopChannelMonitor()
        

    def freeze_image_checked(self):
        if self.reader is not None:
            if self.freeze_image.isChecked():
                self.timer_poll.stop()
                self.timer_plot.stop()
                self.timer_avgs.stop()
                
            else:
                self.start_timers()

    def update_mouse_pos(self, pos):
        if pos is not None:
            if self.reader is not None:
                img = self.image_view.getImageItem()
                q_pointer = img.mapFromScene(pos)
                x, y = q_pointer.x(), q_pointer.y()
                self.mouse_x_val.setText(f"{x:.7f}")
                self.mouse_y_val.setText(f"{y:.7f}")
                img_data = self.reader.getPvaImage()
                if img_data is not None:
                    img_data = np.rot90(img_data, k = self.rot_num)
                    if 0 <= x < self.reader.shape[0] and 0 <= y < self.reader.shape[1]:
                       self.mouse_px_val.setText(f'{img_data[int(x)][int(y)]}')

    def update_labels(self):
        provider_name = f"{self.reader.provider if self.reader.channel.isMonitorActive() else None}"
        channel_name = self.reader.channel.getName() if self.reader.channel.isMonitorActive() else "none"
        is_connected = "Connected" if self.reader.channel.isMonitorActive() else "Disconnected"
        self.provider_name.setText(provider_name)
        self.name_val.setText(channel_name)
        self.is_connected.setText(is_connected)
        self.missed_frames_val.setText(f"{self.reader.frames_missed:d}")
        self.frames_received_val.setText(f"{self.reader.frames_received:d}")
        self.plot_call_id.setText(f"{self.call_id_plot:d}")
        self.size_x_val.setText(f"{self.reader.shape[0]}")
        self.size_y_val.setText(f"{self.reader.shape[1]}")
        self.data_type_val.setText(self.reader.data_type)

    def rotation_count(self):
        self.rot_num = next(gen)
        print(f'rotation num: {self.rot_num}')

    def update_image(self):
        if self.reader is not None:
            self.call_id_plot +=1
            image = self.reader.getPvaImage()
            if image is not None:
                image = np.rot90(image, k = self.rot_num)
                if len(image.shape) == 2:
                    min_level, max_level = np.min(image), np.max(image)
                    if self.log_image.isChecked():
                            image = np.log(image + 1)
                            min_level = np.log(min_level + 1)
                            max_level = np.log(max_level + 1)
                    if self.first_plot:
                        height, width = image.shape[:2]
                        coordinates = pg.QtCore.QRectF(0, 0, width - 1, height - 1)
                        self.image_view.imageItem.setImage(image, autoRange=False, autoLevels=False, rect=coordinates, axes={'y': 0, 'x': 1})
                        #need to also clone the image to self.image_view.image so the ROI feature  works | probably a memory leak here ?
                        self.image_view.setImage(image, autoRange=False, autoLevels=False, levels=(min_level, max_level)) #, levels=(min_level, max_level)
                        self.max_setting_val.setValue(max_level)
                        self.first_plot = False
                    else:
                        height, width = image.shape[:2]
                        coordinates = pg.QtCore.QRectF(0, 0, width - 1, height - 1)
                        self.image_view.imageItem.setImage(image, autoRange=False, autoLevels=False, rect=coordinates, axes={'y': 0, 'x': 1})
                        self.image_view.setImage(image, autoRange=False, autoLevels=False)
                        
                self.min_px_val.setText(f"{min_level:.2f}")
                self.max_px_val.setText(f"{max_level:.2f}")
    
    def update_max_setting(self):
        max = float(self.max_setting_val.text())
        self.image_view.setLevels(0, max)

            

if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    window = ImageWindow()

    sys.exit(app.exec_())