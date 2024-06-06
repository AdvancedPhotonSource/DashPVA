import sys
import pvaccess as pva
import json
from epics import camonitor
from epics import caget
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from PyQt5 import uic, QtWidgets

def cycle_1234():
            #generator for the rotation
            while True:
                for i in range(1,5):
                    yield i
gen = cycle_1234()


class ROI_Stats_Dialog(QDialog):

    def __init__(self, parent, text):
        super(ROI_Stats_Dialog,self).__init__()
        uic.loadUi("gui/roi_stats_dialog.ui", self)
        self.text = text
        self.setWindowTitle(f"{self.text} Info")

        #self.stat_num = int(text[-1])
        self.parent = parent

        self.timer_labels = QTimer()
        self.timer_labels.timeout.connect(self.update_stats_labels)
        self.timer_labels.start(1000/100)

    def update_stats_labels(self):
        prefix = self.parent.reader.pva_prefix
        self.stats_total_value.setText(f"{self.parent.stats_data.get(f'{prefix}:{self.text}:Total_RBV', '0.0')}")
        #self.stats_net_value.setText(f"{self.parent.stats_data.get(f'{prefix}:{self.text}:Net_RBV', '0.0')}")
        self.stats_min_value.setText(f"{self.parent.stats_data.get(f'{prefix}:{self.text}:MinValue_RBV', 0.0)}")
        self.stats_max_value.setText(f"{self.parent.stats_data.get(f'{prefix}:{self.text}:MaxValue_RBV', 0.0)}")
        self.stats_sigma_value.setText(f"{self.parent.stats_data.get(f'{prefix}:{self.text}:Sigma_RBV', 0.0):.4f}")
        self.stats_mean_value.setText(f"{self.parent.stats_data.get(f'{prefix}:{self.text}:MeanValue_RBV', 0.0):.4f}")

    def closeEvent(self, event):
        self.timer_labels.stop()
        self.parent.stats_dialog[self.text] = None
        super(ROI_Stats_Dialog,self).closeEvent(event)

    


    
class PVA_Reader:

    def __init__(self, pva_prefix="dp-ADSim",provider=pva.PVA):
        
        """variables needed for monitoring a connection"""
        self.provider = provider
        self.pva_prefix = pva_prefix
        self.channel = pva.Channel(self.pva_prefix+":Pva1:Image", self.provider)

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
        self.pvs = {}
        self.metadata = {}
        self.num_rois = 0

        with open("gui/PVs.json", "r") as json_file:
            self.pvs = json.load(json_file)

    def ca_callback(self, pvname=None, value=None, **kwargs):
        self.metadata[pvname] = value
        
    def pva_callbackSuccess(self, pv):
        self.pva_object = pv
        if len(self.pva_cache) < 1000: 
            self.pva_cache.append(pv)
            self.parseImageDataType()
            self.pvaToImage()
        else:
            self.pva_cache = self.pva_cache[1:]
            self.pva_cache.append(pv)
            self.parseImageDataType()
            self.pvaToImage()
            
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
                    self.image = np.array(self.pva_object["value"][0][self.data_type])
                    self.image= np.reshape(self.image, self.shape)
                else:
                    self.image = None
                
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
        self.channel.subscribe('pva callback success', self.pva_callbackSuccess)
        self.channel.startMonitor()
        if self.pvs and self.pvs is not None:
            try:
                for key in self.pvs:
                    if f"{self.pvs[key]}".startswith(f"ROI"):
                        self.metadata[f"{self.pva_prefix}:{self.pvs[key]}"] = caget(f"{self.pva_prefix}:{self.pvs[key]}")
                        camonitor(f"{self.pva_prefix}:{self.pvs[key]}", callback=self.ca_callback)

                        if not(f"{self.pvs[key]}".startswith(f"ROI{self.num_rois}")):
                            self.num_rois += 1
                    
                    
                        
            except:
                print("Failed to connect to PV")
        
    def stopChannelMonitor(self):
        self.channel.unsubscribe('pva callback success')
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
        self.rois = []
        self.stats_dialog = {}
        self.stats_data = {}
        self.timer_poll = QTimer()
        self.timer_plot = QTimer()

        self.timer_poll.timeout.connect(self.update_labels)
        self.timer_plot.timeout.connect(self.update_image)

        # Making image_view a plot to show axes
        plot = pg.PlotItem()        
        self.image_view = pg.ImageView(view=plot)
        self.viewer_layout.addWidget(self.image_view,1,1)

        self.image_view.view.getAxis('left').setLabel(text='Row [pixels]')
        self.image_view.view.getAxis('bottom').setLabel(text='Columns [pixels]')
        
        #Connecting the signals to the code that will be executed
        self.pv_prefix.returnPressed.connect(self.start_live_view_clicked)
        self.start_live_view.clicked.connect(self.start_live_view_clicked)
        self.plotting_frequency.valueChanged.connect(self.start_timers)
        self.stop_live_view.clicked.connect(self.stop_live_view_clicked)
        self.btn_Stats1.clicked.connect(self.stats_button_clicked)
        self.btn_Stats2.clicked.connect(self.stats_button_clicked)
        self.btn_Stats3.clicked.connect(self.stats_button_clicked)
        self.btn_Stats4.clicked.connect(self.stats_button_clicked)
        self.freeze_image.stateChanged.connect(self.freeze_image_checked)
        self.log_image.clicked.connect(self.update_image)
        self.max_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.min_setting_val.valueChanged.connect(self.update_min_max_setting)
        self.image_view.getView().scene().sigMouseMoved.connect(self.update_mouse_pos)
        self.log_image.clicked.connect(self.reset_first_plot)
        self.rotate90degCCW.clicked.connect(self.rotation_count)



        self.horizontal_avg_plot = pg.PlotWidget()
        self.horizontal_avg_plot.invertY(True)
        self.horizontal_avg_plot.setMaximumWidth(175)
        self.horizontal_avg_plot.setYLink(self.image_view.getView())
        self.viewer_layout.addWidget(self.horizontal_avg_plot, 1,0)



    def start_timers(self):
        self.timer_poll.start(1000/100)
        self.timer_plot.start(int(1000/float(self.plotting_frequency.text())))

    def stop_timers(self):
        self.timer_plot.stop()
        self.timer_poll.stop()

    def start_live_view_clicked(self):
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
            self.start_stats_monitors()
            self.add_rois()

        except:
            print("Failed to Connect")
            self.image_view.clear()
            self.horizontal_avg_plot.getPlotItem().clear()
            del self.reader
            self.reader = None
            self.provider_name.setText("N/A")
            self.is_connected.setText("Disconnected")
        
    def stop_live_view_clicked(self):
        if self.reader is not None:
            self.reader.stopChannelMonitor()
            self.stop_timers()
            for key in self.stats_dialog:
                self.stats_dialog[key] = None
            del self.reader
            self.reader = None
            self.provider_name.setText("N/A")
            self.is_connected.setText("Disconnected")
    
    def start_stats_monitors(self):
        try:
            if self.reader.pvs and self.reader.pvs is not None:
                for key in self.reader.pvs:
                    if f"{self.reader.pvs[key]}".startswith("Stats"):
                            camonitor(f"{self.reader.pva_prefix}:{self.reader.pvs[key]}", callback=self.stats_ca_callback)
        except:
            print("Failed to Connect to PV")

    def stats_ca_callback(self, pvname, value, **kwargs):
        self.stats_data[pvname] = value
        
    def stats_button_clicked(self):
        if self.reader:
            sending_button = self.sender()
            text = sending_button.text()
            self.stats_dialog[text] = ROI_Stats_Dialog(parent=self,text=text)
            self.stats_dialog[text].show()
    
    #TODO: Add functionality to show ROIs checkbox

    def freeze_image_checked(self):
        if self.reader is not None:
            if self.freeze_image.isChecked():
                self.timer_poll.stop()
                self.timer_plot.stop()
                
            else:
                self.start_timers()
    def reset_first_plot(self):
        self.first_plot = True

    def rotation_count(self):
        self.rot_num = next(gen)
        print(f'rotation num: {self.rot_num}')

    def add_rois(self):
        self.roi_colors = ["ff0000", "0000ff", "4CBB17", "ff00ff"]
        for i in range(self.reader.num_rois):
            roi = pg.ROI(
                pos=[self.reader.metadata[f"{self.reader.pva_prefix}:ROI{i+1}:MinX"], self.reader.metadata[f"{self.reader.pva_prefix}:ROI{i+1}:MinY"]],
                size=[self.reader.metadata[f"{self.reader.pva_prefix}:ROI{i+1}:SizeX"], self.reader.metadata[f"{self.reader.pva_prefix}:ROI{i+1}:SizeY"]],
                movable=False,
                pen=pg.mkPen(self.roi_colors[i])
            )
            self.rois.append(roi)
            self.image_view.addItem(roi)
    

    def update_mouse_pos(self, pos):
        if pos is not None:
            if self.reader is not None:
                img = self.image_view.getImageItem()
                q_pointer = img.mapFromScene(pos)
                x, y = q_pointer.x(), q_pointer.y()
                self.mouse_x_val.setText(f"{x:.3f}")
                self.mouse_y_val.setText(f"{y:.3f}")
                img_data = self.reader.getPvaImage()
                if img_data is not None:
                    img_data = np.rot90(img_data, k = self.rot_num)
                    if 0 <= x < self.reader.shape[0] and 0 <= y < self.reader.shape[1]:
                       self.mouse_px_val.setText(f'{img_data[int(x)][int(y)]}')

    def update_labels(self):
        provider_name = f"{self.reader.provider if self.reader.channel.isMonitorActive() else 'N/A'}"
        is_connected = "Connected" if self.reader.channel.isMonitorActive() else "Disconnected"
        self.provider_name.setText(provider_name)
        self.is_connected.setText(is_connected)
        self.missed_frames_val.setText(f"{self.reader.frames_missed:d}")
        self.frames_received_val.setText(f"{self.reader.frames_received:d}")
        self.plot_call_id.setText(f"{self.call_id_plot:d}")
        self.size_x_val.setText(f"{self.reader.shape[0]}")
        self.size_y_val.setText(f"{self.reader.shape[1]}")
        self.data_type_val.setText(self.reader.data_type)
        self.roi1_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats1:Total_RBV', '0.0')}")
        self.roi2_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats2:Total_RBV', '0.0')}")
        self.roi3_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats3:Total_RBV', '0.0')}")
        self.roi4_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats4:Total_RBV', '0.0')}")
        self.stats5_total_value.setText(f"{self.stats_data.get(f'{self.reader.pva_prefix}:Stats5:Total_RBV', '0.0')}")



    def update_image(self):
        if self.reader is not None:
            self.call_id_plot +=1
            image = self.reader.image
            if image is not None:
                image = np.rot90(image, k = self.rot_num)
                if len(image.shape) == 2:
                    min_level, max_level = np.min(image), np.max(image)
                    height, width = image.shape[:2]
                    coordinates = pg.QtCore.QRectF(0, 0, width - 1, height - 1)
                    if self.log_image.isChecked():
                            image = np.log(image + 1)
                            min_level = np.log(min_level + 1)
                            max_level = np.log(max_level + 1)
                    if self.first_plot:
                        self.image_view.setImage(image, autoRange=False, autoLevels=False, levels=(min_level, max_level)) 
                        self.image_view.imageItem.setRect(rect=coordinates)

                        self.max_setting_val.setValue(max_level)
                        self.first_plot = False
                    else:
                        self.image_view.setImage(image, autoRange=False, autoLevels=False)
                        self.image_view.imageItem.setRect(rect=coordinates)
            
                self.horizontal_avg_plot.plot(x=np.mean(image, axis=0), y=np.arange(0,self.reader.shape[1]), clear=True)


                        
                self.min_px_val.setText(f"{min_level:.2f}")
                self.max_px_val.setText(f"{max_level:.2f}")
    

    def update_min_max_setting(self):
        min = float(self.min_setting_val.text())
        max = float(self.max_setting_val.text())
        self.image_view.setLevels(min, max)

            

if __name__ == "__main__":
    
    app = QApplication(sys.argv)

    window = ImageWindow()

    sys.exit(app.exec_())