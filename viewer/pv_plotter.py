import sys
import json
import copy
import time
import numpy as np
import os.path as osp
import pvaccess as pva
import pyqtgraph as pg
from PyQt5 import uic
from epics import caget
from epics import camonitor
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog


class PVPlotter(QMainWindow):
    def __init__(self, parent):
        super(PVPlotter,self).__init__()
        uic.loadUi('gui/pv_plotter.ui', self)
        self.setWindowTitle("Config")
        