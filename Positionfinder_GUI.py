# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:01:33 2015

@author: mittelberger
"""

from find_positions import Positionfinder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from tkFileDialog import askopenfilename


class Finder_GUI(object):
    def __init__(self):
        self.Finder = None
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.3)
        self.radio_ax = plt.axes([0.05, 0.7, 0.15, 0.15])
        self.open_ax = plt.axes([0.05, 0.05, 0.075, 0.075])
        self.update_ax = plt.axes([0.125, 0.05, 0.075, 0.075])
        self.openbutton = widgets.Button(self.open_ax, 'Open')
    
    def open_button_clicked(self, event):
        pass
    
    def update_button_clicked(self, event):
        pass
    
    def mouseclick(self, event):
        pass
    
    def radio_changed(self, label):
        pass