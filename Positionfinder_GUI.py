# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:01:33 2015

@author: mittelberger
"""

from find_positions import Positionfinder
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from tkinter.filedialog import askopenfilename
import tkinter
import os
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tifffile

class Finder_GUI(object):
    def __init__(self):
        self.Finder = None
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.075, top=0.975)
        self.radio_ax = plt.axes([0.05, 0.7, 0.15, 0.15])
        #self.radio_ax.set_axis_off()
        self.radio_ax.set_frame_on(False)
        self.radio_ax.set_xticks([])
        self.radio_ax.set_yticks([])
        self.open_ax = plt.axes([0.01, 0.05, 0.07, 0.075])
        self.update_ax = plt.axes([0.09, 0.05, 0.07, 0.075])
        self.save_ax = plt.axes([0.17, 0.05, 0.07, 0.075])
        self.slider_ax = plt.axes([0.35, 0.01, 0.5, 0.02])
        self.openbutton = widgets.Button(self.open_ax, 'Open')
        self.updatebutton = widgets.Button(self.update_ax, 'Update')
        self.savebutton = widgets.Button(self.save_ax, 'Save\nImage')
        self.contrastslider = widgets.Slider(self.slider_ax, 'Contrast', 0, 1, valinit=0.8)
        self.contrastslider.on_changed(self.slider_changed)
        self.openbutton.on_clicked(self.open_button_clicked)
        self.updatebutton.on_clicked(self.update_button_clicked)
        self.savebutton.on_clicked(self.save_button_clicked)
        self.radiobuttons = None
        self.deletemap = None
        self.show_image = None
        self.npzfilepath = None
        self.mapinfofilepath = None
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.mouseclick)
        self.contrastvalue = 0.8
        plt.show()
        
    def slider_changed(self, event):
        self.contrastvalue = self.contrastslider.val
        if self.Finder is not None:
            self.Finder.draw_optimized_positions(lastval=self.contrastvalue)
            self.show_image.set_data(self.Finder.colored_optimized_positions)
            plt.draw()
            
    def save_button_clicked(self, event):
        if self.show_image is None:
            print('No image is open. Nothing to save.')
            return
        savepath = os.path.join(os.path.dirname(self.npzfilepath),
                                'positions_' + os.path.basename(os.path.dirname(GUI.npzfilepath)) + '.tif')
        resolution = tuple(np.array(self.Finder.overview.shape) /
                           np.array((self.Finder.size_overview, self.Finder.size_overview)))
        tifffile_metadata={'kwargs': {'unit': 'nm'}}
        tifffile.imsave(savepath, self.Finder.colored_optimized_positions, resolution=resolution, imagej=True,
                        metadata=tifffile_metadata)
        print('Saved current image to: ' + savepath + '.')
        
    def open_button_clicked(self, event):
        root = tkinter.Tk()
        root.withdraw()
        filepath = askopenfilename()
        if os.path.splitext(os.path.basename(filepath))[1] == '.npz':
            self.npzfilepath = filepath
            self.load_positionfinder()
        else:
            self.mapinfofilepath = filepath
            if self.Finder is not None:
                self.create_radio_buttons()
    
    def update_button_clicked(self, event):
        if self.Finder is None:
            print('Update only works with a Positionfinder results file loaded.')
            return
        if self.Finder.result_is_final:
            print('Cannot overwrite a result that was marked as final.')
            return
        if self.deletemap is None:
            return
        
        shape = self.Finder.optimized_positions.shape
        for j in range(shape[0]):
            for i in range(shape[1]):
                if self.deletemap[j,i]:
                    self.Finder.optimized_positions[j,i] = -1
                    self.deletemap[j,i].remove()
        
        if (self.Finder.optimized_positions == -1).any():
            self.Finder.interpolate_positions()
            self.Finder.relax_positions()
            self.Finder.remove_outliers()
            self.Finder.interpolate_positions()
            self.Finder.draw_optimized_positions(lastval=self.contrastvalue)
            self.show_image.set_data(self.Finder.colored_optimized_positions)
            plt.draw()
            self.deletemap = None
    
    def mouseclick(self, event):
        if event.x > 0.3*self.fig.get_figwidth()*self.fig.get_dpi() and event.xdata is not None \
        and event.ydata is not None:
            if event.button == 1 and event.dblclick:
                self.mark_frame_to_delete(event.xdata, event.ydata)
        #print(event.button, event.xdata, event.ydata, event.x, event.y)
        #print(dir(event))
        #print(event.dblclick)
                    
    def mark_frame_to_delete(self, xposition, yposition):
        shape = self.Finder.optimized_positions.shape
        if self.deletemap == None:
            self.deletemap = np.zeros(shape[:2], dtype=object)
        
        # indices of point with smallest distance to xposition and yposition
        smallestdistance = np.array((0,0, np.inf))
        # positions are saved aas top-left corner, but smallest distance should be when clicking on the
        #center of a frame
        xposition -= self.Finder.scaledframes[0].shape[1]/2
        yposition -= self.Finder.scaledframes[0].shape[0]/2
        for j in range(shape[0]):
            for i in range(shape[1]):
                distance = np.sum((self.Finder.optimized_positions[j,i] - np.array((yposition, xposition)))**2)
                if distance < smallestdistance[2]:
                    smallestdistance[0] = j
                    smallestdistance[1] = i
                    smallestdistance[2] = distance
        if not self.deletemap[tuple(smallestdistance[:2])]:
            res = tuple(self.Finder.optimized_positions[tuple(smallestdistance[:2])] +
                        np.array(self.Finder.scaledframes[0].shape)/2)            
            self.deletemap[tuple(smallestdistance[:2])] = self.ax.plot(res[1], res[0], 'rx', markersize=10, mew=5)[0]
            plt.draw()
        else:
            self.deletemap[tuple(smallestdistance[:2])].remove()
            self.deletemap[tuple(smallestdistance[:2])] = 0
            plt.draw()
    
    def radio_changed(self, label):
        self.Finder.add_info_color(label)
        self.show_image.set_data(self.Finder.colored_optimized_positions)
        plt.draw()
    
    def create_radio_buttons(self):
        self.Finder.read_frame_info_file(name=self.mapinfofilepath)
        height = (len(self.Finder.frameinfo_columns)-1)*0.05
        self.radio_ax.clear()
        self.radio_ax.set_position([0.05, 0.95-height, 0.15, height])
        self.radiobuttons = widgets.RadioButtons(self.radio_ax, self.Finder.frameinfo_columns[1:])
        self.radiobuttons.on_clicked(self.radio_changed)
        plt.draw()
        
    def load_positionfinder(self):
        self.Finder = Positionfinder(os.path.dirname(self.npzfilepath))
        if self.mapinfofilepath and self.radiobuttons is None:
            self.create_radio_buttons()
        self.Finder.load_data()
        self.Finder.draw_optimized_positions(lastval=self.contrastvalue)
        self.show_image = self.ax.imshow(self.Finder.colored_optimized_positions)
        #self.ax.imshow(np.random.rand(1000,1000))
        self.ax.axis('image')
        self.fig.canvas.set_window_title(os.path.basename(os.path.dirname(GUI.npzfilepath)))
        plt.draw()
            
if __name__ == '__main__':
    GUI = Finder_GUI()