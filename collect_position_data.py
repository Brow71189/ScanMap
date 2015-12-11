# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:16:20 2015

@author: mittelberger
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from ScanMap.find_positions import Positionfinder

class Positioncollector(object):
    
    positionsfilepath = '/home/mittelberger/git/ScanMap/positioncollection.npz'
        
    def __init__(self, **kwargs):
        self.paths = kwargs.get('paths')
        if os.path.isfile(Positioncollector.positionsfilepath):
            self.positionsfile = np.load(Positioncollector.positionsfilepath)
        else:
            self.positionsfile = None
            
        if self.positionsfile is None:
            # the structure is always: (y,x) on the first axis, datasets on the second axis (line or column, depending
            # on if it is x- or y-data) and the actual deviations from the ideal positions on the third axis.
            self.firstlines = np.empty((2, 0, 101))
            self.evenlines = np.empty((2, 0, 101))
            self.oddlines = np.empty((2, 0, 101))
            # mapnames contain name and pixelsize of the maps
            self.mapnames = (np.empty(0)).astype(np.dtype([('mapname', 'U256'), ('pixelsize', 'f32'),
                                                           ('size_overview', 'f32'), ('size_frames', 'f32'),
                                                           ('number_frames', '2f32')]))
        else:
            self.firstlines = self.positionsfile['firstlines']
            self.evenlines = self.positionsfile['evenlines']
            self.oddlines = self.positionsfile['oddlines']
            self.mapnames = self.positionsfile['mapnames']
    
    def get_data(self, folder):
        Finder = Positionfinder(framepath=folder)
        Finder.load_data()
        bad_data = False

        if Finder.optimized_positions is None or Finder.positions is None:
            print('No position data found in ' + folder + '.')
            bad_data = True
        if not Finder.number_frames:
            print('Number of frames not found in ' + folder + '.')
            bad_data = True
        if not Finder.size_overview or not Finder.size_frames:
            print('Size of the overview image and/or the single frames not found in ' + folder + '.')
            bad_data = True
        if Finder.overview is None:
            print('No overview image was found in ' + folder + '.')
            bad_data = True
            
        if not bad_data:
            return (Finder.optimized_positions.copy(), Finder.positions.copy(), 
                    Finder.number_frames, Finder.size_overview, Finder.overview.shape, Finder.size_frames)
    
    def append_data(self, folder):
        data = self.get_data(folder)
        if data is None:
            print('No data received.')
            return
        else:
            optimized_positions, positions, number_frames, size_overview, shape_overview, size_frames = data
            pixelsize = size_overview/np.amax(shape_overview)
        
        if folder in self.mapnames['mapname']:
            print('\nReplacing ' + folder + ' because it is already in the saved data.')
            i = np.where(self.mapnames['mapname'] == folder)[0].item()
        else:
            print('\nAppending ' + folder + '.')
            shape = self.evenlines.shape
            i = shape[1]
    
            resized_evenlines = np.empty((shape[0], i+1, shape[2]), dtype=self.evenlines.dtype)
            resized_evenlines[:, :i, :] = self.evenlines
            self.evenlines = resized_evenlines
    
            resized_oddlines = np.empty((shape[0], i+1, shape[2]), dtype=self.oddlines.dtype)
            resized_oddlines[:, :i, :] = self.oddlines
            self.oddlines = resized_oddlines
    
            resized_firstlines = np.empty((shape[0], i+1, shape[2]), dtype=self.firstlines.dtype)
            resized_firstlines[:, :i, :] = self.firstlines
            self.firstlines = resized_firstlines
            
            resized_mapnames = np.empty(i+1, dtype=self.mapnames.dtype)
            resized_mapnames[:i] = self.mapnames
            self.mapnames = resized_mapnames
        
#        for i in range(self.evenlines.shape[1]):
#            if np.isnan(self.evenlines[0, i]).all():
        self.mapnames['mapname'][i] = folder
        self.mapnames['pixelsize'][i] = pixelsize
        self.mapnames['number_frames'][i] = np.array(number_frames)
        self.mapnames['size_frames'][i] = size_frames
        self.mapnames['size_overview'][i] = size_overview
        differences = positions - optimized_positions
        # Even lines have odd indices
        evenlines = differences[1::2]
        yevenlines = np.median(evenlines[:,:,0], axis=1)
        xevenlines = np.median(evenlines[:,:,1], axis=0)
        inter_yevenlines = interp1d(np.mgrid[0:100:len(yevenlines)*1j], yevenlines)
        inter_xevenlines = interp1d(np.mgrid[0:100:len(xevenlines)*1j], xevenlines)
        self.evenlines[0, i] = inter_yevenlines(np.arange(101))
        self.evenlines[1, i] = inter_xevenlines(np.arange(101))
        # odd lines have even indices
        oddlines = differences[2::2]
        yoddlines = np.median(oddlines[:,:,0], axis=1)
        xoddlines = np.median(oddlines[:,:,1], axis=0)
        inter_yoddlines = interp1d(np.mgrid[0:100:len(yoddlines)*1j], yoddlines)
        inter_xoddlines = interp1d(np.mgrid[0:100:len(xoddlines)*1j], xoddlines)
        self.oddlines[0, i] = inter_yoddlines(np.arange(101))
        self.oddlines[1, i] = inter_xoddlines(np.arange(101))
        # special treatment for first line
        firstlines = differences[0]
        yfirstlines = np.median(firstlines[:,0])
        xfirstlines = firstlines[:,1]
        inter_xfirstlines = interp1d(np.mgrid[0:100:len(xfirstlines)*1j], xfirstlines)
        self.firstlines[0, i] = np.resize(yfirstlines, (101))
        self.firstlines[1, i] = inter_xfirstlines(np.arange(101))
        
#        break
        
    
    def save(self):
        savedict = {}
        savedict['firstlines'] = self.firstlines
        savedict['evenlines'] = self.evenlines
        savedict['oddlines'] = self.oddlines
        savedict['mapnames'] = self.mapnames
        
        np.savez(Positioncollector.positionsfilepath, **savedict)
        
    def main(self, prefix='/3tb/maps_data'):
        for path in self.paths:
            print('\nAppending data of ' + path + '...')
            if prefix:
                path = os.path.join(prefix, path)

            self.append_data(path)
            
        print('Done Appending.\n\nSaving data...')
        self.save()
        print('Done')
        

if __name__ == '__main__':
    paths = ['map_2015_04_15_13_13', 'map_2015_04_16_00_25', 'map_2015_06_30_14_44/all', 'map_2015_08_18_17_07',
             'map_2015_10_19_17_49', 'map_2015_10_19_20_44', 'map_2015_10_19_22_26', 'map_2015_12_10_12_30',
             'map_2015_12_10_17_27']
    
    
    Collector = Positioncollector(paths=paths)
    Collector.main()