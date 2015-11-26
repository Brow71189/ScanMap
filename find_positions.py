# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:33:46 2015

@author: mittelberger
"""

import numpy as np
import cv2
import os
#import tifffile
from scipy.stats import theilslopes
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import matplotlib.pyplot as plt
#from multiprocessing import Pool, Lock, Array
#from ctypes import c_float
    
#def find_position(name, added, over, shape_over, size_frames, size_overview, color):
#    #global added, l, over, shape_over, size_frame, size_overview, color
#    
#    im = cv2.imread(dirpath+name, -1)
#    shape_im = np.shape(im)
#    scale = (size_frames/float(shape_im[0])/(size_overview/float(shape_over[0])))
#    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#    result = cv2.matchTemplate(over, im, method=cv2.TM_SQDIFF_NORMED)
#    maxi = (np.argmin(result), result.shape)
#    l.acquire()
#    try:
#        added[maxi:] += im.ravel()
#        cv2.rectangle(added, (maxi[1],maxi[0]), (maxi[1]+im.shape[1], maxi[0]+im.shape[0]), color, thickness=2)
#        cv2.putText(added, name[0:4], (maxi[1]-4,maxi[0]-2), cv2.FONT_HERSHEY_PLAIN, 3, color, thickness=2)
#    except Exception as detail:
#        print('Error in '+name+': '+str(detail))
#    finally:
#        l.release()
#
#def init(l):
#    global lock
#    lock=l
class Positionfinder(object):
    
    def __init__(self, **kwargs):
        self.overview = kwargs.get('overview')
        self.framelist = kwargs.get('framelist', [])
        self.number_frames = kwargs.get('number_frames')
        self.scaledframes = [] # has the form (framenumber, scaled image)
        self.size_overview = kwargs.get('size_overview')
        self.size_frames = kwargs.get('size_frames')
        self.framepath = kwargs.get('framepath')
        self.leftborder = [] # items have the form ((framenumber, correlation value), (position y, position x))
        self.topborder = []
        self.rightborder = []
        self.bottomborder = []
        self.allborders = []
        self.colored_borders = None
        self.colored_positions = None
        self.corners = [] # in the order top-left, top-right, bottom-right, bottom-left. (y, x)
        self.border_parameters = [] # parameters of the lines connecting the four corners
                                    # Order: Top, right, bottom, left
        self.positions = None
        self.optimized_positions = None
        self.data_to_save = kwargs.get('data_to_save', ['scaledframes', 'leftborder', 'topborder', 'rightborder',
                                                        'bottomborder', 'allborders', 'corners', 'border_parameters',
                                                        'positions', 'optimized_positions'])
        self.data_to_load = kwargs.get('data_to_load', ['scaledframes', 'leftborder', 'topborder', 'rightborder',
                                                        'bottomborder', 'allborders', 'corners', 'border_parameters',
                                                        'positions', 'optimized_positions'])
        self.loaded_data = []
        self.positions_to_relax = []
        
    
    def save_data(self):
        savedict = {}
        for item in self.data_to_save:
            if getattr(self, item) is None or len(getattr(self, item)) < 1:
                self.data_to_save.remove(item)
            else:
                savedict[item] = np.array(getattr(self, item), dtype=np.float32)
        savedict['data_to_save'] = self.data_to_save
        
        np.savez(os.path.join(self.framepath, 'Positionfinder_data.npz'), **savedict)
        
        print('\nSaved: ')
        for item in self.data_to_save:
            print(item)
        print('to \"' + os.path.join(self.framepath, 'Positionfinder_data.npz' + '\"'))
    
    def load_data(self):
        if os.path.isfile(os.path.join(self.framepath, 'Positionfinder_data.npz')):
            with np.load(os.path.join(self.framepath, 'Positionfinder_data.npz')) as data:
                data_in_save = data['data_to_save']
                for item in self.data_to_load:
                    if item in data_in_save:
                        print('Loading ' + item + ' from disk.')
                        setattr(self, item, data[item])
                        self.loaded_data.append(item)
        print('Finished loading.')
    
    def scale_images(self):
        print('\nStarted scaling of images...')
        self.framelist.sort()
        if 'scaledframes' in self.loaded_data:
            print('Loaded scaled frames from disk.')
        else:
            image = ndimage.imread(os.path.join(self.framepath, self.framelist[0]))
            scale = (self.size_frames/float(image.shape[0])/(self.size_overview/float(self.overview.shape[0])))
            scaledframe = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self.scaledframes.append(scaledframe)
            
            for i in range(1, len(self.framelist)):
                image = ndimage.imread(os.path.join(self.framepath, self.framelist[i]))
                scaledframe = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                self.scaledframes.append(scaledframe)
                if i%(np.rint(len(self.framelist)/50)) == 0:
                    print('Done {:.0f} / {:.0f} ({:.1%})'.format(i, len(self.framelist), i/len(self.framelist)),
                          end='\r')

#            np.save(os.path.join(self.framepath, 'scaledframes.npy'), scaledframes)
#            print('\nSaved scaled frames to ' + os.path.join(self.framepath, 'scaledframes.npy'))
        
        print('Finished scaling.')            
        
    def find_position(self, framenumber, searchrange=None):
        # searchrange in c-order, e.g. ((lower-y, higher-y), (lower-x, higher-x))
        if searchrange is None:
            searchrange = ((0, None), (0, None))
        
        result = cv2.matchTemplate(self.overview[searchrange[0][0]:searchrange[0][1],
                                                 searchrange[1][0]:searchrange[1][1]],
                                   self.scaledframes[framenumber], method=cv2.TM_CCOEFF_NORMED)
        
        maxi = np.array(np.unravel_index(np.argmax(result), np.shape(result)))
        return (tuple(maxi + np.array((searchrange[0][0], searchrange[1][0]))), np.amax(result))
        
        
    def find_borders(self, border_min_correlation=0.6):
        min_correlation = border_min_correlation
        print('\nStarted finding the borders of the map...')
        #positions = []
        if 'leftborder' in self.loaded_data and 'topborder' in self.loaded_data and 'rightborder' in self.loaded_data \
        and 'bottomborder' in self.loaded_data and 'allborders':
            
            print('Loaded borders from disk.')
        else:
            for i in range(len(self.scaledframes)):
                if i < self.number_frames[0]:
                    position = self.find_position(i)
                    if position[1] > min_correlation:
                        self.topborder.append(((i, position[1]), position[0]))
                        self.allborders.append(((i, position[1]), position[0]))
                elif i > len(self.scaledframes)-self.number_frames[0]:
                    position = self.find_position(i)
                    if position[1] > min_correlation:
                        self.bottomborder.append(((i, position[1]), position[0]))
                        self.allborders.append(((i, position[1]), position[0]))
                elif i%self.number_frames[0] == 0:
                    position = self.find_position(i)
                    if position[1] > min_correlation:
                        self.leftborder.append(((i, position[1]), position[0]))
                        self.allborders.append(((i, position[1]), position[0]))
                elif (i+1)%self.number_frames[0] == 0:
                    position = self.find_position(i)
                    if position[1] > min_correlation:
                        self.rightborder.append(((i, position[1]), position[0]))
                        self.allborders.append(((i, position[1]), position[0]))
    
                if i%(np.rint(len(self.scaledframes)/50)) == 0:
                    print('Done {:.0f} / {:.0f} ({:.1%})'.format(i, len(self.scaledframes), i/len(self.scaledframes)),
                          end='\r')
        
        print('Finished finding the borders.')
    
    def draw_borders(self):
        added = np.zeros((3,)+np.shape(self.overview))
        added[2] = self.overview
        color = float(np.mean(self.overview))
        for i in range(len(self.allborders)):
            added[1,self.allborders[i][1][0]:self.allborders[i][1][0]+self.scaledframes[i].shape[0],
                  self.allborders[i][1][1]:self.allborders[i][1][1]+self.scaledframes[i].shape[1]] = \
                  self.scaledframes[self.allborders[i][0][0]]
            cv2.putText(added[0], str(int(self.allborders[i][0][0])), (int(np.rint(self.allborders[i][1][1]-4)),
                        int(np.rint(self.allborders[i][1][0]-2))), cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
        
        added2 = added.copy()
        added2 = np.swapaxes(added2, 0, 2)
        added2 = np.swapaxes(added2, 0, 1)
        percentile95 = np.percentile(added, 95)
        added2[added2>percentile95] = percentile95
        added2 *= 255/percentile95
        added2 = added2.astype('uint8')
        
        self.colored_borders = added2
        
    def draw_positions(self):
        added = np.zeros((3,)+np.shape(self.overview))
        added[2] = self.overview
        color = float(np.mean(self.overview))
        for i in range(len(self.scaledframes)):
            coordinates = np.unravel_index(i, (self.number_frames[1], self.number_frames[0]))
            added[1,self.positions[coordinates][0]:self.positions[coordinates][0]+self.scaledframes[i].shape[0],
                  self.positions[coordinates][1]:self.positions[coordinates][1]+self.scaledframes[i].shape[1]] = \
                  self.scaledframes[i]
            cv2.putText(added[0], str(int(i)), (int(np.rint(self.positions[coordinates][1]-4)),
                        int(np.rint(self.positions[coordinates][0]-2))), cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
        
        added2 = added.copy()
        added2 = np.swapaxes(added2, 0, 2)
        added2 = np.swapaxes(added2, 0, 1)
        percentile95 = np.percentile(added, 95)
        added2[added2>percentile95] = percentile95
        added2 *= 255/percentile95
        added2 = added2.astype('uint8')
        
        self.colored_positions = added2
        
    def draw_optimized_positions(self):
        added = np.zeros((3,)+np.shape(self.overview))
        added[2] = self.overview
        color = float(np.mean(self.overview))
        for i in range(len(self.scaledframes)):
            coordinates = np.unravel_index(i, (self.number_frames[1], self.number_frames[0]))
            position = self.optimized_positions[coordinates]
            if (position > -1).all():
                added[1,position[0]:position[0] + self.scaledframes[i].shape[0],
                      position[1]:position[1]+self.scaledframes[i].shape[1]] = self.scaledframes[i]
                cv2.putText(added[0], str(int(i)), (int(np.rint(position[1]-4)), int(np.rint(position[0]-2))),
                            cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
        
        added2 = added.copy()
        added2 = np.swapaxes(added2, 0, 2)
        added2 = np.swapaxes(added2, 0, 1)
        percentile95 = np.percentile(added, 95)
        added2[added2>percentile95] = percentile95
        added2 *= 255/percentile95
        added2 = added2.astype('uint8')
        
        self.colored_optimized_positions = added2
    
    def find_corners(self):
        # Fit lines to borders
        print('\nFinding corners of the map.')
        if 'corners' in self.loaded_data:
            print('Loaded corners from disk.')
        else:
            top = np.array(self.topborder)
#            topslope = np.median(top[:, 1, 0][1:]-top[:, 1, 0][:-1]) / np.median(top[:, 1, 1][1:]-top[:, 1, 1][:-1])
#            topoffset = np.median(top[:, 1, 0]) - topslope * np.median(top[:, 1, 1])
#            print(topslope)
#            print(topoffset)
#            topperc1 = np.percentile(top[:, 1, 0], 20)
#            topperc2 = np.percentile(top[:, 1, 0], 80)
#            top = top[:, 1][top[:, 1, 0]>topperc1]
#            top = top[top[:, 0]<topperc2]
            self.border_parameters.append(self.fit_line(top[:, 1, 1], top[:, 1, 0]))

            right = np.array(self.rightborder)
#            rightslope = np.median(right[:, 1, 0][1:]-right[:, 1, 0][:-1]) / \
#                         np.median(right[:, 1, 1][1:]-right[:, 1, 1][:-1])
#            rightoffset = -rightslope*np.median(right[:, 1, 1])
#            rightperc1 = np.percentile(right[:, 1, 1], 20)
#            rightperc2 = np.percentile(right[:, 1, 1], 80)
#            right = right[:, 1][right[:, 1, 1]>rightperc1]
#            right = right[right[:, 1]<rightperc2]
            self.border_parameters.append(self.fit_line(right[:, 1, 1], right[:, 1, 0]))

            bottom = np.array(self.bottomborder)
#            bottomslope = np.median(bottom[:, 1, 0][1:]-bottom[:, 1, 0][:-1]) / \
#                         np.median(bottom[:, 1, 1][1:]-bottom[:, 1, 1][:-1])
#            bottomoffset = np.median(bottom[:, 1, 0])
#            bottomperc1 = np.percentile(bottom[:, 1, 0], 20)
#            bottomperc2 = np.percentile(bottom[:, 1, 0], 80)
#            bottom = bottom[:, 1][bottom[:, 1, 0]>bottomperc1]
#            bottom = bottom[bottom[:, 0]<bottomperc2]
            self.border_parameters.append(self.fit_line(bottom[:, 1, 1], bottom[:, 1, 0]))

            left = np.array(self.leftborder)
#            leftslope = np.median(left[:, 1, 0][1:]-left[:, 1, 0][:-1]) / \
#                         np.median(left[:, 1, 1][1:]-left[:, 1, 1][:-1])
#            leftoffset = -rightslope*np.median(left[:, 1, 1])
#            leftperc1 = np.percentile(left[:, 1, 1], 20)
#            leftperc2 = np.percentile(left[:, 1, 1], 80)
#            left = left[:, 1][left[:, 1, 1]>leftperc1]
#            left = left[left[:, 1]<leftperc2]
            self.border_parameters.append(self.fit_line(left[:, 1, 1], left[:, 1, 0]))
            
            # Now calculate pairwise intersections to get corners
            # pairs to calculate to get top-left, top-right, bottom-right and bottom-left corner:
            pairs = [(0, 3), (0, 1), (2, 1), (2, 3)]
            
            for pair in pairs:
                self.corners.append(self.lines_intersection(*(tuple(self.border_parameters[pair[0]]) +
                                                              tuple(self.border_parameters[pair[1]]))))
        print('Finished finding the corners.\n')
        
    def plot_border_fits(self):
        # Top
#        top = np.array(self.topborder)
#        topxrange = np.arange(np.amin(top[:, 1, 1]), np.amax(top[:, 1, 1]))
#        plt.gci()
#        plt.plot(topxrange, self.linear(topxrange, *self.border_parameters[0]), 'r-')
#        # Right
#        #right = np.array(self.rightborder)
#        rightxrange = np.arange(self.overview.shape[1]-1) #np.arange(np.amin(right[:, 1, 1]), np.amax(right[:, 1, 1]))
#        righty = self.linear(rightxrange, *self.border_parameters[1])
#        rightxrange = rightxrange[righty<self.overview.shape[0]-200]
#        righty = righty[righty<self.overview.shape[0]-200]
#        rightxrange = rightxrange[righty>200]
#        righty = righty[righty>200]
#        plt.gci()
#        plt.plot(rightxrange, righty, 'r-')
#        # Bottom
#        bottom = np.array(self.bottomborder)
#        bottomxrange = np.arange(np.amin(bottom[:, 1, 1]), np.amax(bottom[:, 1, 1]))
#        plt.gci()
#        plt.plot(bottomxrange, self.linear(bottomxrange, *self.border_parameters[2]), 'r-')
#        # Left
#        #left = np.array(self.leftborder)
#        leftxrange = np.arange(self.overview.shape[1]-1) #leftxrange = np.arange(np.amin(left[:, 1, 1]), np.amax(left[:, 1, 1]))
#        lefty = self.linear(leftxrange, *self.border_parameters[3])
#        leftxrange = leftxrange[lefty<self.overview.shape[0]-100]
#        lefty = lefty[lefty<self.overview.shape[0]-100]
#        leftxrange = leftxrange[lefty>100]
#        lefty = lefty[lefty>100]
#        plt.gci()
#        plt.plot(leftxrange, lefty, 'r-')
    
        corners = self.corners.copy()
        corners.append(self.corners[0])
        corners = np.array(corners)
        #plt.gci()
        return plt.plot(corners[:, 1], corners[:, 0], 'r-')
        
    def lines_intersection(self, a1, b1, a2, b2):
        x = (b2-b1)/(a1-a2)
        y = (a1*b2 - a2*b1)/(a1-a2)
        return np.array((y, x))
    
    def fit_line(self, x, y, startvalues=None):
#        popt, pcov = curve_fit(self.linear, x, y, p0=startvalues)
#        return popt
        #slope = np.median((y[1:]-y[:-1]))/np.median((x[1:]-x[:-1]))
        #slope = np.median((y[1:]-y[:-1])/(x[1:]-x[:-1]))
        #return np.array((slope, np.median(y[1:-1]) - slope*np.median(x[1:-1])))
        return theilslopes(y, x)[0:2]
        
    def linear(self, x, a, b):
        return a*x + b
    
    def get_correct_position(self, coordinates):
        coordinates = np.array(coordinates)
        topXpos = self.corners[0] + (coordinates[1]/(self.number_frames[0]-1)) * (self.corners[1] - self.corners[0])
        botXpos = self.corners[3] + (coordinates[1]/(self.number_frames[0]-1)) * (self.corners[2] - self.corners[3])
        
        return topXpos + (coordinates[0]/(self.number_frames[1]-1)) * (botXpos - topXpos)
        
    def place_subframes(self):
        print('\nPlacing all frames on a regular grid.')
        if 'positions' in self.loaded_data:
            print('Loaded initial positions from disk.')
        else:
            number_frames = (self.number_frames[1], self.number_frames[0])
            self.positions = np.empty(number_frames + (2,))
            for i in range(len(self.scaledframes)):
                coordinates = np.unravel_index(i, number_frames)
                self.positions[coordinates] = self.get_correct_position(coordinates)
        print('Finished placing frames to their initial positions.')
        
    def optimize_positions(self, optimize_searchrange=1, optimize_min_correlation=0.75):
        searchradius = optimize_searchrange
        min_correlation = optimize_min_correlation
        print('\nOptimizing positions of the frames...')
        if 'optimized_positions' in self.loaded_data:
            print('Loaded optimized positions from disk.')
        else:
            self.optimized_positions = np.ones(np.shape(self.positions)) * -1
            number_frames = (self.number_frames[1], self.number_frames[0])
            for j in range(number_frames[0]):
                for i in range(number_frames[1]):
                    number = j*number_frames[1]+i
                    shape = np.shape(self.scaledframes[number])
                    searchrange = ((self.positions[j,i][0]-searchradius*shape[0],
                                    self.positions[j,i][0]+searchradius*shape[0]),
                                   (self.positions[j,i][1]-searchradius*shape[1],
                                    self.positions[j,i][1]+searchradius*shape[1]))
                    position = self.find_position(number, searchrange=searchrange)
                    if position[1] > min_correlation:
                        self.optimized_positions[j,i] = np.array(position[0])
                        
        print('Finished optimizing frame positions.')
        
    def relax_positions(self, relax_searchrange=3, relax_min_correlation=0.6):
        searchradius = relax_searchrange
        min_correlation = relax_min_correlation
        print('\nRelaxing positions of the given frames...')
        if len(self.positions_to_relax) == 0:
            print('No positions to relax.')
        else:
            number_frames = (self.number_frames[1], self.number_frames[0])
            for j,i in self.positions_to_relax:
                number = j*number_frames[1]+i
                shape = np.shape(self.scaledframes[number])
                searchrange = ((self.optimized_positions[j,i][0]-searchradius*shape[0],
                                self.optimized_positions[j,i][0]+searchradius*shape[0]),
                               (self.optimized_positions[j,i][1]-searchradius*shape[1],
                                self.optimized_positions[j,i][1]+searchradius*shape[1]))
                position = self.find_position(number, searchrange=searchrange)
                if position[1] > min_correlation:
                    self.optimized_positions[j,i] = np.array(position[0])
                else:
                    self.optimized_positions[j,i] = np.array((-1, -1))
        self.positions_to_relax = []
        print('Finished relaxing frame positions.')
        
    def interpolate_positions(self):
        print('\nInterpolating positions where correct position was not found yet...')
        if not (self.optimized_positions < 0).any():
            print('All positions already found. Nothing to do for me here.')
        else:
            number_frames = (self.number_frames[1], self.number_frames[0])
            for j in range(number_frames[0]):
                for i in range(number_frames[1]):
                    # check if at a bad position
                    if (self.optimized_positions[j, i] < 0).all():
                        left = right = i
                        # get position of next found frame to the left and right of current frame
                        while left > 0:
                            left -= 1
                            if (self.optimized_positions[j, left] > -1).all():
                                break
                        else:
                            left = -1
                        while right < number_frames[1]-1:
                            right += 1
                            if (self.optimized_positions[j, right] > -1).all():
                                break
                        else:
                            right = -1
                        
                        if i == 0:
                            right2 = right
                            # Only extrapolate when close to left border to avoid large errors
                            while right2 < 5:
                                right2 += 1
                                if (self.optimized_positions[j, right2] > -1).all():
                                    break
                            else:
                                right2 = -1
                            
                            if right2 > -1:
                                newposition = self.optimized_positions[j, right] - right * (
                                             (self.optimized_positions[j, right2] -
                                              self.optimized_positions[j, right]) / (right2 - right))
                            else:
                                newposition = (self.get_correct_position((j,-1)) +
                                               1/right*self.optimized_positions[j, right]) / (1/right + 1)
                        elif i == number_frames[1]-1:
                            left2 = left
                            # Only etrapolate when close to right border to avoid large errors
                            while left2 > number_frames[1] - 5:
                                left2 -= 1
                                if (self.optimized_positions[j, left2] > -1).all():
                                    break
                            else:
                                left2 = -1
                            
                            if left2 > -1:
                                print(left, left2)
                                newposition = self.optimized_positions[j, left] + (number_frames[1]-left-1) * (
                                             (self.optimized_positions[j, left] -
                                              self.optimized_positions[j, left2]) / (left - left2))
                            else:
                                newposition = (1/(i-left)*self.optimized_positions[j, left] +
                                               self.get_correct_position((j, number_frames[1]))) / \
                                               (1 + 1/(i-left))
                        # interpolate position if frame has two neighbors
                        elif left > -1 and right > -1:
                            newposition = (1/(i-left)*self.optimized_positions[j, left] +
                                           1/(right-i)*self.optimized_positions[j, right]) / (1/(right-i) + 1/(i-left))
                        # Use left border if there is no left neighbor
                        elif left < 0 and right > -1:
                            newposition = (1/i*self.get_correct_position((j,0)) +
                                           1/(right-i)*self.optimized_positions[j, right]) / (1/(right-i) + 1/i)
                        # use right border if there is no right neighbor
                        elif left > -1 and right < 0:
                            newposition = (1/(i-left) * self.optimized_positions[j, left] +
                                           1/(number_frames[1]-1-i) *
                                           self.get_correct_position((j,number_frames[1]-1))) / \
                                           (1/(number_frames[1]-1-i) + 1/(i-left))
                        else:
                            newposition = None
                            
                        if newposition is not None:
                            self.optimized_positions[j,i] = np.array(newposition)
                            self.positions_to_relax.append((j,i))
                            
            print('Done interpolating positions.')
            
    def remove_outliers(self, outlier_tolerance=1):
        tolerance = outlier_tolerance
        print('\nStarted removing outliers in the lines.')
        # Checks for image positions that deviate a lot from a line in the map
        for j in range(self.optimized_positions.shape[0]):
            xfitdata = self.optimized_positions[j, :, 1][self.optimized_positions[j, :, 1] > -1]
            yfitdata = self.optimized_positions[j, :, 0][self.optimized_positions[j, :, 0] > -1]
            linefit = self.fit_line(xfitdata, yfitdata)
            for i in range(self.optimized_positions.shape[1]):
                if (self.optimized_positions[j, i] > -1).all():
                    position = self.linear(self.optimized_positions[j, i, 1], *linefit)
                    #print(position)
                    if np.abs(self.optimized_positions[j, i, 0] - position) > tolerance*self.scaledframes[0].shape[0]:
                        self.optimized_positions[j, i] = np.array((-1, -1))
        print('\nFinished removing outliers.')
        
    def get_framelist(self, extension='tif', separator='_', name_overview='Overview', choose_frame=0):
        frames = os.listdir(self.framepath)
        
        for name in frames:
            splitname = os.path.splitext(name)
            if splitname[1].lower().endswith(extension.lower()):
                if self.overview is None and splitname[0].lower().startswith(name_overview.lower()):
                    self.overview = np.array(cv2.imread(os.path.join(self.framepath, name), -1))
                    self.overview = cv2.GaussianBlur(self.overview, None, 1)
                else:
                    splitbase = splitname[0].split(separator)
                    try:
                        int(splitbase[0])
                    except:
                        pass
                    else:
                        if len(splitbase) == 3:
                            self.framelist.append(name)
                        elif len(splitbase) == 4:
                            try:
                                number = int(splitbase[-1])
                            except:
                                pass
                            else:
                                if number == choose_frame:
                                    self.framelist.append(name)
        
        self.framelist.sort()
                    
    
    def main(self, iterations=10, plot_results=True, save_plots=False, **kwargs):
        """
        kwargs takes all additional parameters for the subfunctions called here. These are in detail:
            For get_framelist:
                - extension : String that specifies the file extension of the single frames (default: tif)
                - separator : String that specifies the separator used to separate the frame number from the residual
                              file name (default: _) (Frame number has to be at the beginning of the filename)
                - name_overview : String that specifies how the overview image is called. (default: Overview) (Overview
                                  frame name has to start with this)
                - choose_frame : Number that specifies which frame will be taken if there exist more frames with the
                                 same number. (default: 0) (Number has to be at the end of the frame name)
                
            For find_borders:
                - border_min_correlation : Number between 0 and 1 (default: 0.6)
                
            For optimize_positions:
                - optimize_min_correlation : Number between 0 and 1 (default: 0.75)
                - optimize_searchrange : Number that specifies in which radius around the initial position of the frames
                                         the search for a better match should be carried out in units of image size
                                         (default: 3). It must be bigger than 0.5
                                         
            For remove_outliers:
                - outlier_tolerance : Number that specifies the maximum deviation of frames within a line from the line
                                      in units of image size (default: 1)
                                      
            For relax_positions:
                - relax_min_correlation : Number between 0 and 1 (default: 0.6)
                - relax_searchrange : Number that specifies in which radius around the previous position of the frames
                                      the search for a better match should be carried out in units of image size
                                      (default: 3). It must be bigger than 0.5
        """
        # Check for input:
        get_framelist_params = {}
        find_borders_params = {}
        optimize_positions_params = {}
        remove_outliers_params = {}
        relax_positions_params = {}
        
        if plot_results:
            fig = plt.figure()
            plots = []
            
        for key, value in kwargs.items():
            if key in ['extension', 'separator', 'name_overview', 'coose_frame']:
                get_framelist_params[key] = value
            elif key in ['border_min_correlation']:
                find_borders_params[key] = value
            elif key in ['optimize_min_correlation', 'optimize_searchrange']:
                optimize_positions_params[key] = value
            elif key in ['outlier_tolerance']:
                remove_outliers_params[key] = value
            elif key in ['relax_min_correlation', 'relax_searchrange']:
                relax_positions_params[key] = value
        
        try:        
            self.get_framelist(**get_framelist_params)    
            self.load_data()
            self.scale_images()
            self.find_borders(**find_borders_params)
            self.find_corners()
            self.place_subframes()
            self.optimize_positions(**optimize_positions_params)
            
            if plot_results:
                print('\nPlotting stuff...')
                self.draw_borders()
                self.draw_positions()
                self.draw_optimized_positions()
                plots.append((plt.imshow(self.colored_borders), ))
                plots.append((plt.imshow(self.colored_borders), self.plot_border_fits()[0]))
                #plots.append(self.plot_border_fits())
                plots.append((plt.imshow(self.colored_positions), ))
                plots.append((plt.imshow(self.colored_optimized_positions), ))
                print('Finished plotting.')
            
            for i in range(iterations):
                self.remove_outliers(**remove_outliers_params)
                self.interpolate_positions()
                self.relax_positions(**relax_positions_params)
                if plot_results:
                    self.draw_optimized_positions()
                    plots.append((plt.imshow(self.colored_optimized_positions), ))
        
            self.interpolate_positions()
            if plot_results:
                self.draw_optimized_positions()
                plots.append((plt.imshow(self.colored_optimized_positions), ))
                #plt.ioff()
                im_ani = animation.ArtistAnimation(fig, plots, interval=3000, repeat_delay=1000, blit=True)
                
            if save_plots and plot_results:
                im_ani.save(os.path.join(self.framepath, 'positions_result.avi'), dpi=300, bitrate=100000)
                
            #if plot_results:
                #plt.show()

        finally:
            self.save_data()
            
            
if __name__=='__main__':
    
    dirpath = '/3tb/maps_data/map_2015_10_19_17_49'
    
#    overview = '/3tb/maps_data/map_2015_08_18_17_07/Overview_1576.59891322_nm.tif'
    
    size_overview = 1587 #nm
    size_frames = 20 #nm
    #number of frames in x- and y-direction
    number_frames = (34,32)
    #borders of the area where the first frame is expected (in % of the overview image)
    #has to be a tuple of the form (lower-y, higher-y, lower-x, higher-x)
#    area_first_frame = (15,25,15,25) #%
    #enter search area for the following images. Search is always performed around the position of the last image.
    #tuple hast to have the form (negtive-y, positive-y, negative-x, positive-x)
#    tolerance_within_rows = (20,20,20,20)
#    tolerance_within_columns = (20,20,20,20)
    #Enter the number of frames that should be included here. Type None to use all frames.
#    number_frames_included = -1
    #
#    bad_correlation_threshold = 0
    
    
    
##    area_first_frame = np.array(area_first_frame)/100.0
#    frames = os.listdir(dirpath)
#    frames.sort()
#    matched_frames = []
#    for name in frames:
#        num = None
#        try:
#            int(name[0:4])
#        except:
#            pass
#        else:
#            try:
#                num = int(name[-6:-4])
#            except:
#                pass
#            else:
#                if num is 0:
#                    matched_frames.append(name)
#
#    matched_frames.sort()
#    over = np.array(cv2.imread(overview, -1))#[1023:3072, 0:2048]*3
#    shape_over = np.shape(over)
#    over = cv2.GaussianBlur(over, None, 1)
    
#    Finder = Positionfinder(overview=over, framelist=matched_frames, number_frames=number_frames,
#                            size_overview=size_overview, size_frames=size_frames, framepath=dirpath)
    
    Finder = Positionfinder(number_frames=number_frames, size_overview=size_overview, size_frames=size_frames,
                            framepath=dirpath)

#    Finder.data_to_load.remove('optimized_positions')
#    Finder.data_to_load.remove('positions')
#    Finder.data_to_load.remove('borders')
    Finder.data_to_load = ['scaledframes', 'leftborder', 'topborder', 'rightborder', 'bottomborder', 'allborders']
    Finder.main(iterations=10, save_plots=False, plot_results=False, border_min_correlation=0.6, optimize_searchrange=3,
                outlier_tolerance=0.8, relax_searchrange=3, relax_min_correlation=0.6)
#    Finder.get_framelist()    
#    Finder.load_data()
#    Finder.scale_images()
#    Finder.find_borders(min_correlation=0.6)
#    Finder.find_corners()
#    Finder.place_subframes()
#    Finder.optimize_positions(searchradius=3)
#    
#    Finder.draw_borders()
#    fig1 = plt.figure(1)
#    plt.imshow(Finder.colored_borders)
#    Finder.plot_border_fits()
##    
##    Finder.draw_positions()
##    fig2 = plt.figure(2)
##    plt.imshow(Finder.colored_positions)
#    
#    Finder.draw_optimized_positions()
#    fig3 = plt.figure(3)
#    plt.imshow(Finder.colored_optimized_positions)
#
#    for i in range(20):
#        Finder.remove_outliers(tolerance=0.8)
##    Finder.draw_optimized_positions()
##    fig4 = plt.figure(4)
##    plt.imshow(Finder.colored_optimized_positions)
#    
#        Finder.interpolate_positions()
##    Finder.draw_optimized_positions()
##    fig5 = plt.figure(5)
##    plt.imshow(Finder.colored_optimized_positions)
#    
#        Finder.relax_positions(searchradius=3, min_correlation=0.6)
#        
#    Finder.draw_optimized_positions()
#    fig6 = plt.figure(6)
#    plt.imshow(Finder.colored_optimized_positions)
#    
#    Finder.interpolate_positions()
#    Finder.draw_optimized_positions()
#    fig7 = plt.figure(7)
#    plt.imshow(Finder.colored_optimized_positions)
#    
#    Finder.save_data()
##    l = Lock()
##    added = Array(c_float, over.ravel(), lock=l)
#    added = np.zeros((3,)+np.shape(over))
#    #added = over.copy()
#    added[2] += over
#    color = float(np.mean(over))
#    position_list = []
#    correlation_list = []
#    in_row_distance_list = []
#    in_column_distance_list = []
#    
#    #find position of first frame
#    name = matched_frames[0]
#    im = np.array(cv2.imread(dirpath+name, -1))
#    shape_im = np.shape(im)
#    scale = (size_frames/float(shape_im[0])/(size_overview/float(shape_over[0])))
#    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#    #im = cv2.GaussianBlur(im, None, 3)
#    result = cv2.matchTemplate(over[int(area_first_frame[0]*shape_over[0]):int(area_first_frame[1]*shape_over[0]),
#                                    int(area_first_frame[2]*shape_over[1]):int(area_first_frame[3]*shape_over[1])],
#                                    im, method=cv2.TM_CCOEFF_NORMED)
#
#    maxi = tuple(np.array(np.unravel_index(np.argmax(result), result.shape), dtype='int') + 
#                 np.array((area_first_frame[0]*shape_over[0], area_first_frame[2]*shape_over[1]), dtype='int'))
#    correlation_list.append(np.amax(result))
#    position_list.append((int(name[0:4]), ) + maxi)
#    added[1,maxi[0]:maxi[0]+im.shape[0], maxi[1]:maxi[1]+im.shape[1]] += im
#    cv2.putText(added[0], str(int(name[0:4])), (maxi[1]-4,maxi[0]-2), cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
#    
#    if number_frames_included < 1:
#        number_frames_included = None
#        
#    counter = 0
#    for name in matched_frames[1:number_frames_included]:
#        im = np.array(cv2.imread(dirpath+name, -1))#*3
#        shape_im = np.shape(im)
#        scale = (size_frames/float(shape_im[0])/(size_overview/float(shape_over[0])))
#        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#        if int(name[0:4])%number_frames[0] == 0 and int(name[0:4]) != 0:
#            subarray = np.array(( position_list[counter-number_frames[0]+1][1]-int(tolerance_within_columns[0]*scale*shape_im[0]),
#                                  position_list[counter-number_frames[0]+1][1]+int(tolerance_within_columns[1]*scale*shape_im[0]),
#                                  position_list[counter-number_frames[0]+1][2]-int(tolerance_within_columns[2]*scale*shape_im[0]),
#                                  position_list[counter-number_frames[0]+1][2]+int(tolerance_within_columns[3]*scale*shape_im[0]) ))
#        else:
#            subarray = np.array(( position_list[counter][1]-int(tolerance_within_rows[0]*scale*shape_im[0]),
#                                  position_list[counter][1]+int(tolerance_within_rows[1]*scale*shape_im[0]),
#                                  position_list[counter][2]-int(tolerance_within_rows[2]*scale*shape_im[0]),
#                                  position_list[counter][2]+int(tolerance_within_rows[3]*scale*shape_im[0]) ))
#        subarray[subarray>= shape_over[0]] = shape_over[0]-1
#        subarray[subarray< 0] = 0
#        result = cv2.matchTemplate(over[subarray[0]:subarray[1],subarray[2]:subarray[3]], im, method=cv2.TM_CCOEFF_NORMED)
#        corr_max = np.amax(result)
#        maxi = tuple(np.array(np.unravel_index(np.argmax(result), result.shape), dtype='int') + np.array((subarray[0], subarray[2]), dtype='int'))
#        if int(name[0:4])%number_frames[0] == 0 and int(name[0:4]) != 0:
#            in_column_distance_list.append(np.array(maxi) - np.array(position_list[counter-number_frames[0]+1][1:]))
#        else:
#            if corr_max < bad_correlation_threshold and len(in_row_distance_list) > 1:
#                maxi = (position_list[counter][1] + int(np.median(np.array(in_row_distance_list)[:,0])),
#                        position_list[counter][2] + int(np.median(np.array(in_row_distance_list)[:,1])))
#            in_row_distance_list.append(np.array(maxi) - np.array(position_list[counter][1:]))
#            
#        correlation_list.append(corr_max)
#        position_list.append((int(name[0:4]), ) + maxi)
#        added[1,maxi[0]:maxi[0]+im.shape[0], maxi[1]:maxi[1]+im.shape[1]] += im
#        #cv2.rectangle(added, (maxi[1],maxi[0]), (maxi[1]+im.shape[1], maxi[0]+im.shape[0]), color, thickness=2)
#        cv2.putText(added[0], str(int(name[0:4])), (maxi[1]-4,maxi[0]-2), cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
#        counter += 1
    
#    added2 = added.copy()
#    added2 = np.swapaxes(added2, 0, 2)
#    added2 = np.swapaxes(added2, 0, 1)
#    percentile95 = np.percentile(added, 95)
#    added2[added2>percentile95] = percentile95
#    added2 *= 255/percentile95
#    added2 = added2.astype('uint8')
    
#    correlation_array = np.ones(np.prod(np.array(number_frames)), dtype='float32')
#    correlation_array[0:len(correlation_list)] = np.array(correlation_list, dtype='float32')
#    correlation_array = correlation_array.reshape((number_frames[1], number_frames[0]))
#    in_row_distance_array = np.array(in_row_distance_list, dtype='float32')
#    in_column_distance_array = np.array(in_column_distance_list, dtype='float32')
#    
#    tifffile.imsave(dirpath+'positions.tif', added2)
#    
#    np.savez( dirpath+'positions.npz',
#              position_list=np.array(position_list, dtype='uint16'),
#              overview_with_frames_raw=added.astype('float32'),
#              overview_with_frames_rgb=added2, 
#              correlation_list=correlation_array,
#              in_row_distance_list = in_column_distance_array,
#              in_column_distance_list = in_column_distance_array )
    



    #cv2.imwrite(dirpath+'positions.tif', added2)
#    p = Pool(initializer=init, initargs=(l,))
#    
#    res = [p.apply_async(find_position, (name, added,over, shape_over, size_frames, size_overview, color)) for name in matched_frames[0:10]]
#    res_list = [r.get() for r in res]
#    
#    p.close()
#    p.join()
#    p.terminate()
#    
#    added = np.array(added).reshape(shape_over)