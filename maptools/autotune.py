# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:54:54 2015

@author: mittelberger
"""

import logging
import time
import os

import numpy as np
import scipy.optimize
import cv2
import matplotlib as plt

try:
    import ViennaTools.ViennaTools as vt
except:
    try:
        import ViennaTools as vt
    except:
        logging.warn('Could not import Vienna tools!')

from nion.swift import Application
from nion.swift.model import Image
from nion.swift.model import Operation
from nion.swift.model import Region
from nion.swift.model import HardwareSource

try:
    import nionccd1010
except:
    #warnings.warn('Could not import nionccd1010. If You\'re not on an offline version of Swift the ronchigram camera might not work!')
    logging.warn('Could not import nionccd1010. If You\'re not on an offline version of Swift the ronchigram camera might not work!')
    
try:    
    from superscan import SuperScanPy as ss    
except:
    logging.warn('Could not import SuperScanPy. Maybe you are running in offline mode.')
    
    
def autofocus(imsize=None, image=None, start_stepsize=4, end_stepsize=1, position_tolerance=1,start_def=None):
    """
    Tries to find the correct focus in an atomically resolved STEM image of graphene.
    The focus is optimized by maximizing the intensity of the 6 first-order peaks in the FFT
    
    Parameters
    ----------
    imsize : Optional, float
        FOV set in the SuperScan settings
    image : Optional, numpy array
        
    
    """
    flag  = False    
    if imsize == None:
        imsize = ss.SS_Functions_SS_GetFrameParams()[5]
    try:
        FrameParams = ss.SS_Functions_SS_GetFrameParams()
        if FrameParams[0] > 1024:
            ss.SS_Functions_SS_SetFrameParams(1024, 1024,FrameParams[2],FrameParams[3], FrameParams[4], FrameParams[5], FrameParams[6], FrameParams[7], FrameParams[8], FrameParams[9], FrameParams[10])
            flag = True
    except:
        logging.warn('Could not check Frame Parameters.')
        
    estimated_focus = optimize_focus(imsize, im=image, start_stepsize=start_stepsize, end_stepsize=end_stepsize)
    try:
        current_focus = vt.as2_get_control('EHTFocus')
        vt.as2_set_control('EHTFocus', current_focus+estimated_focus*1e-9)
    except:
        pass
    #ss.SS_Functions_SS_StartFrame(1)
    if flag:
        ss.SS_Functions_SS_SetFrameParams(FrameParams[0], FrameParams[1],FrameParams[2],FrameParams[3], FrameParams[4], FrameParams[5], FrameParams[6], FrameParams[7], FrameParams[8], FrameParams[9], FrameParams[10])
    
    return estimated_focus



#generates the defocused, noisy image
def image_grabber(defocus=0, astig=[0,0], im=None, start_def=0.0, start_astig=[0,0], imsize=1.0):
    
    if im is None:
        current_focus = vt.as2_get_control('EHTFocus')
        defocus *= 1.0e-9
        vt.as2_set_control('EHTFocus', current_focus+defocus)
        time.sleep(0.1)
        frame_nr = ss.SS_Functions_SS_StartFrame(0)
        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
        im = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
        vt.as2_set_control('EHTFocus', current_focus)
        return im
    else:
        defocus -= start_def
        shape = np.shape(im)
        astig = np.array(astig)
        start_astig = np.array(start_astig)
        astig -= start_astig
        
        if defocus == 0 and (astig == 0).all():
            im = np.random.poisson(lam=im.flatten(), size=np.size(im))
            return im.reshape(shape)
            
        #fft = np.fft.fftshift(np.fft.fft2(im))
        kernelsize = np.around((np.array(shape)/2.0)-0.5, 1)
        y,x = np.mgrid[-kernelsize[0]:kernelsize[0]+1.0, -kernelsize[1]:kernelsize[1]+1.0]/imsize#/(kernelsize[0])
        #compute aberration function up to twofold astigmatism
        raw_kernel = -np.pi*defocus*4.9e-3*(x**2+y**2) + np.pi*np.sum(np.power(astig,2))*4.9e-3*(x**2+y**2)*np.cos(2*(np.arctan2(y,x)-np.arctan2(astig[1], astig[0])) )
        #raw_kernel = 1+np.sqrt(3)*(2*(x**2+y**2)-1)/defocus + np.sqrt(6)*np.sum(np.power(astig, 2))*(x**2+y**2)*np.cos(2*(np.arctan2(y,x)-np.arctan2(astig[1], astig[0])))
        kernel = np.cos(raw_kernel)+1j*np.sin(raw_kernel)
        #kernel = np.exp(-np.sign(defocus)*raw_kernel)
        kernel *= np.exp(-(x**2+y**2)/(2*kernelsize[0]))
        kernel = np.abs(np.fft.fftshift(np.fft.ifft2(kernel)))**2
        kernel /= np.sum(kernel)
        im = cv2.filter2D(im, -1, kernel)
        #fft = np.real(fft) * kernel/np.sum(np.abs(kernel)) + 1j * np.imag(fft)
        #fft *= kernel/np.sum(np.abs(kernel))
        #im = np.abs(np.real(np.fft.ifft2(np.fft.fftshift(fft))))
        im = np.random.poisson(lam=im.flatten(), size=np.size(im))
        return im.reshape(shape)
        #return kernel


def check_tuning(imsize, defocus=0, astig=[0,0], im=None, check_astig=False):
    im = image_grabber(defocus=defocus, astig=astig, im=im)
    peaks = find_peaks(im, imsize)
    if peaks is not None:    
        coordinates = np.zeros((len(peaks), 2))
        intensities = np.zeros(len(peaks))
        for i in range(len(peaks)):
            coordinates[i,:] = np.array(peaks[i][0:2], dtype='int')
            intensities[i] = peaks[i][2]
    else:
        return 0
        
    if check_astig:
        center = np.array(np.shape(im))/2
        if len(peaks) == 6:
            max_pos = np.argmax(intensities)
            #find point with maximum intensitiy and its neighbours such that left < right < maximum
            if max_pos != 0 and max_pos != 5:
                if intensities[max_pos-1] < intensities[max_pos+1]:
                    left = max_pos-1
                    right  = max_pos+1
                else:
                    left = max_pos+1
                    right = max_pos-1
            elif max_pos == 0:
                if intensities[5] < intensities[1]:
                    left = 5
                    right = 1
                else:
                    left = 1
                    right = 5
            else:
                if intensities[4] < intensities[0]:
                    left = 4
                    right = 0
                else:
                    left = 0
                    right = 4
            #Empirical formula for finding the angle of the twofold astigmatism in an FFT-image
            #From 3 reflections with different intensities the direction of the astigmatism can be calculated
            #positive angles are clockwise
            relative_astig_angle = (intensities[left]-intensities[right])/(1.9*(intensities[left]-intensities[max_pos]))
            absolute_astig_angle = np.arctan2(coordinates[max_pos,0]-center[0],coordinates[max_pos,1]-center[1]) + relative_astig_angle*np.sign(np.arctan2(coordinates[right,0]-center[0],coordinates[right,1]-center[1])-np.arctan2(coordinates[max_pos,0]-center[0],coordinates[max_pos,1]-center[1]))
        elif len(peaks) == 4:
            #if only 4 peaks are visible, assume that the missing peak has the intensity 0
#            #calculate angle between first two reflections (first one is the brightest)
#            max_pos = np.argmax(intensities)
#            if max_pos != 3:
#                right = max_pos + 1
#            else:
#                right = 0
#                
#            angle_first = np.arctan2(coordinates[max_pos,0]-center[0],coordinates[max_pos, 1]-center[1]) - np.arctan2(coordinates[right,0]-center[0], coordinates[right,1]-center[1])
#            
#            #calculate coordinates of missing reflection
#            if np.abs(angle_first) < np.pi/2.0:
#                missing_coords = coordinates[max_pos] - coordinates[right]
#            else:
#                missing_coords = coordinates[max_pos]-center + coordinates[right]-center
#            #calculate angle of astigmatism as in case of 6 visible reflections (setting intensities[left])
            relative_astig_angle = (0.0-intensities[right])/(1.9*(0.0-intensities[max_pos]))
            absolute_astig_angle = np.arctan2(coordinates[max_pos,0]-center[0],coordinates[max_pos,1]-center[1]) + relative_astig_angle*np.sign(np.arctan2(coordinates[right,0]-center[0],coordinates[right,1]-center[1])-np.arctan2(coordinates[max_pos,0]-center[0],coordinates[max_pos,1]-center[1]))
        elif len(peaks) == 2:
            #if only two peaks are visible, assume that astigmatism is in direction of brightest peak
            max_pos = np.argmax(intensities)
            absolute_astig_angle = relative_astig_angle = np.arctan2(coordinates[max_pos, 0]-center[0], coordinates[max_pos, 1]-center[1])
            
        return (intensities, coordinates, absolute_astig_angle, relative_astig_angle)
        
    else:
        return -np.sum(intensities)
        
        
def optimize_focus(imsize, im=None, start_stepsize=4, end_stepsize=1):
    stepsize = start_stepsize
    defocus = 0
    current = check_tuning(imsize, im=im)
    
    while stepsize >= end_stepsize:
        #previous = current
        #last_defocus = defocus
        #initial = check_focus(defocus, im, shape)
        plus = check_tuning(imsize, defocus=(defocus + stepsize), im=im)
        minus = check_tuning(imsize, defocus=(defocus - stepsize), im=im)
        if plus < current and minus < current:
            logging.warn('Found ambigious focusing!')
        
        if minus < plus and minus < current:
            defocus -= stepsize
            current = minus
        elif  plus < minus and plus < current:
            defocus += stepsize
            current = plus

        stepsize /= 2.0
    
    return defocus


def find_peaks(im, imsize, half_line_thickness = 5, position_tolerance = 3):
    """
        This function can find the 6 first-order peaks in the FFT of an atomic-resolution image of graphene.
        Input:
                im: Image as a numpy array or any type that can be simply casted to a numpy array.
                imsize: Size of the input image in nm.
        Output:
                List of tuples that contain the coordinates of the reflections. The tuples have the form (y, x, intensity_of_peak_maximum)
                If no peaks were found the return value will be None.
                Note that the returned intesities might be smaller than that of the raw fft because of the processing done in the function.
    """
    def gaussian2D(xdata, x0, y0, x_var, y_var, amplitude, offset):
        x0, y0, x_var, y_var, amplitude, offset = float(x0), float(y0), float(x_var), float(y_var), float(amplitude), float(offset)
        return (amplitude*np.exp( -( (xdata[1]-x0)**2/(2*x_var) + (xdata[0]-y0)**2/(2*y_var) ) ) + offset).ravel()    
    
    global fft
    fft = np.abs(np.fft.fftshift(np.fft.fft2(im)))
    shape = np.shape(im)
    
    first_order = imsize/0.213
    #second_order = imsize/0.123
    
    fft *= gaussian2D(np.mgrid[0:shape[0], 0:shape[1]], shape[1]/2, shape[0]/2, first_order, first_order, -1, 1).reshape(shape)
    
    #remove vertical and horizontal lines
    central_area = fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1].copy()
    
    horizontal = fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, :]
    horizontal_popt, horizontal_pcov = scipy.optimize.curve_fit(gaussian2D, np.mgrid[0:2*half_line_thickness+1, 0:shape[0]],horizontal.ravel(), p0=(shape[1]/2, half_line_thickness, 1, 1, np.amax(horizontal),0))
    
    vertical = fft[:,shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1]
    vertical_popt, vertical_pcov = scipy.optimize.curve_fit(gaussian2D, np.mgrid[0:shape[1], 0:2*half_line_thickness+1],vertical.ravel(), p0=(half_line_thickness, shape[0]/2, 1, 1, np.amax(vertical),0))
    
    fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, :] /= gaussian2D(np.mgrid[0:2*half_line_thickness+1, 0:shape[1]], horizontal_popt[0], horizontal_popt[1], horizontal_popt[2], horizontal_popt[3], horizontal_popt[4], horizontal_popt[5]).reshape((2*half_line_thickness+1,shape[1])) #horizontal
    fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, :] *= np.mean(fft)    
    fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1] = central_area
    fft[:, shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1] /= gaussian2D(np.mgrid[0:shape[0], 0:2*half_line_thickness+1], vertical_popt[0], vertical_popt[1], vertical_popt[2], vertical_popt[3], vertical_popt[4], vertical_popt[5]).reshape((shape[0], 2*half_line_thickness+1)) #vertical
    fft[:, shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1] *= np.mean(fft)
    
    
    fft *= gaussian2D(np.mgrid[0:shape[0], 0:shape[1]], shape[1]/2, shape[0]/2, first_order*3.0, first_order*3.0, -1, 1).reshape(shape)
    
    #find peaks
    success = False
    std_dev_fft = np.std(fft)
    mean_fft = np.mean(fft)
    center = np.array(shape)/2
    counter = 0
    
    while success is False:
        counter += 1
        if counter > np.sqrt(shape[0]):
            return None
        peaks = []
        first_peak = np.unravel_index(np.argmax(fft), shape)+(np.amax(fft), )
        #check if found peak is on cross
#        if first_peak[0] in range(center[0]-half_line_thickness,center[0]+half_line_thickness+1) or first_peak[1] in range(center[1]-half_line_thickness,center[1]+half_line_thickness+1):
#            fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 0
        if first_peak[2] < mean_fft + 6.0*std_dev_fft:
            fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 0
        elif np.sqrt(np.sum((np.array(first_peak[0:2])-center)**2)) < first_order/1.5:
            fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 0
        else:
            try:            
                peaks.append(first_peak)
                
                for i in range(1,6):
                    rotation_matrix = np.array( ( (np.cos(i*np.pi/3), -np.sin(i*np.pi/3)), (np.sin(i*np.pi/3), np.cos(i*np.pi/3)) ) )
                    next_peak = np.rint(np.dot( rotation_matrix , (np.array(peaks[0][0:2])-center) ) + center).astype(int)
                    area_next_peak = fft[next_peak[0]-position_tolerance:next_peak[0]+position_tolerance+1, next_peak[1]-position_tolerance:next_peak[1]+position_tolerance+1]
                    max_next_peak = np.amax(area_next_peak)
#TODO: Find better criterion for deciding if peak at a position
                    if  max_next_peak > mean_fft + 4.0*std_dev_fft:#peaks[0][2]/4:
                        next_peak += np.array( np.unravel_index( np.argmax(area_next_peak), np.shape(area_next_peak) ) ) - position_tolerance
                        peaks.append(tuple(next_peak)+(max_next_peak,))
                
                if len(peaks) > 1:
                    success = True
#                    for coord in peaks:
#                        fft[coord[0]-position_tolerance:coord[0]+position_tolerance+1, coord[1]-position_tolerance:coord[1]+position_tolerance+1] = -100
#                    plt.pyplot.matshow(fft)
#                    plt.show()
                    return peaks
                else:
                    fft[peaks[0][0]-position_tolerance:peaks[0][0]+position_tolerance+1, peaks[0][1]-position_tolerance:peaks[0][1]+position_tolerance+1] = 0
            except:
                fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 0
