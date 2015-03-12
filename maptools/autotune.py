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
def image_grabber(defocus, im=None, start_def=None):
    try:
        current_focus = vt.as2_get_control('EHTFocus')
        defocus *= 1.0e-9
        vt.as2_set_control('EHTFocus', current_focus+defocus)
        time.sleep(0.1)
        frame_nr = ss.SS_Functions_SS_StartFrame(0)
        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
        im = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
        vt.as2_set_control('EHTFocus', current_focus)
        return cv2.GaussianBlur(im, (5,5), 3)
    except:
        defocus = int(round((abs(defocus-start_def))))
        shape = np.shape(im)
        if defocus == 0:
            im = np.random.poisson(lam=im.flatten(), size=np.size(im))
            return im.reshape(shape)
        defocus = defocus * 2 + 1    
        im = cv2.GaussianBlur(im, (defocus,defocus), defocus)
        im = np.random.poisson(lam=im.flatten(), size=np.size(im))
        return im.reshape(shape)


#integrates over the 1100- and 1120-type reflections in the fft of the given image
def check_focus(defocus, imsize, im=None):
    im = image_grabber(defocus, im=im)
    #ft = np.abs(np.fft.fftshift(np.fft.fft2(im))))
    maxima = 0
    peaks = find_peaks(im, imsize)

    if peaks is not None:
        for coord in peaks:
            maxima += coord[2]
        return -maxima
    else:
        return 0

    
def optimize_focus(imsize, im=None, start_stepsize=4, end_stepsize=1):
    stepsize = start_stepsize
    defocus = 0
    current = check_focus(defocus, imsize,  im=im)
    
    while stepsize >= end_stepsize:
        #previous = current
        #last_defocus = defocus
        #initial = check_focus(defocus, im, shape)
        plus = check_focus(defocus + stepsize, imsize, im=im)
        minus = check_focus(defocus - stepsize, imsize, im=im)
        if plus < current and minus < current:
            logging.warn('Found ambigious focusing!')
        #Find direction of optimization
        if minus < plus and minus < current:
            defocus -= stepsize
            current = minus
        elif  plus < minus and plus < current:
            defocus += stepsize
            current = plus

        stepsize /= 2.0
    
    return defocus


def find_peaks(im, imsize, half_line_thickness = 2, position_tolerance = 1):
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
    
    fft *= gaussian2D(np.mgrid[0:shape[0], 0:shape[1]], shape[1]/2, shape[0]/2, first_order/2, first_order/2, -1, 1).reshape(shape)
    
    #remove vertical and horizontal lines
    central_area = fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1].copy()
    
    horizontal = fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, :]
    horizontal_popt, horizontal_pcov = scipy.optimize.curve_fit(gaussian2D, np.mgrid[0:2*half_line_thickness+1, 0:shape[0]],horizontal.ravel(), p0=(shape[1]/2, half_line_thickness, 1, 1, np.amax(horizontal),0))
    
    vertical = fft[:,shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1]
    vertical_popt, vertical_pcov = scipy.optimize.curve_fit(gaussian2D, np.mgrid[0:shape[1], 0:2*half_line_thickness+1],vertical.ravel(), p0=(half_line_thickness, shape[0]/2, 1, 1, np.amax(vertical),0))
    
    fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, :] /= gaussian2D(np.mgrid[0:2*half_line_thickness+1, 0:shape[1]], horizontal_popt[0], horizontal_popt[1], horizontal_popt[2], horizontal_popt[3], horizontal_popt[4], horizontal_popt[5]).reshape((2*half_line_thickness+1,shape[1])) #horizontal
    fft[shape[0]/2-half_line_thickness:shape[0]/2+half_line_thickness+1, shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1] = central_area
    fft[:, shape[1]/2-half_line_thickness:shape[1]/2+half_line_thickness+1] /= gaussian2D(np.mgrid[0:shape[0], 0:2*half_line_thickness+1], vertical_popt[0], vertical_popt[1], vertical_popt[2], vertical_popt[3], vertical_popt[4], vertical_popt[5]).reshape((shape[0], 2*half_line_thickness+1)) #vertical
    
    
    fft *= gaussian2D(np.mgrid[0:shape[0], 0:shape[1]], shape[1]/2, shape[0]/2, first_order, first_order, -1, 1).reshape(shape)
    
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
        if first_peak[0] in range(center[0]-half_line_thickness,center[0]+half_line_thickness+1) or first_peak[1] in range(center[1]-half_line_thickness,center[1]+half_line_thickness+1):
            fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 0
        elif first_peak[2] < mean_fft + 6.0*std_dev_fft:
            fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 0
        else:
            try:            
                peaks.append(first_peak)
                
                for i in range(1,6):
                    rotation_matrix = np.array( ( (np.cos(i*np.pi/3), -np.sin(i*np.pi/3)), (np.sin(i*np.pi/3), np.cos(i*np.pi/3)) ) )
                    next_peak = np.rint(np.dot( rotation_matrix , (np.array(peaks[0][0:2])-center) ) + center).astype(int)
                    area_next_peak = fft[next_peak[0]-position_tolerance:next_peak[0]+position_tolerance+1, next_peak[1]-position_tolerance:next_peak[1]+position_tolerance+1]
                    max_next_peak = np.amax(area_next_peak)
                    if  max_next_peak > peaks[0][2]/2:
                        next_peak += np.array( np.unravel_index( np.argmax(area_next_peak), np.shape(area_next_peak) ) ) - 1
                        peaks.append(tuple(next_peak)+(max_next_peak,))
                
                if len(peaks) > 1:
                    success = True
    #                for coord in peaks:
    #                    fft[coord[0]-position_tolerance:coord[0]+position_tolerance+1, coord[1]-position_tolerance:coord[1]+position_tolerance+1] = 0
                    return peaks
                else:
                    fft[peaks[0][0]-position_tolerance:peaks[0][0]+position_tolerance+1, peaks[0][1]-position_tolerance:peaks[0][1]+position_tolerance+1] = 0
            except:
                fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 0