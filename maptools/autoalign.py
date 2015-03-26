# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:19:43 2015

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
    
    
#def find_offset_and_rotation():
#    """
#    This function finds the current rotation of the scan with respect to the stage coordinate system and the offset that has to be set between two neighboured images when no overlap should occur.
#    It takes no input arguments, so the current frame parameters are used for image acquisition.
#    
#    It returns a tuple of the form (rotation(degrees), offset(fraction of images)).
#    
#    """
#    
#    try:
#        FrameParams = ss.SS_Functions_SS_GetFrameParams()
#    except:
#        logging.error('Could not get Frame Parameters. Make sure SuperScan funtctions are available.')
#    
#    imsize = FrameParams[5]
#    
#    leftX = vt.as2_get_control()
#    vt.as2_set_control('StageOutX', leftX-3.0*imsize)
#    vt.as2_set_control('StageOutY', topY)
#    time.sleep(3)
#    #Goto point for first image and aquire it
#    vt.as2_set_control('StageOutX', leftX)
#    vt.as2_set_control('StageOutY', topY)
#    if use_z_drive:
#        vt.as2_set_control('StageOutZ', interpolation((leftX,  topY), coords)[0])
#    vt.as2_set_control('EHTFocus', interpolation((leftX,  topY), coords)[1])
#    time.sleep(3)
#    frame_nr = ss.SS_Functions_SS_StartFrame(0)
#    ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
#    im1 = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
#    #Go to the right by one half image size
#    vt.as2_set_control('StageOutX', leftX+imsize/2.0)
#    if use_z_drive:
#        vt.as2_set_control('StageOutZ', interpolation((leftX+imsize/2.0,  topY), coords)[0])
#    vt.as2_set_control('EHTFocus', interpolation((leftX+imsize/2.0,  topY), coords)[1])
#    time.sleep(1)
#    frame_nr = ss.SS_Functions_SS_StartFrame(0)
#    ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
#    im2 = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
#    #find offset between the two images        
#    frame_rotation, frame_distance = autoalign.align(im1, im2)
#    #check if the correlation worked correctly and raise an error if not
#    if frame_rotation or frame_distance is None:
#        logging.error('Could not find offset and/or rotation automatically. Please disable these two options and set values manually.')
#        raise RuntimeError('Could not find offset and/or rotation automatically. Please disable these two options and set values manually.')
#    
#    logging.info('Found rotation between x-axis of stage and scan to be: '+str(frame_rotation*180/np.pi))
#    logging.info('Found that the stage moves %.2f times the image size when setting the moving distance to the image size.' % (frame_distance*2.0/impix))


def correlation(im1, im2):
    """"Calculates the cross-correlation of two images im1 and im2. Images have to be numpy arrays."""
    return np.sum( (im1-np.mean(im1)) * (im2-np.mean(im2)) / ( np.std(im1) * np.std(im2) ) ) / np.prod(np.shape(im1))

def translated_correlation(translation, im1, im2):
    """Returns the correct correlation between two images. Im2 is moved with respect to im1 by the vector translation"""
    shape = np.shape(im1)
    translation = np.array(np.round(translation), dtype='int')
    if (translation >= shape).any():
        return 1
    if (translation >= 0).all():
        return -correlation(im1[translation[0]:, translation[1]:], im2[:shape[0]-translation[0], :shape[1]-translation[1]])
    elif (translation < 0).all():
        translation *= -1
        return -correlation(im1[:shape[0]-translation[0], :shape[1]-translation[1]], im2[translation[0]:, translation[1]: ])
    elif translation[0] >= 0 and translation[1] < 0:
        translation[1] *= -1
        return -correlation(im1[translation[0]:, :shape[1]-translation[1]], im2[:shape[0]-translation[0], translation[1]:])
    elif translation[0] < 0 and translation[1] >= 0:
        translation[0] *= -1
        return -correlation(im1[:shape[0]-translation[0], translation[1]:], im2[translation[0]:, :shape[1]-translation[1]])
    else:
        raise ValueError('The translation you entered is not a proper translation vector. It has to be an array-like datatype containing the [y,x] components in C-like order.')

def find_shift(im1, im2, ratio=0.1):
    """Finds the shift between two images im1 and im2."""
    shape = np.shape(im1)
    im1 = cv2.GaussianBlur(im1, (5,5), 3)
    im2 = cv2.GaussianBlur(im2, (5,5), 3)
    start_values = []
    for j in (-1,0,1):
        for i in (-1,0,1):
            start_values.append( np.array((j*shape[0]*ratio, i*shape[1]*ratio))+1 )
    #start_values = np.array( ( (1,1), (shape[0]*ratio, shape[1]*ratio),  (-shape[0]*ratio, -shape[1]*ratio), (shape[0]*ratio, -shape[1]*ratio), (-shape[0]*ratio, shape[1]*ratio) ) )
    function_values = np.zeros(len(start_values))
    for i in range(len(start_values)):
        function_values[i] = translated_correlation(start_values[i], im1, im2)
    start_value = start_values[np.argmin(function_values)]
    res = scipy.optimize.minimize(translated_correlation, start_value, method='Nelder-Mead', args=(im1,im2))
    return (res.x, -res.fun)

def align(im1, im2, ratio=None):
    if ratio is not None:
        res = find_shift(im1, im2, ratio=ratio)
    else:
        res = find_shift(im1, im2, ratio=0.0)
        counter = 1
        while res[1] < 0.85 and counter < 10:
            res = find_shift(im1, im2, ratio=counter*0.1)
            counter += 1
    
    if res[1] < 0.85:
        return (None, None)
        
    rotation = np.arctan2(-res[0][0], res[0][1])*180.0/np.pi
    distance = np.sqrt(np.dot(res[0],res[0]))
    
    return (rotation, distance)