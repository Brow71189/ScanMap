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

