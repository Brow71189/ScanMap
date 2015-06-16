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
    from ViennaTools.ViennaTools import tifffile
except:
    try:
        import ViennaTools as vt
        from ViennaTools import tifffile
    except:
        logging.warn('Could not import Vienna tools!')

#from nion.swift import Application
#from nion.swift.model import Image
#from nion.swift.model import Operation
#from nion.swift.model import Region
#from nion.swift.model import HardwareSource
#
#try:
#    import nionccd1010
#except:
#    pass
#    #warnings.warn('Could not import nionccd1010. If You\'re not on an offline version of Swift the ronchigram camera might not work!')
#    #logging.warn('Could not import nionccd1010. If You\'re not on an offline version of Swift the ronchigram camera might not work!')
#    
#try:    
#    from superscan import SuperScanPy as ss    
#except:
#    pass
#    #logging.warn('Could not import SuperScanPy. Maybe you are running in offline mode.')

import autotune    
    
def find_offset_and_rotation():
    """
    This function finds the current rotation of the scan with respect to the stage coordinate system and the offset that has to be set between two neighboured images when no overlap should occur.
    It takes no input arguments, so the current frame parameters are used for image acquisition.
    
    It returns a tuple of the form (rotation(degrees), offset(fraction of images)).
    
    """
    
    try:
        FrameParams = ss.SS_Functions_SS_GetFrameParams()
    except:
        logging.error('Could not get Frame Parameters. Make sure SuperScan funtctions are available.')
    
    imsize = FrameParams[5]
    
    leftX = vt.as2_get_control()
    vt.as2_set_control('StageOutX', leftX-2.0*imsize)
    time.sleep(3)
    #Goto point for first image and aquire it
    vt.as2_set_control('StageOutX', leftX)
    time.sleep(3)
    im1 = autotune.image_grabber()
    #Go to the right by one half image size
    vt.as2_set_control('StageOutX', leftX+imsize/2.0)
    time.sleep(3)
    im2 = autotune.image_grabber()
    #go back to inital position
    vt.as2_set_control('StageOutX', leftX)
    #find offset between the two images
    try:
        frame_rotation, frame_distance = shift_fft(im1, im2)
    except:
        raise
    
    return (frame_rotation, frame_distance)
    
def shift_fft(im1, im2, return_cps=False):
    shape = np.shape(im1)
    if shape != np.shape(im2):
        raise ValueError('Input images must have the same shape')
    fft1 = np.fft.fft2(im1)
    fft2 = np.fft.fft2(im2)
    translation = np.real(np.fft.ifft2((fft1*np.conjugate(fft2))/np.abs(fft1*fft2)))
    if return_cps:
        return translation
    #translation = cv2.GaussianBlur(translation, (0,0), 3)
    if np.amax(translation) <= 0.03: #3.0*np.std(translation)+np.abs(np.amin(translation)):
        #return np.zeros(2)
        raise RuntimeError('Could not determine any match between the input images.')
    transy, transx = np.unravel_index(np.argmax(translation), shape)
    if transy > shape[0]/2:
        transy -= shape[0]
    if transx > shape[1]/2:
        transx -= shape[1]
    
    return np.array((transy,transx))

def rot_dist_fft(im1, im2):
    try:
        shift_vector = shift_fft(im1, im2)
    except RuntimeError:
        raise
    rotation = np.arctan2(-shift_vector[0], shift_vector[1])*180.0/np.pi
    distance = np.sqrt(np.dot(shift_vector,shift_vector))
    
    return (rotation, distance)
    

def align_fft(im1, im2):
    """
    Aligns im2 with respect to im1 using the result of shift_fft
    Return value is im2 which is cropped at one edge and paddded with zeros at the other
    """
    shift = shift_fft(im1, im2)
    shape = np.shape(im2)
    result = np.zeros(shape)
    if (shift >= 0).all():
        result[shift[0]:, shift[1]:] = im2[0:shape[0]-shift[0], 0:shape[1]-shift[1]]
    if (shift < 0).all():
        result[0:shape[0]+shift[0], 0:shape[1]+shift[1]] = im2[-shift[0]:, -shift[1]:]
    elif shift[0] < 0 and shift[1] >= 0:
        result[0:shape[0]+shift[0], shift[1]:] = im2[-shift[0]:, 0:shape[1]-shift[1]]
    elif shift[0] >= 0 and shift[1] < 0:
        result[shift[0]:, 0:shape[1]+shift[1]] = im2[0:shape[0]-shift[0], -shift[1]:]
    return result
    
def align_series_fft(dirname):
    """
    Aligns all images in dirname to the first image there and saves the results in a subfolder.
    """
    dirlist = os.listdir(dirname)
    dirlist.sort()
    im1 = cv2.imread(dirname+dirlist[0], -1)
    savepath = dirname+'aligned/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    tifffile.imsave(savepath+dirlist[0], np.asarray(im1, dtype=im1.dtype))
    
    for i in range(1, len(dirlist)):
        if os.path.isfile(dirname+dirlist[i]):
            im2 = cv2.imread(dirname+dirlist[i], -1)
            tifffile.imsave(savepath+dirlist[i], np.asarray(align_fft(im1, im2), dtype=im1.dtype))
    

def correlation(im1, im2):
    """"Calculates the cross-correlation of two images im1 and im2. Images have to be numpy arrays."""
    return np.sum( (im1-np.mean(im1)) * (im2-np.mean(im2)) / ( np.std(im1) * np.std(im2) ) ) / np.prod(np.shape(im1))

def translated_correlation(translation, im1, im2):
    """Returns the correct correlation between two images. Im2 is moved with respect to im1 by the vector 'translation'"""
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

def rot_dist(im1, im2, ratio=None):
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