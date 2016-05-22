# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:19:43 2015

@author: mittelberger

"""

import warnings
import os

import numpy as np
from scipy import optimize, ndimage
#try:
#    import cv2
#except:
#    pass

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from . import tifffile

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
    
def shift_fft(im1, im2, return_cps=False):
    shape = np.shape(im1)
    if shape != np.shape(im2):
        raise ValueError('Input images must have the same shape')
    fft1 = np.fft.fft2(im1)
    fft2 = np.fft.fft2(im2)
    translation = np.abs(np.fft.ifft2(((fft1*np.conjugate(fft2))/np.abs(fft1*fft2))))
    if return_cps:
        return np.fft.fftshift(translation)
    #translation = cv2.GaussianBlur(translation, (0,0), 3)
    if np.amax(translation) <= 0.03: #3.0*np.std(translation)+np.abs(np.amin(translation)):
        #return np.zeros(2)
        raise RuntimeError('Could not determine any match between the input images (Maximum correlation: {:.4f}).'
                           .format(np.amax(translation)))
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

def align(im1, im2, method='correlation', ratio=0.1):
    """
    Aligns im2 with respect to im1 using the result of shift_fft
    Return value is im2 which is cropped at one edge and paddded with zeros at the other
    """
    if method == 'correlation':
        shift = find_shift(im1, im2, ratio=ratio)
        print(shift)
        shift = np.rint(shift[0])
    elif method == 'fft':
        try:
            shift = shift_fft(im1, im2)
        except RuntimeError as detail:
            print(detail)
            shift = np.array((0,0))
    else:
        raise TypeError('Unknown method: {:s}.'.format(method))
        
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
    
def align_fft(im1, im2):
    return align(im1, im2, method='fft')

def align_series(dirname, method='correlation', ratio=0.1):
    """
    Aligns all images in dirname to the first image there and saves the results in a subfolder.
    """
    dirlist = os.listdir(dirname)
    dirlist.sort()
    im1 = ndimage.imread(dirname+dirlist[0])
    savepath = dirname+'aligned/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        
    tifffile.imsave(savepath+dirlist[0], np.asarray(im1, dtype=im1.dtype))
    
    for i in range(1, len(dirlist)):
        if os.path.isfile(dirname+dirlist[i]):
            im2 = ndimage.imread(dirname+dirlist[i])
            tifffile.imsave(savepath+dirlist[i],
                            np.asarray(align(im1, im2, method=method, ratio=ratio), dtype=im1.dtype))
            
def align_series_fft(dirname):
    align_series(dirname, method='fft')
    

def correlation(im1, im2):
    """"Calculates the cross-correlation of two images im1 and im2. Images have to be numpy arrays."""
    #return np.sum( (im1-np.mean(im1)) * (im2-np.mean(im2)) / ( np.std(im1) * np.std(im2) ) ) / np.prod(np.shape(im1))
    return np.sum((im1) * (im2)) / np.sqrt(np.sum((im1)**2) * np.sum((im2)**2))

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

def find_shift(im1, im2, ratio=0.1, num_steps=6):
    """Finds the shift between two images im1 and im2."""
    shape = np.shape(im1)
    #im1 = cv2.GaussianBlur(im1, (5,5), 3)
    #im2 = cv2.GaussianBlur(im2, (5,5), 3)
#    if ratio > 0:
#        start_values = []
#        for j in (-1,0,1):
#            for i in (-1,0,1):
#                start_values.append( np.array((j*shape[0]*ratio, i*shape[1]*ratio)) )
#        #start_values = np.array( ( (1,1), (shape[0]*ratio, shape[1]*ratio),  (-shape[0]*ratio, -shape[1]*ratio), (shape[0]*ratio, -shape[1]*ratio), (-shape[0]*ratio, shape[1]*ratio) ) )
#        function_values = np.zeros(len(start_values))
#        for i in range(len(start_values)):
#            function_values[i] = translated_correlation(start_values[i], im1, im2)
#        start_value = start_values[np.argmin(function_values)]
#    else:
#        start_value = (0,0)
#    print(start_value)
#    res = optimize.minimize(translated_correlation, start_value, method='Nelder-Mead', args=(im1,im2))
#    def mod_corr(translation, im1, im2):
#        val = 1+translated_correlation(translation, im1, im2)
#        return np.array((val, val))
#        
#    res = optimize.root(mod_corr, start_value, method='hybr', args=(im1, im2))
    max_distance = (shape[0]*ratio, shape[1]*ratio)
    res = optimize.brute(translated_correlation,
                         ((-max_distance[0], max_distance[0]), (-max_distance[1], max_distance[1])),
                         args=(im1, im2), Ns=num_steps, full_output=True)
    return (res[0], -res[1])
    #return (res.x, -res.fun)

def rot_dist(im1, im2, ratio=None):
    if ratio is not None:
        res = find_shift(im1, im2, ratio=ratio)
    else:
        res = find_shift(im1, im2, ratio=0.0)
        counter = 1
        while res[1] < 0.8 and counter < 10:
            res = find_shift(im1, im2, ratio=counter*0.1)
            counter += 1
    
    if res[1] < 0.8:
        return (None, None)
        
    rotation = np.arctan2(-res[0][0], res[0][1])*180.0/np.pi
    distance = np.sqrt(np.dot(res[0],res[0]))
    
    return (rotation, distance)