# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:22:52 2015

@author: mittelberger
"""
import cv2
import numpy as np
import os
import logging
from multiprocessing import Pool
#import matplotlib.pyplot as plt
import time
import tifffile
import scipy.optimize

#try:
#   from maptools import autotune as at
#except:
from .maptools import autotune as at

def ellipse(polar_angle, a, b, rotation):
    """
    Returns the radius of a point lying on an ellipse with the given parameters.
    """
    return a*b/np.sqrt((b*np.cos(polar_angle-rotation))**2+(a*np.sin(polar_angle-rotation))**2)

def fit_ellipse(angles, radii):
    if len(angles) != len(radii):
        raise ValueError('The input sequences have to have the same lenght!.')
    if len(angles) < 3:
        logging.warn('Can only fit a circle and not an ellipse to a set of less than 3 points.')
        return (np.mean(radii), np.mean(radii), 0.)
    try:
        popt, pcov = scipy.optimize.curve_fit(ellipse, angles, radii, p0=(np.mean(radii), np.mean(radii), 0.0))
    except:
        logging.warn('Fit of the ellipse faied. Using a circle as best approximation of the data.')
        return (np.mean(radii), np.mean(radii), 0.)
    else:
        popt[2] %= np.pi
        return tuple(popt)

def rotation_radius(Peak, find_distortions=True):
    """
    Finds the rotation of the graphene lattice in a frame with repect to the x-axis
    
    Parameters
    -----------
    image : array-like
        Image data as array
    
    imsize : float
        Size of the input image in nm
    
    Returns
    --------
    angle : float
        Angle (in rad) between x-axis and the first reflection in counter-clockwise direction
    """
    try:
        peaks_first, peaks_second = Peak.find_peaks(half_line_thickness=2, position_tolerance = 10, integration_radius = 1,
                                                    second_order=True)
    except:
        raise
    else:
        #Calculate angles to x-axis and radius of reflections
        angles = []
        radii = []
        #center = np.array(np.shape(image))/2
        
        for peak in peaks_first:
            if not (peak==0).all():
                peak = peak[0:2] - np.array(Peak.center)
                angles.append(at.positive_angle(np.arctan2(-peak[0], peak[1])))
                radii.append(np.sqrt(np.sum(peak**2)))
            
        sum_rotation = 0
        for angle in angles:
#            while angle > np.pi/3.0:
#                angle -= np.pi/3.0
            sum_rotation += angle%(np.pi/3)
        
        if find_distortions:
            return (sum_rotation/float(len(angles)), np.mean(radii), np.count_nonzero(peaks_first[:,-1])+np.count_nonzero(peaks_second[:,-1]), \
                    np.sum(peaks_first[:,-1])+np.sum(peaks_second[:,-1])) + fit_ellipse(angles, radii)
        else:
            return (sum_rotation/float(len(angles)), np.mean(radii), np.count_nonzero(peaks_first[:,-1])+np.count_nonzero(peaks_second[:,-1]), \
                    np.sum(peaks_first[:,-1])+np.sum(peaks_second[:,-1]))
    
def calculate_counts(image, threshold=1e-9):
    """
    Returns the divisor to translate float values in "image" to actual counts.
    """
    #set all values <0 to 0
    image[image<0] = 0.0
    #flatten and sort image by pixel values
    sort_im = np.sort(np.ravel(image))
    #find "steps" in intensity
    
    differences = sort_im[1:] - sort_im[0:-1]
    steps = differences[differences>threshold]
    #int_steps = []

    min_step = np.amin(steps)
    
    int_steps = steps[steps<1.5*min_step]
#    for i in range(len(steps)):
#        if len(int_steps) > 2:
#            mean_step = np.mean(int_steps)
#        else:
#            mean_step = 0.0
#        if mean_step == 0.0 or (steps[i] < mean_step*1.5 and steps[i] > 0.5*mean_step):
#            int_steps.append(steps[i])
            
#    int_steps = []
#    for i in range(1, len(sort_im)):
#        difference = sort_im[i] - sort_im[i-1]
#        #only append values if they are "one step" (i.e. one count more)
#        if difference > 1e-9:
#            if len(int_steps) > 2:
#                mean_step = np.mean(int_steps)
#            else:
#                mean_step = 0.0
#            if mean_step == 0.0 or (difference < mean_step*1.5 and difference > 0.5*mean_step):
#                int_steps.append(difference)
    
    return (np.mean(int_steps), np.std(int_steps))

def counts(path):
    im = cv2.imread(path, -1)
    return calculate_counts(im)

def subframes_preprocessing(filename, dirname, imsize, counts_threshold=1e-9, dirt_threshold=0.02, median_blur_diameter=39,
                            gaussian_blur_radius=3, maximum_dirt_coverage=0.5, dirt_border=100, save_fft=True):
    """
    Returns tuple of the form:
            (filename, success, dirt coverage, counts divisor, angle of lattice rotation, mean peak radius)
        For files with more than 50% dirt coverage, the last 3 values will be 'None' and success will be False.
        
    """
    success = True
    #load image
    image = cv2.imread(dirname+filename, -1)
    if image is None:
        raise ValueError(dirname+filename+' is not an image file. Make sure you give the total path as input argument.')
    image_org = image.copy()
    
    Peak = at.Peaking(image=image, imsize=imsize)
    #get mask to filter dirt and check if image is covered by more than "maximum_dirt_coverage"
    mask = Peak.dirt_detector(dirt_threshold=dirt_threshold, median_blur_diam=median_blur_diameter, gaussian_blur_radius=gaussian_blur_radius)
    dirt_coverage = float(np.sum(mask))/(np.shape(image)[0]*np.shape(image)[1])
    if dirt_coverage > maximum_dirt_coverage:
        success = False
        return (filename, dirt_coverage, None, None, None, None, None, None,  success)
    
    #get angle of rotation and peak radius
    try:
        rotation, radius, number_peaks, peak_intensities_sum, ellipse_a, ellipse_b, angle = rotation_radius(Peak)
    except RuntimeError as detail:
        print('Error in '+ filename + ': ' + str(detail))
        rotation = ellipse_a = ellipse_b = angle = np.NaN
        number_peaks = peak_intensities_sum = 0
        #peaks = None
        success = False
    
    #Get counts divisor for image
    counts_divisor = calculate_counts(image, threshold=counts_threshold)[0]
    #Calculate actual counts in image and "translate" it to 16bit unsigned integer.
    image[image<0]=0.0
    image = np.asarray(np.rint(image/counts_divisor), dtype='uint16')
    #dilate mask if dirt_border > 0
    if dirt_border > 0:
        mask = cv2.dilate(mask, np.ones((dirt_border, dirt_border)))
    #Set pixels where dirt was detected to maximum of 16bit range
    image[mask==1] = 65535
    #save the image to disk
    if not os.path.exists(dirname+'prep_'+dirname.split('/')[-2]+'/'):
        os.makedirs(dirname+'prep_'+dirname.split('/')[-2]+'/')
    if save_fft:
        if not os.path.exists(dirname+'fft_'+dirname.split('/')[-2]+'/'):
            os.makedirs(dirname+'fft_'+dirname.split('/')[-2]+'/')
    
    if success:
        #cv2.imwrite(dirname+'subframes_preprocessing/'+filename, image)
        
        tifffile.imsave(dirname+'prep_'+dirname.split('/')[-2]+'/'+filename, image)
        if save_fft:
            fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image_org)))).astype('float32')
            center = np.array(np.shape(image))/2
            ell = np.ones(np.shape(fft), dtype='float32')
            cv2.ellipse(ell, (tuple(center), (ellipse_a*2, ellipse_b*2), -angle*180/np.pi), 4)
            cv2.ellipse(ell, (tuple(center), (ellipse_a*2*np.sqrt(3), ellipse_b*2*np.sqrt(3)), -angle*180/np.pi), 4)            
            fft *= ell
            savesize = int(2.0*imsize/0.213)
            tifffile.imsave(dirname+'fft_'+dirname.split('/')[-2]+'/'+filename, fft[center[0]-savesize:center[0]+savesize+1, center[1]-savesize:center[1]+savesize+1])
        
    
    #return image parameters
    return (filename, dirt_coverage, number_peaks, peak_intensities_sum, rotation, ellipse_a, ellipse_b, angle, success)

if __name__ == '__main__':
    
    overall_starttime = time.time()

    dirpath = '/home/mittelberger/Documents/jk-randomwalk/divac-seq3-lowdose'
    #dirpath = '/3tb/Dark_noise/'
    imsize = 4
    dirt_threshold = 1
    dirt_border = 0

    if not dirpath.endswith('/'):
        dirpath += '/'
    dirlist=os.listdir(dirpath)
    matched_dirlist = []
    for filename in dirlist:
        try:
            int(filename[0:4])
            matched_dirlist.append(filename)
        except:
            pass
    matched_dirlist.sort()
    #starttime = time.time()
    
    pool = Pool()
    res = [pool.apply_async(subframes_preprocessing, (filename, dirpath, imsize), {'dirt_threshold': dirt_threshold, 'dirt_border':dirt_border, \
            'median_blur_diameter': 59, 'gaussian_blur_radius': 5,'save_fft': True, 'maximum_dirt_coverage': 0.5}) for filename in matched_dirlist]
    res_list = [p.get() for p in res]
    pool.close()
    pool.terminate()

    #duration = time.time()-starttime

    #print('Time for calculation: %.2f s' %(duration,))
    
    res_list.sort()
    
    if not os.path.exists(dirpath+'prep_'+dirpath.split('/')[-2]+'/'):
        os.makedirs(dirpath+'prep_'+dirpath.split('/')[-2]+'/')
    
    frame_data_file = open(dirpath+'prep_'+dirpath.split('/')[-2]+'/'+'frame_init_' + dirpath.split('/')[-2] + '.txt', 'w')
    
    frame_data_file.write('#This file contains informations about all frames of '+(dirpath.split('/')[-2]+'\n'))
    frame_data_file.write('#Created: ' + time.strftime('%Y/%m/%d %H:%M') + '\n')
    frame_data_file.write('#Imagesize in nm: %.1f\tDirt threshold: %f\tDirt border: %d\n' %(imsize, dirt_threshold, dirt_border))
    frame_data_file.write('#Meanings of the values are:\n')
    frame_data_file.write('#filename\tdirt\tnumpeak\ttuning\ttilt\tella\tellb\tellphi\n\n')
    
    for frame_data in res_list:
        if frame_data[-1]:
                frame_data_file.write('%s\t%.3f\t%d\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\n' % frame_data[0:8])
    
    frame_data_file.close()
    
    overall_time = time.time() - overall_starttime
    
    print('Done analysing %d files in %.2f s.' %(len(matched_dirlist), overall_time))
        
#    res_list = []
#    for name in matched_dirlist:
#        res_list.append(counts(name))
    
#    means = []
#    stddevs = []
#    for res in res_list:
#        means.append(res[0])
#        stddevs.append(res[1])
#        
#    fig1 = plt.figure()
#    ax1 = fig1.add_subplot(111)
#    ax1.plot(means)
#    
#    fig2 = plt.figure()
#    ax2 = fig2.add_subplot(111)
#    ax2.plot(stddevs)
#    
#    fig1.show()
#    fig2.show()