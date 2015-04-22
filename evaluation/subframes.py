# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:22:52 2015

@author: mittelberger
"""
import cv2
import ScanMap.maptools.autotune as at
import numpy as np
import os
import logging
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import tifffile
import scipy.optimize

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

def rotation_radius(image, imsize, find_distortions=True):
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
        peaks = at.find_peaks(image, imsize, half_line_thickness=4, position_tolerance = 9)
    except:
        raise
    else:
        #Calculate angles to x-axis and radius of reflections
        angles = []
        radii = []
        center = np.array(np.shape(image))/2
        
        for peak in peaks:
            peak = np.array(peak[0:2])-center
            angles.append(at.positive_angle(np.arctan2(-peak[0], peak[1])))
            radii.append(np.sqrt(np.sum(peak**2)))
            
        sum_rotation = 0
        for angle in angles:
#            while angle > np.pi/3.0:
#                angle -= np.pi/3.0
            sum_rotation += angle%(np.pi/3)
        
        if find_distortions:
            return (sum_rotation/float(len(angles)), np.mean(radii)) + fit_ellipse(angles, radii)
        else:
            return (sum_rotation/float(len(angles)), np.mean(radii))
    
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
    int_steps = []
    for i in range(len(steps)):
        if len(int_steps) > 2:
            mean_step = np.mean(int_steps)
        else:
            mean_step = 0.0
        if mean_step == 0.0 or (steps[i] < mean_step*1.5 and steps[i] > 0.5*mean_step):
            int_steps.append(steps[i])
            
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
    
    
def dirt_detector(image, threshold=0.02, median_blur_diam=85, gaussian_blur_radius=3):
    """
    Returns a mask with the same shape as "image" that is 1 where there is dirt and 0 otherwise
    """
    #apply Gaussian Blur to improve dirt detection
    if gaussian_blur_radius > 0:
        image = cv2.GaussianBlur(image, (0,0), gaussian_blur_radius)
    #create mask
    mask = np.zeros(np.shape(image), dtype='uint8')
    mask[image>threshold] = 1
    #apply median blur to mask to remove noise influence
    return cv2.medianBlur(mask, median_blur_diam)

def counts(path):
    im = cv2.imread(path, -1)
    return calculate_counts(im)

def subframes_preprocessing(filename, dirname, imsize, counts_threshold=1e-9, dirt_threshold=0.02, median_blur_diameter=39, gaussian_blur_radius=3,\
                             maximum_dirt_coverage=0.5, save_fft=True):
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
    #get mask to filter dirt and check if image is covered by more than "maximum_dirt_coverage"
    mask = dirt_detector(image, threshold=dirt_threshold, median_blur_diam=median_blur_diameter, gaussian_blur_radius=gaussian_blur_radius)
    dirt_coverage = float(np.sum(mask))/(np.shape(image)[0]*np.shape(image)[1])
    if dirt_coverage > maximum_dirt_coverage:
        success = False
        return (filename, dirt_coverage, None, None, None, None, success)
    
    #get angle of rotation and peak radius
    try:
        rotation, radius, ellipse_a, ellipse_b, angle = rotation_radius(image, imsize)
    except:
        rotation = ellipse_a = ellipse_b = angle = None
        success = False
    
    #Get counts divisor for image
    counts_divisor = calculate_counts(image, threshold=counts_threshold)[0]
    #Calculate actual counts in image and "translate" it to 16bit unsigned integer.
    image[image<0]=0.0
    image = np.asarray(np.rint(image/counts_divisor), dtype='uint16')
    #Set pixels where dirt was detected to maximum of 16bit range
    image[mask==1] = 65535
    #save the image to disk
    if not os.path.exists(dirname+'subframes_preprocessing/'):
        os.makedirs(dirname+'subframes_preprocessing/')
    if save_fft:
        if not os.path.exists(dirname+'fft_stack/'):
            os.makedirs(dirname+'fft_stack/')
    
    if success:
        #cv2.imwrite(dirname+'subframes_preprocessing/'+filename, image)
        tifffile.imsave(dirname+'subframes_preprocessing/'+filename, image)
        if save_fft:
            fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image_org)))).astype('float32')
            center = np.array(np.shape(image))/2
            cv2.ellipse(fft, (tuple(center), (ellipse_a*2, ellipse_b*2), -angle*180/np.pi), -1)
            savesize = int(2.0*imsize/0.213)
            tifffile.imsave(dirname+'fft_stack/'+filename, fft[center[0]-savesize:center[0]+savesize+1, center[1]-savesize:center[1]+savesize+1])
        
    
    #return image parameters
    return (filename, dirt_coverage, rotation, ellipse_a, ellipse_b, angle, success)

if __name__ == '__main__':
    
    overall_starttime = time.time()

    dirpath = '/3tb/maps_data/map_2015_04_15_13_13/'
    #dirpath = '/3tb/test_ellipse/'
    imsize = 12
    dirt_threshold = 0.0085
    
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
    res = [pool.apply_async(subframes_preprocessing, (filename, dirpath, imsize), {'dirt_threshold': dirt_threshold}) for filename in matched_dirlist]
    res_list = [p.get() for p in res]
    pool.close()
    pool.terminate()

    #duration = time.time()-starttime

    #print('Time for calculation: %.2f s' %(duration,))
    
    res_list.sort()
    
    if not os.path.exists(dirpath+'subframes_preprocessing/'):
        os.makedirs(dirpath+'subframes_preprocessing/')
    
    frame_data_file = open(dirpath+'subframes_preprocessing/'+'frame_init.txt', 'w')
    
    frame_data_file.write('#This file contains informations about all frames of '+(dirpath.split('/')[-2] if dirpath.endswith('/') else dirpath.split('/')[-1])+'\n')
    frame_data_file.write('#Imagesize in nm: %.1f\tDirt threshold: %f\n' %(imsize, dirt_threshold))
    frame_data_file.write('#Meanings of the values are:\n')
    frame_data_file.write('#filename\tdirt coverage\ttilt\tellipse half-axis a\tellipse half-axis b\tangle of half-axis a (rad)\n\n')
    
    for frame_data in res_list:
        if frame_data[6]:
                frame_data_file.write('%s\t%.3f\t%.6f\t%.6f\t%.6f\t%.6f\n' % frame_data[0:6])
    
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