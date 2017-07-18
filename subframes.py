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
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from .maptools import tifffile
    except:
        from maptools import tifffile
import scipy.optimize

try:
   from maptools import autotune as at
except:
    from .maptools import autotune as at
import pyximport; pyximport.install()
import c_electron_counting

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
dirpath = '/3tb/maps_data/map_2017_07_13_14_25'
imsize = 40
graphene_threshold = 0.023
light_threshold = -1
heavy_threshold = 0.038
dirt_border = 50
minimum_graphene_area = 0.3
minimum_number_peaks = 2
maximum_number_peaks = 12
only_process_this_number_of_images = -1
only_process_images_of_shape = None #(2048, 2048) # None or tuple
remove_left_edge_number_pixels = -1 # -1 nothing to remove
save_fft = True
#parameters for electron counting
baseline = 0.02
countlevel = 0.06
peaklength = 3
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

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
        peaks_first, peaks_second = Peak.find_peaks(half_line_thickness=2, position_tolerance = 10,
                                                    integration_radius = 1, second_order=True)
    except:
        raise
    else:
        #Calculate angles to x-axis and radius of reflections
        angles = []
        radii = []
        #center = np.array(np.shape(image))/2
        center = np.array(Peak.center)
        
        for peak in peaks_first:
            if not (peak == 0).all():
                peak = peak[0:2]-center
                angles.append(at.positive_angle(np.arctan2(-peak[0], peak[1])))
                radii.append(np.sqrt(np.sum(peak**2)))
            
#        sum_rotation = 0
#        for angle in angles:
##            while angle > np.pi/3.0:
##                angle -= np.pi/3.0
#            sum_rotation += angle%(np.pi/3)
        #angles2 = np.array(angles)%np.pi/3
        angles2 = np.array(angles) * 6
        cos_angles = np.cos(angles2)
        sin_angles = np.sin(angles2)
        mean_angle = at.positive_angle(np.arctan2(np.mean(sin_angles), np.mean(cos_angles))) / 6
        #mean_angle = angles[0]
        
        if find_distortions:
            #sum_rotation/float(len(angles))
            return (mean_angle, np.mean(radii), np.count_nonzero(peaks_first[:,-1]) + 
                    np.count_nonzero(peaks_second[:,-1]), np.sum(peaks_first[:,-1]) + 
                    np.sum(peaks_second[:,-1])) + fit_ellipse(angles, radii)
        else:
            return (mean_angle, np.mean(radii), np.count_nonzero(peaks_first[:,-1]) + 
                    np.count_nonzero(peaks_second[:,-1]), np.sum(peaks_first[:,-1])+np.sum(peaks_second[:,-1]))
    
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
    
def create_mask(Peak, graphene_threshold, light_threshold, heavy_threshold, dirt_border=0):
    pixelsize = imsize/Peak.shape[0]
    if graphene_threshold > 0:
        mask = Peak.dirt_detector(dirt_threshold=graphene_threshold, median_blur_diam=0.6/pixelsize, gaussian_blur_radius=0.03/pixelsize)
        if dirt_border > 0:
            mask = cv2.erode(mask, np.ones((dirt_border, dirt_border)))
    else:
        mask = np.ones(Peak.shape, dtype=np.uint8)
    
    if light_threshold > 0 and light_threshold != heavy_threshold:
        mask[Peak.dirt_detector(dirt_threshold=light_threshold, median_blur_diam=0.6/pixelsize, gaussian_blur_radius=0.03/pixelsize)==1] = 4
        
    if heavy_threshold > 0:
        heavy_dirt_mask = Peak.dirt_detector(dirt_threshold=heavy_threshold, median_blur_diam=0.6/pixelsize, gaussian_blur_radius=0.03/pixelsize)
        if dirt_border > 0:
            heavy_dirt_mask = cv2.dilate(heavy_dirt_mask, np.ones((dirt_border, dirt_border)))
        
        mask[heavy_dirt_mask==1] = 16

        
    return mask

def electron_counting(image, baseline=0.002, countlevel=0.01, peaklength=5):
    res = np.zeros(image.shape, dtype=np.uint16)
    for k in range(image.shape[0]):
        integral = 0
        index = -1
        for i in range(image.shape[1]):
            if index != -1 and (image[k, i] < countlevel/2 or i - index >= peaklength): #integral/2 > countlevel:
                #integral /= 2
                integral /= i - index
                res[k, index] = int(np.rint(integral/countlevel))
                integral = 0
                index = -1
            if image[k, i] >= countlevel/2:
                if index == -1:
                    index = i
                    integral = 0
                #    integral += im[k, i]
                
                integral += image[k, i]
    return res

def subframes_preprocessing(filename, dirname, imsize, counts_threshold=1e-9, graphene_threshold=0, light_threshold=0, 
                            heavy_threshold=0.02, median_blur_diameter=39, gaussian_blur_radius=3,
                            minimum_graphene_area=0.5, dirt_border=100, save_fft=True, counts_divisor=None,
                            minimum_number_peaks=-1, baseline=0.002, countlevel=0.01, peaklength=5):
    """
    Returns tuple of the form:
            (filename, success, dirt coverage, counts divisor, angle of lattice rotation, mean peak radius)
        For files with more than 50% dirt coverage, the last 3 values will be 'None' and success will be False.
        
    """
    print('Working on: ' + filename)
    success = True
    #load image
    image = cv2.imread(dirname+filename, -1)
    if image is None:
        raise ValueError(dirname+filename+' is not an image file. Make sure you give the total path as input argument.')
    #image_org = image.copy()
    if only_process_images_of_shape is not None and image.shape != tuple(only_process_images_of_shape):
        success = False
        return (filename, 0, None, None, None, None, None, None, None, success)
    
    Peak = at.Peaking(image=image.copy(), imsize=imsize)
    #get mask to filter dirt and check if less than "minimum_graphene_area" is graphene in the image
#    mask = Peak.dirt_detector(dirt_threshold=dirt_threshold, median_blur_diam=median_blur_diameter,
#                              gaussian_blur_radius=gaussian_blur_radius)
    mask = create_mask(Peak, graphene_threshold, light_threshold, heavy_threshold, dirt_border)
    
    if remove_left_edge_number_pixels > 0:
        mask[:, :remove_left_edge_number_pixels] = 16
    
    graphene_area = float(np.sum(mask[mask==1]))/(np.shape(image)[0]*np.shape(image)[1])
    if graphene_area < minimum_graphene_area:
        success = False
        return (filename, graphene_area, None, None, None, None, None, None, None, success)
    
    # to improve peak finding set areas without graphene to mean intensity of graphene
    graphene_mean = np.mean(Peak.image[mask==1])
    Peak.image[mask!=1] = graphene_mean
    # Check for beam blanker artifacts at beginning of image
    if np.mean(Peak.image[:int(median_blur_diameter/2)]) > 1.1*graphene_mean:
        Peak.image[:int(median_blur_diameter/2)] = graphene_mean
        mask[:int(median_blur_diameter/2)] = 16
        print('Correcting for blanker artifacts in ' + filename + '.')
        
    #get angle of rotation and peak radius
    try:
        rotation, radius, number_peaks, peak_intensities_sum, ellipse_a, ellipse_b, angle = rotation_radius(Peak)
    except RuntimeError as detail:
        print('Error in '+ filename + ': ' + str(detail))
        rotation = ellipse_a = ellipse_b = angle = np.NaN
        number_peaks = peak_intensities_sum = radius = 0
        #peaks = None
        success = False
    else:
        if ((minimum_number_peaks > 0 and number_peaks < minimum_number_peaks) or (
             maximum_number_peaks > 0 and number_peaks > maximum_number_peaks)):
            success = False
    
    #Get counts divisor for image
#    if counts_divisor is None:        
#        counts_divisor = calculate_counts(image, threshold=counts_threshold)[0]
#    #Calculate actual counts in image and "translate" it to 16bit unsigned integer.
#    image[image<0]=0.0
#    image = np.asarray(np.rint(image/counts_divisor), dtype='uint16')
    # New version of calculating counts
    image = c_electron_counting.electron_counting(image, baseline=baseline, countlevel=countlevel, peaklength=peaklength)
    #dilate mask if dirt_border > 0
#    if dirt_border > 0:
#        mask = cv2.dilate(mask, np.ones((dirt_border, dirt_border)))
    #Set pixels where dirt was detected to maximum of 16bit range
    #image[mask==1] = 65535
    #save the image to disk
#    if not os.path.exists(dirname+'prep_'+dirname.split('/')[-2]+'/'):
#        os.makedirs(dirname+'prep_'+dirname.split('/')[-2]+'/')
#    if not os.path.exists(dirname+'mask_'+dirname.split('/')[-2]+'/'):
#        os.makedirs(dirname+'mask_'+dirname.split('/')[-2]+'/')
#    if save_fft:
#        if not os.path.exists(dirname+'fft_'+dirname.split('/')[-2]+'/'):
#            os.makedirs(dirname+'fft_'+dirname.split('/')[-2]+'/')
    
    if success:
        #cv2.imwrite(dirname+'subframes_preprocessing/'+filename, image)
        
        tifffile.imsave(dirname+'prep_'+dirname.split('/')[-2]+'/'+filename, image)
        tifffile.imsave(dirname+'mask_'+dirname.split('/')[-2]+'/'+filename, mask)
        
        if save_fft:
            #fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image_org)))).astype('float32')
            fft = np.log(np.abs(Peak.fft)).astype(np.float32)
            center = np.array(np.shape(image))/2
            ell = np.ones(np.shape(fft), dtype='float32')
            if np.mean(fft) > 0:
                ellipse_color = 1.5
            else:
                ellipse_color = 0.5
            cv2.ellipse(ell, (tuple(center), (ellipse_a*2, ellipse_b*2), -angle*180/np.pi), ellipse_color)
            cv2.ellipse(ell, (tuple(center), (ellipse_a*2*np.sqrt(3), ellipse_b*2*np.sqrt(3)), -angle*180/np.pi),
                        ellipse_color)
            if np.isfinite(rotation):
                endpoint = np.array((np.cos(rotation), -np.sin(rotation)))
                endpoint *= radius
                endpoint = np.rint(endpoint)
                endpoint += center
                cv2.line(ell, tuple(center.astype(np.int)), tuple(endpoint.astype(np.int)), ellipse_color)
            fft *= ell
            savesize = int(2.0*imsize/0.213)
            tifffile.imsave(dirname+'fft_'+dirname.split('/')[-2]+'/'+filename,
                            fft[center[0]-savesize:center[0]+savesize+1, center[1]-savesize:center[1]+savesize+1])
        
    
    #return image parameters
    return (os.path.splitext(filename)[0], graphene_area, number_peaks, peak_intensities_sum*(2-graphene_area), rotation,
            ellipse_a, ellipse_b, angle, success)

if __name__ == '__main__':
    
    overall_starttime = time.time()

    if not dirpath.endswith('/'):
        dirpath += '/'
    dirlist=os.listdir(dirpath)
    matched_dirlist = []
    for filename in dirlist:
        try:
            splitname = os.path.splitext(filename)
            #int(splitname[0][-4:])
            int(filename[:4])
            #if filename.startswith('image'):
            #    matched_dirlist.append(filename)
        except:
            pass
        else:
            matched_dirlist.append(filename)
    matched_dirlist.sort()
    #starttime = time.time()
    #matched_dirlist=matched_dirlist[400:600]
    if not os.path.exists(dirpath+'prep_'+dirpath.split('/')[-2]+'/'):
        os.makedirs(dirpath+'prep_'+dirpath.split('/')[-2]+'/')
    if not os.path.exists(dirpath+'mask_'+dirpath.split('/')[-2]+'/'):
        os.makedirs(dirpath+'mask_'+dirpath.split('/')[-2]+'/')
    if save_fft:
        if not os.path.exists(dirpath+'fft_'+dirpath.split('/')[-2]+'/'):
            os.makedirs(dirpath+'fft_'+dirpath.split('/')[-2]+'/')
    
    pool = Pool()
    res = [pool.apply_async(subframes_preprocessing, (filename, dirpath, imsize),
                            {'graphene_threshold': graphene_threshold, 'light_threshold': light_threshold,
                             'heavy_threshold': heavy_threshold, 'dirt_border':dirt_border, 'median_blur_diameter': 67,
                             'gaussian_blur_radius': 4, 'save_fft': save_fft,
                             'minimum_graphene_area': minimum_graphene_area, 'minimum_number_peaks': minimum_number_peaks,
                             'baseline': baseline, 'countlevel': countlevel, 'peaklength': peaklength,
                             }) for filename in matched_dirlist[0:only_process_this_number_of_images if
                                                                only_process_this_number_of_images > 0 else None]]
    res_list = [p.get() for p in res]
    pool.close()
    pool.terminate()

    #duration = time.time()-starttime

    #print('Time for calculation: %.2f s' %(duration,))
    
    res_list.sort()
    
    if not os.path.exists(dirpath+'prep_'+dirpath.split('/')[-2]+'/'):
        os.makedirs(dirpath+'prep_'+dirpath.split('/')[-2]+'/')
    
    frame_data_file = open(dirpath+'prep_'+dirpath.split('/')[-2]+'/'+'frame_init_' + dirpath.split('/')[-2] + '.txt',
                           'w')
    
    frame_data_file.write('#Informations about all frames of '+(dirpath.split('/')[-2]+'\n'))
    frame_data_file.write('#Created: ' + time.strftime('%Y/%m/%d %H:%M') + '\n')
    frame_data_file.write('#Imagesize in nm: {:.1f}\tgraphene threshold: {:f}\t'.format(imsize,graphene_threshold))
    frame_data_file.write('light threshold: {:f}\theavy threshold: {:f}\t'.format(light_threshold, heavy_threshold))
    frame_data_file.write('minimum number peaks: {:.0f}\t'.format(minimum_number_peaks))
    frame_data_file.write('Dirt border: {:n}\tminimum graphene area: {:f}\n'.format(dirt_border, minimum_graphene_area))
    frame_data_file.write('#Baseline: {:f}\tcountlevel: {:f}\tpeaklength: {:n}\n'.format(baseline, countlevel, peaklength))
    if os.path.isfile(os.path.join(dirpath, 'map_info.txt')):
        with open(os.path.join(dirpath, 'map_info.txt')) as infofile:
            for line in infofile:
                frame_data_file.write('#' + line)
    frame_data_file.write('#label\tgraphene\tnumpeak\ttuning\ttilt\tella\tellb\tellphi\n\n')
    
    for frame_data in res_list:
        if frame_data[-1]:
                frame_data_file.write('%s\t%.3f\t%d\t%.2f\t%.6f\t%.6f\t%.6f\t%.6f\n' % frame_data[0:-1])
    
    frame_data_file.close()
    
    overall_time = time.time() - overall_starttime
    
    print('Done analysing %d files in %.2f s.' %(only_process_this_number_of_images if
                                                 only_process_this_number_of_images > 0 else
                                                 len(matched_dirlist), overall_time))
        
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
