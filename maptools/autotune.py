# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:54:54 2015

@author: mittelberger
"""

import logging
import time
import os
import warnings
#from threading import Event

import numpy as np
import scipy.optimize
from scipy.ndimage import convolve, gaussian_filter, median_filter, uniform_filter, fourier_gaussian
from scipy.signal import fftconvolve, convolve2d, medfilt2d
#import cv2

#try:
#    import cv2
#except:
#    logging.warn('Could not import opencv')
#import matplotlib as plt

#try:
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import ViennaTools.ViennaTools as vt
    from ViennaTools import tifffile
#except:
#    try:
#        import ViennaTools as vt
#        from ViennaTools import tifffile
#    except:
#        logging.warn('Could not import Vienna tools!')

try:
    from . import autoalign
except:
    try:
        import autoalign
    except:
        pass


#global variable to store aberrations when simulating them (see function image_grabber() for details)
#global_aberrations = {'EHTFocus': 0, 'C12_a': 5, 'C12_b': 0, 'C21_a': 801.0, 'C21_b': 0, 'C23_a': -500, 'C23_b': 0}
global_aberrations = {'EHTFocus': 2, 'C12_a': 3, 'C12_b': -1, 'C21_a': 894, 'C21_b': 211.0, 'C23_a': -174, 'C23_b': 142.0}
    
class DirtError(Exception):
    """
    Custom Exception to specify that too much dirt was found in an image to perform a certain operation.
    """

def measure_symmetry(image, imsize):
    point_mirrored = np.flipud(np.fliplr(image)) 
    return autoalign.find_shift(image, point_mirrored, ratio=0.142/imsize/2)
    
def fourier_filter(image, imsize):
    peaks = find_peaks(image, imsize, second_order=True, half_line_thickness=3)
    xdata = np.mgrid[-7:8, -7:8]
    mask = gaussian2D(xdata, 0, 0, 4, 4, 1, 0)
    maskradius = int(np.shape(mask)[0]/2)
    fft = np.fft.fftshift(np.fft.fft2(image))
    fft_masked = np.zeros(np.shape(fft), dtype=fft.dtype)
    for order in peaks:
        for peak in order:
            if np.count_nonzero(peak) > 0:
                fft_masked[peak[0]-maskradius:peak[0]+maskradius+1, peak[1]-maskradius:peak[1]+maskradius+1] += \
                fft[peak[0]-maskradius:peak[0]+maskradius+1, peak[1]-maskradius:peak[1]+maskradius+1]*mask    
    
    return np.real(np.fft.ifft2(np.fft.fftshift(fft_masked)))
    
def symmetry_merit(image, imsize, mask=None):
    if mask is None:
        mean = np.mean(image)
    else:
        mean = np.mean(image[mask==0])
        
    ffil = fourier_filter(image, imsize)
    
    if mask is None:
        return (measure_symmetry(ffil, imsize)[1], np.var(ffil)/mean**2*100)
    else:
        return (measure_symmetry(ffil, imsize)[1]*(1.0-np.sum(mask)/np.size(mask)), np.var(ffil[mask==0]/mean**2*50))
    
        
#def get_frame_parameters(superscan):
#    """
#    Gets the current frame parameters of the microscope.
#    
#    Parameters
#    -----------
#    superscan : hardware source object
#        An instance of the superscan hardware source
#    
#    Returns
#    --------
#    frame_parameters : dictionary
#        Contains the following keys:
#        
#        - size_pixels: Number of pixels in x- and y-direction of the acquired frame as tuple (x,y)
#        - center: Offset for the center of the scanned area in x- and y-direction (nm) as tuple (x,y)
#        - pixeltime: Time per pixel (us)
#        - fov: Field-of-view of the acquired frame (nm)
#        - rotation: Scan rotation (deg)
#    """
#    
#    parameters = superscan.get_default_frame_parameters()
#    
#    return {'size_pixels': parameters.size, 'center': parameters.center_nm, 'pixeltime': parameters.pixel_time_us, \
#            'fov': parameters.fov_nm, 'rotation': parameters.rotation_deg}

def create_record_parameters(superscan, frame_parameters, detectors={'HAADF': False, 'MAADF': True}):
    """
    Returns the frame parameters in a form that they can be used in the record and view functions.
    
    Parameters
    -----------
    superscan : hardware source object
        An instance of the superscan hardware source
    
    frame_parameters : dictionary
        Frame parameters to set in the microscope. Possible keys are:
        
        - size_pixels: Number of pixels in x- and y-direction of the acquired frame as tuple (x,y)
        - center: Offset for the center of the scanned area in x- and y-direction (nm) as tuple (x,y)
        - pixeltime: Time per pixel (us)
        - fov: Field-of-view of the acquired frame (nm)
        - rotation: Scan rotation (deg)
        
    detectors : optional, dictionary
        By default, only MAADF is used. Dictionary has to be in the form:
        {'HAADF': False, 'MAADF': True}
    
    Returns
    --------
    record_parameters : dictionary
        It has the form: {'frame_parameters': frame_parameters, 'channels_enabled': [HAADF, MAADF, False, False]}
    """
    if frame_parameters is not None:
        parameters = superscan.get_default_frame_parameters()
        
        if frame_parameters.get('size_pixels') is not None:
            parameters['size'] = list(frame_parameters['size_pixels'])
        
        if frame_parameters.get('center') is not None:
            parameters['center_nm'] = list(frame_parameters['center'])
        
        if frame_parameters.get('pixeltime') is not None:
            parameters['pixel_time_us'] = frame_parameters['pixeltime']
        
        if frame_parameters.get('fov') is not None:
            parameters['fov_nm'] = frame_parameters['fov']
        
        if frame_parameters.get('rotation') is not None:
            parameters['rotation_rad'] = frame_parameters['rotation']*np.pi/180.0
    else:
        parameters = None
        
    if detectors is not None:
        channels_enabled = [detectors['HAADF'], detectors['MAADF'], False, False]
    else:
        channels_enabled = [False, True, False, False]

    return {'frame_parameters': parameters, 'channels_enabled': channels_enabled}
    

def graphene_generator(imsize, impix, rotation):
    rotation = rotation*np.pi/180
    
    #increase size of initially generated image by 20% to avoid missing atoms at the edges (image will be cropped
    #to actual size before returning it)
    image = np.zeros((int(impix*1.2), int(impix*1.2)))
    rotation_matrix = np.array( ( (np.cos(2.0/3.0*np.pi), np.sin(2.0/3.0*np.pi)), (-np.sin(2.0/3.0*np.pi), np.cos(2.0/3.0*np.pi)) ) )
    #define basis vectors of unit cell, a1 and a2
    basis_length = 0.142 * np.sqrt(3) * impix/float(imsize)
    a1 = np.array((np.cos(rotation), np.sin(rotation))) * basis_length
    a2 = np.dot(a1, rotation_matrix)
    #print(a1)
    #print(a2)
    a1position = np.array((0.0,0.0))
    a2position = np.array((0.0,0.0))
    a2direction = 1.0
    
    
    while (a1position < impix*2.4).all():
        success = True
        
        while success:
            firsta2 = a2position.copy()
            cellposition = a1position + a2position
            #print(str(a1position) + ', '  + str(a2position))
            #print(cellposition)
            
            #place atoms
            if (cellposition+a1/3.0+a2*(2.0/3.0) < impix*1.2).all() and (cellposition+a1/3.0+a2*(2.0/3.0) >= 0).all():
                success = True
                y,x = cellposition+a1/3.0+a2*(2.0/3.0)
                pixelvalues = distribute_intensity(x,y)
                pixelpositions = [(0,0), (0,1), (1,1), (1,0)]
                
                for i in range(len(pixelvalues)):
                    try:
                        image[np.floor(y)+pixelpositions[i][0],np.floor(x)+pixelpositions[i][1]] = pixelvalues[i]
                    except IndexError:
                        pass
                        #print('Could not put pixel at: ' + str((np.floor(y)+pixelpositions[i][0],np.floor(x)+pixelpositions[i][1])))
            else:
                success = False
                
            if (cellposition+a2/3.0+a1*(2.0/3.0) < impix*1.2).all() and (cellposition+a2/3.0+a1*(2.0/3.0) >= 0).all():
                success = True
                y,x = cellposition+a2/3.0+a1*(2.0/3.0)
                pixelvalues = distribute_intensity(x,y)
                pixelpositions = [(0,0), (0,1), (1,1), (1,0)]
                
                for i in range(len(pixelvalues)):
                    try:
                        image[np.floor(y)+pixelpositions[i][0],np.floor(x)+pixelpositions[i][1]] = pixelvalues[i]
                    except IndexError:
                        pass
                        #print('Could not put pixel at: ' + str((np.floor(y)+pixelpositions[i][0],np.floor(x)+pixelpositions[i][1])))
            else:
                success = False
        
            if not success and a2direction == 1:
                a2position = firsta2-a2
                a2direction = -1.0
                success = True
            elif not success and a2direction == -1:
                a2position += 3.0*a2
                a2direction = 1.0
            else:
                a2position += a2direction*a2
        
        a1position += a1
    
    start = int(impix * 0.1)
    return image[start:start+impix, start:start+impix]
    #return image

def distribute_intensity(x,y):
    """
    Distributes the intensity of a pixel at a non-integer-position (x,y) over four pixels.
    Returns a list of four values. The first element belongs to the pixel (floor(x), floor(y)),
    the following are ordered clockwise.
    """
    result = []
    result.append( ( 1.0-(x-np.floor(x)) ) * ( 1.0-(y-np.floor(y)) ) )
    result.append( ( x-np.floor(x) ) * ( 1.0-(y-np.floor(y)) ) )
    result.append( ( x-np.floor(x) ) * ( y-np.floor(y) ) )
    result.append( ( 1.0-(x-np.floor(x)) ) * ( y-np.floor(y) ) )
    
    return result
            
def dirt_detector(image, threshold=0.02, median_blur_diam=59, gaussian_blur_radius=3):
    """
    Returns a mask with the same shape as "image" that is 1 where there is dirt and 0 otherwise
    """
    #apply Gaussian Blur to improve dirt detection
    if gaussian_blur_radius > 0:
        #image = cv2.GaussianBlur(image, (0,0), gaussian_blur_radius)
        image = gaussian_filter(image, gaussian_blur_radius)
    #create mask
    mask = np.zeros(np.shape(image))
    mask[image>threshold] = 1
    #apply median blur to mask to remove noise influence
    if median_blur_diam % 2==0:
        median_blur_diam+=1
    #return cv2.medianBlur(mask, median_blur_diam)
    #return median_filter(mask, median_blur_diam)
    return np.rint(uniform_filter(mask, median_blur_diam)).astype('uint8')
    
def find_biggest_clean_spot(image):
    pass

def find_dirt_threshold(image, median_blur_diam=59, gaussian_blur_radius=3, debug_mode=False):
    # set up the search range
    search_range = np.mgrid[0:2*np.mean(image):30j]
    shape = np.array(np.shape(image))
    mask_sizes = []
    dirt_start = None
    dirt_end = None
    # go through list of thresholds and determine the amount of dirt with this threshold
    for threshold in search_range:
        mask_size = np.sum(dirt_detector(image, threshold=threshold, median_blur_diam=median_blur_diam,
                                         gaussian_blur_radius=gaussian_blur_radius)) / np.prod(shape)
        # remember value where the mask started to shrink
        if mask_size < 0.99 and dirt_start is None:
            dirt_start = threshold
        # remember value where the mask is almost zero and end search
        if mask_size < 0.01:
            dirt_end = threshold
            break
        
        mask_sizes.append(mask_size)
    
    # determine if there was really dirt present and return an appropriate threshold
    if dirt_end-dirt_start < 3*(search_range[1] - search_range[0]):
    # if distance between maximum and minimum mask size is very small, no dirt is present
    # set threshold to a value 25% over dirt_end
        threshold = dirt_end * 1.25
        print('here')
    else:
    # if distance between dirt_start and dirt_end is longer, set threshold to a value 
    # 10% smaller than mean to prevent missing dirt that is actually there in the image
        threshold = (dirt_end + dirt_start) * 0.45
    
    if debug_mode:
        return (threshold, search_range, np.array(mask_sizes))
    else:
        return threshold
    
def tuning_merit(imsize, average_frames, integration_radius, save_images, savepath, dirt_threshold, kwargs):
    intensities, image, mask = check_tuning(imsize, average_frames=average_frames, integration_radius=integration_radius, \
                        save_images=save_images, savepath=savepath, return_image = True, dirt_threshold=dirt_threshold, **kwargs)
    
    symmetry = symmetry_merit(image, imsize, mask=mask)
    
    print('sum intensities: ' + str(np.sum(intensities)) + '\tvar intensities: ' + str(np.std(intensities)/np.sum(intensities)) + '\tsymmetry: ' + str(symmetry))
    #return 1.0/(np.sum(intensities/1e6) + np.sum(symmetry) + np.count_nonzero(intensities)/10.0)
    return 1.0/(np.sum(intensities)/1e6 + np.sum(symmetry))
    

def kill_aberrations(superscan=None, as2=None, document_controller=None, average_frames=3, integration_radius=1, image=None, \
                    imsize=None, only_focus=False, save_images=False, savepath=None, event=None, dirt_threshold=0.015, \
                    steps = {'EHTFocus': 2, 'C12_a': 2, 'C12_b': 2, 'C21_a': 300, 'C21_b': 300, 'C23_a': 75, 'C23_b': 75}, \
                    keys = ['EHTFocus', 'C12_a', 'C12_b', 'C21_a', 'C21_b', 'C23_a', 'C23_b'], \
                    frame_parameters={'size_pixels': (512, 512), 'center': (0,0), 'pixeltime': 8, 'fov': 4, 'rotation': 0}):
    
    def logwrite(msg, level='info'):
        if document_controller is None:
            if level.lower() == 'info':
                logging.info(str(msg))
            elif level.lower() == 'warn':
                logging.warn(str(msg))
            elif level.lower() == 'error':
                logging.error(str(msg))
            else:
                logging.debug(str(msg))
        else:
            if level.lower() == 'info':
                document_controller.queue_task(lambda: logging.info(str(msg)))
            elif level.lower() == 'warn':
                document_controller.queue_task(lambda: logging.warn(str(msg)))
            elif level.lower() == 'error':
                document_controller.queue_task(lambda: logging.error(str(msg)))
            else:
                document_controller.queue_task(lambda: logging.debug(str(msg)))
    
    def merit(intensities):
#        return (np.sqrt((intensities[0]-intensities[1])**2 + (intensities[0]-intensities[2])**2 + (intensities[1]-intensities[2])**2 + 1))/ \
#                (np.count_nonzero(intensities)*np.sum(intensities**2)+1)
#        return 1 / (np.sum(intensities) - \
#            np.sqrt((intensities[0]-intensities[1])**2 + (intensities[0]-intensities[2])**2 + (intensities[1]-intensities[2])**2))
        return 1.0/np.sum(intensities)        
    
    #this is the simulated microscope
    global global_aberrations
    #global_aberrations = {'EHTFocus': 0.0, 'C12_a': 1.0, 'C12_b': -1.0, 'C21_a': 600.0, 'C21_b': 300.0, 'C23_a': 120, 'C23_b': -90.0}
    if superscan is None or as2 is None:
        online=False
        logwrite('Going to offline mode because no instance of superscan and as2 was provided.')
    else:
        online=True
        global_aberrations = {'EHTFocus': 0.0, 'C12_a': 0.0, 'C12_b': 0.0, 'C21_a': 0.0, 'C21_b': 0.0, 'C23_a': 0.0, 'C23_b': 0.0}
    
    total_tunings = []
#    total_lens = []
    counter = 0

    kwargs = {'EHTFocus': 0.0, 'C12_a': 0.0, 'C12_b': 0.0, 'C21_a': 0.0, 'C21_b': 0.0, 'C23_a': 0.0, 'C23_b': 0.0, 'relative_aberrations': True,\
            'reset_aberrations': False, 'frame_parameters': frame_parameters, 'superscan': superscan, 'as2': as2}
    
    if not online:    
        if image is not None and imsize is not None:
            kwargs['image'] = image
            kwargs['imsize'] = imsize
        else:
            raise RuntimeError('You have to provide an image and its size (in nm) to use the offline mode.')
    
#    steps = []        
#    if 'EHTFocus' in keys:
#        steps.append(focus_step)
#    if 'C12_a' in keys:
#        steps.append(astig2f_step)
#        steps.append(astig2f_step)
#    if 'C21_a' in keys:
#        steps.append(coma_step)
#        steps.append(coma_step)
#    if 'C23_a' in keys:
#        steps.append(astig3f_step)
#        steps.append(astig3f_step)
    
    try:
        #current = check_tuning(frame_parameters['fov'], average_frames=average_frames, integration_radius=integration_radius, \
        #                       save_images=save_images, savepath=savepath, dirt_threshold=dirt_threshold, **kwargs)
        current = tuning_merit(frame_parameters['fov'], average_frames, integration_radius, save_images, savepath, dirt_threshold, kwargs)
    except RuntimeError as detail:
        #current = np.ones(12)
        current = 1e5
        print(str(detail))
    except DirtError:
        logwrite('Tuning ended because of too high dirt coverage.', level='warn')
        raise
    
    total_tunings.append(current)
    logwrite('Appending start value: ' + str(current))
    
    while counter < 11:
        if event is not None and event.is_set():
            break
        start_time = time.time()
        aberrations_last_run = global_aberrations.copy()
        if counter > 0 and len(total_tunings) < counter+1:
            logwrite('Finished tuning because no improvements could be found anymore.')
            break

        if len(total_tunings) > 1:        
            logwrite('Improved tuning by '+str(np.abs((total_tunings[-2]-total_tunings[-1])/((total_tunings[-2]+total_tunings[-1])*0.5)*100))+'%.')

        if len(total_tunings) > 2:
            #if total_tunings[-2]-total_tunings[-1] < 0.1*(total_tunings[-3]-total_tunings[-2]):
            if np.abs((total_tunings[-2]-total_tunings[-1])/((total_tunings[-2]+total_tunings[-1])*0.5)) < 0.02:
                logwrite('Finished tuning successfully after %d runs.' %(counter))
                break
        
        logwrite('Starting run number '+str(counter+1))
        part_tunings = []
#        part_lens = []
        
        for key in keys:
            
            if event is not None and event.is_set():
                break
            
#            if counter == 0 and i==0:
#                #total_tunings.append(merit(current))
#                total_tunings.append(current)
#                #logwrite('Appending start value: ' + str(merit(current)))
#                logwrite('Appending start value: ' + str(current))
#                #total_lens.append(np.count_nonzero(current))

            logwrite('Working on: '+ key)
            #vt.as2_set_control(controls[i], start+steps[i]*1e-9)
            #time.sleep(0.1)
            step_multiplicator=1
            while step_multiplicator < 8:
                changes = 0.0
                kwargs[key] = steps[key]*step_multiplicator
                changes += steps[key]*step_multiplicator
                try:            
#                    plus = check_tuning(frame_parameters['fov'], average_frames=average_frames, integration_radius=integration_radius, \
#                                        save_images=save_images, savepath=savepath, mode=mode, dirt_threshold=dirt_threshold, **kwargs)
                    plus = tuning_merit(frame_parameters['fov'], average_frames, integration_radius, save_images, savepath, dirt_threshold, kwargs)
                except RuntimeError:
                    #plus = np.ones(12)
                    plus = 1e5
                except DirtError:
                    if online:
                        for key, value in global_aberrations.items():
                            kwargs[key] = aberrations_last_run[key]-value
                        image_grabber(acquire_image=False, **kwargs)

                    logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                    raise
    
                #passing 2xstepsize to image_grabber to get from +1 to -1
                kwargs[key] = -2.0*steps[key]*step_multiplicator
                changes += -2.0*steps[key]*step_multiplicator
                try:
#                    minus = check_tuning(frame_parameters['fov'], average_frames=average_frames, integration_radius=integration_radius, \
#                                        save_images=save_images, savepath=savepath, mode=mode, dirt_threshold=dirt_threshold, **kwargs)
                    minus = tuning_merit(frame_parameters['fov'], average_frames, integration_radius, save_images, savepath, dirt_threshold, kwargs)
                except RuntimeError:
                    #minus = np.ones(12)
                    minus = 1e5
                except DirtError:
                    if online:
                        for key, value in global_aberrations.items():
                            kwargs[key] = aberrations_last_run[key]-value
                        image_grabber(acquire_image=False, **kwargs)
                    
                    logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                    raise
                
#                if merit(minus) < merit(plus) and merit(minus) < merit(current) \
#                    and np.count_nonzero(minus) >= np.count_nonzero(plus) and np.count_nonzero(minus) >= np.count_nonzero(current):
                if minus < plus and minus < current:
                    direction = -1
                    current = minus
                    #setting the stepsize to new value
                    steps[key] *= step_multiplicator
                    break
#                elif merit(plus) < merit(minus) and merit(plus) < merit(current) \
#                    and np.count_nonzero(plus) >= np.count_nonzero(minus) and np.count_nonzero(plus) >= np.count_nonzero(current):
                elif plus < minus and plus < current:
                    direction = 1
                    current = plus
                    #setting the stepsize to new value
                    steps[key] *= step_multiplicator
                    #Setting aberrations to values of 'plus' which where the best so far
                    kwargs[key] = 2.0*steps[key]*step_multiplicator
                    changes += 2.0*steps[key]*step_multiplicator
                    #update hardware
                    image_grabber(acquire_image=False, **kwargs)
                    break
                else:
                    kwargs[key] = -changes
                    #update hardware
                    image_grabber(acquire_image=False, **kwargs)
                    logwrite('Doubling the stepsize of '+key+'.')
                    kwargs[key] = 0
                    step_multiplicator *= 2
            #This 'else' belongs to the while loop. It is executed when the loop ends 'normally', e.g. not through break or continue
            else:
                #kwargs[keys[i]] = -changes
                #update hardware
                #image_grabber(acquire_image=False, **kwargs)
                logwrite('Could not find a direction to improve '+key+'. Going to next aberration.')
                #reduce stepsize for next iteration
                steps[key] *= 0.5
                #kwargs[keys[i]] = 0
                continue
            
            small_counter = 1
            while True:
                small_counter+=1
                #vt.as2_set_control(controls[i], start+direction*small_counter*steps[i]*1e-9)
                #time.sleep(0.1)
                kwargs[key] = direction*steps[key]
                changes += direction*steps[key]
                try:
#                    next_frame = check_tuning(frame_parameters['fov'], average_frames=average_frames, integration_radius=integration_radius, \
#                                            save_images=save_images, savepath=savepath, mode=mode, dirt_threshold=dirt_threshold, **kwargs)
                    next_frame = tuning_merit(frame_parameters['fov'], average_frames, integration_radius, save_images, savepath, dirt_threshold, kwargs)
                except RuntimeError:
                    #vt.as2_set_control(controls[i], start+direction*(small_counter-1)*steps[i]*1e-9)
                    kwargs[key] = -direction*steps[key]
                    changes -= direction*steps[key]
                    #update hardware
                    image_grabber(acquire_image=False, **kwargs)
                    break
                except DirtError:
                    if online:
                        for key, value in global_aberrations.items():
                            kwargs[key] = aberrations_last_run[key]-value
                        image_grabber(acquire_image=False, **kwargs)
                    logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                    raise
                
                #if merit(next_frame) >= merit(current) or np.count_nonzero(next_frame) < np.count_nonzero(current):
                if next_frame >= current:
                    #vt.as2_set_control(controls[i], start+direction*(small_counter-1)*steps[i]*1e-9)
                    kwargs[key] = -direction*steps[key]
                    changes -= direction*steps[key]
                    #update hardware
                    image_grabber(acquire_image=False, **kwargs)
                    #part_tunings.append(merit(current))
                    part_tunings.append(current)
                    #part_lens.append(np.count_nonzero(current))
                    break
                current = next_frame
            #only keep changes if they improve the overall tuning
            if len(total_tunings) > 0:
                #if merit(current) > np.amin(total_tunings) or np.count_nonzero(current) < np.amax(total_lens):
                if current > np.amin(total_tunings):
                    #vt.as2_set_control(controls[i], start)
                    kwargs[key] = -changes
                    #update hardware
                    try:
#                        current = check_tuning(frame_parameters['fov'], average_frames=average_frames, integration_radius=integration_radius, \
#                                                save_images=save_images, savepath=savepath, mode=mode, dirt_threshold=dirt_threshold, **kwargs)
                        current  = tuning_merit(frame_parameters['fov'], average_frames, integration_radius, save_images, savepath, dirt_threshold, kwargs)
                        
                    except DirtError:
                        if online:
                            for key, value in global_aberrations.items():
                                kwargs[key] = aberrations_last_run[key]-value
                            image_grabber(acquire_image=False, **kwargs)
                        logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                        raise
                    except:
                        pass
                    logwrite('Dismissed changes at '+ key)
            #reduce stepsize for next iteration
            steps[key] *= 0.5
            #set current working aberration to zero
            kwargs[key] = 0
        
        if len(part_tunings) > 0:
            logwrite('Appending best value of this run to total_tunings: '+str(np.amin(part_tunings)))
            total_tunings.append(np.amin(part_tunings))
            #total_lens.append(np.amax(part_lens))
        logwrite('Finished run number '+str(counter+1)+' in '+str(time.time()-start_time)+' s.')
        counter += 1
    
    if save_images:    
        try:
#            check_tuning(frame_parameters['fov'], average_frames=0, integration_radius=integration_radius, \
#                        save_images=save_images, savepath=savepath, mode=mode, dirt_threshold=dirt_threshold, **kwargs)
            tuning_merit(frame_parameters['fov'], average_frames, integration_radius, save_images, savepath, dirt_threshold, kwargs)
        except DirtError:
            if online:
                for key, value in global_aberrations.items():
                    kwargs[key] = aberrations_last_run[key]-value
                image_grabber(acquire_image=False, **kwargs)
            logwrite('Tuning ended because of too high dirt coverage.', level='warn')
            raise
        except:
            pass
    else:
        image_grabber(acquire_image=False, **kwargs)
    
    return global_aberrations

def image_grabber(superscan = None, as2 = None, acquire_image=True, **kwargs):
    """
    acquire_image defines if an image is taken and returned or if just the correctors are updated.
    
    kwargs contains all possible values for the correctors : 
        These are all lens aberrations up to threefold astigmatism. If an image is given, the function will simulate aberrations to this image and add poisson noise to it.
        If not, an image with the current frame parameters and the corrector parameters given in kwargs is taken.
    
    Possible Parameters
    -------------------
    
    lens aberrations : 
        EHTFocus, C12_a, C12_b, C21_a, C21_b, C23_a,  C23_b (in nm)
    
    image : 
        (as numpy array)
            
    relative_aberrations : True/False
            If 'relative_aberrations' is included and set to True, image_grabber will get the current value for each control first and add the given value for the respective aberration
            to the current value. Otherwise, each aberration in kwargs is just set to the value given there.    
            
    reset_aberrations : True/False    
        If 'reset_aberrations' is included and set to True, image_grabber will set each aberration back to its original value after acquiring an image. This is a good choice if
        you want to try new values for the aberration correctors bur are not sure you want to keep them.
    
    frame_parameters : dictionary
        Contains the frame parameters for acquisition. See function create_record_parameters() for details.
        
    detectors : dictionary
        Contains the dectectors used for acquisition. See function create_record_parameters() for details.
        
    Example call of image_grabber:
    ------------------------------
    
    result = image_grabber(EHTFocus=1, C12_a=0.5, image=graphene_lattice, imsize=10)
    
    Note that the Poisson noise is added modulatory, e.g. each pixel value is replaced by a random number from a Poisson distribution that has the original pixel value as
    its mean. That means you can control the noise level by changing the mean intensity in your image.
    """
    keys = ['EHTFocus', 'C12_a', 'C12_b', 'C21_a', 'C21_b', 'C23_a', 'C23_b']
    controls = ['EHTFocus', 'C12.a', 'C12.b', 'C21.a', 'C21.b', 'C23.a', 'C23.b']
    print(kwargs.get('frame_parameters'))
    originals = {}
    global global_aberrations
    #print(kwargs)
    if not 'image' in kwargs:
        for i in range(len(keys)):
            if keys[i] in kwargs:
                if as2 is None:
                    raise RuntimeError('You have to provide an instance of as2 to perform as2-related operations.')
                offset=0.0
                offset2=0.0
                if kwargs.get('reset_aberrations'):
                    originals[controls[i]] = vt.as2_get_control(as2, controls[i])
                if kwargs.get('relative_aberrations'):
                    offset = vt.as2_get_control(as2, controls[i])
                    offset2 = global_aberrations[keys[i]]
                vt.as2_set_control(as2, controls[i], offset+kwargs[keys[i]]*1e-9)
                global_aberrations[keys[i]] = offset2+kwargs[keys[i]]
        if acquire_image:
            if superscan is None:
                raise RuntimeError('You have to provide an instance of superscan to perform superscan-related operations.')
            record_parameters = create_record_parameters(superscan, kwargs.get('frame_parameters'), detectors=kwargs.get('detectors'))
            im = superscan.record(**record_parameters)
            if len(im) > 1:
                im2 = []
                for entry in im:
                    im2.append(entry.data)
            else:
                im2 = im[0].data
            if len(originals) > 0:
                for key in originals.keys():
                    vt.as2_set_control(key, originals[key])
            return im2
    else:
        im = kwargs['image']
        imsize = kwargs['imsize']
        
        shape = np.shape(im)
        
#        kernelsize=[63.5,63.5]
#        y,x = np.mgrid[-kernelsize[0]:kernelsize[0]+1.0, -kernelsize[1]:kernelsize[1]+1.0]/2.0
        
        #raw_kernel = 0
        keys = ['EHTFocus', 'C12_a', 'C12_b', 'C21_a', 'C21_b', 'C23_a', 'C23_b']
        start_keys = ['start_EHTFocus', 'start_C12_a', 'start_C12_b', 'start_C21_a', 'start_C21_b', 'start_C23_a', 'start_C23_b']
        aberrations = np.zeros(len(keys))
        
        for i in range(len(keys)):            
            if kwargs.get(keys[i]):
                offset=0.0
                if kwargs.get('relative_aberrations'):
                    offset = global_aberrations[keys[i]]
                aberrations[i] = offset+kwargs[keys[i]]
                if kwargs.get(start_keys[i]):
                    aberrations[i] -= kwargs[start_keys[i]]
            #if aberrations should not be reset, change global_aberrations
                if not kwargs.get('reset_aberrations'):
                    global_aberrations[keys[i]] = aberrations[i]
            else:
                aberrations[i] = global_aberrations[keys[i]]

        if acquire_image:
            #Create x and y coordinates such that resulting beam has the same scale as the image.
            #The size of the kernel which is used for image convolution is chosen to be "1/kernelsize" of the image size (in pixels)
            kernelsize = 2
            kernelpixel = int(shape[0]/kernelsize)
            frequencies = np.matrix(np.fft.fftshift(np.fft.fftfreq(kernelpixel, imsize/shape[0])))
            x = np.array(np.tile(frequencies, np.size(frequencies)).reshape((kernelpixel,kernelpixel)))
            y = np.array(np.tile(frequencies.T, np.size(frequencies)).reshape((kernelpixel,kernelpixel)))
            
            #compute aberration function up to threefold astigmatism
            #formula taken from "Advanced Computing in Electron Microscopy", Earl J. Kirkland, 2nd edition, 2010, p. 18
            #wavelength for 60 keV electrons: 4.87e-3 nm
            #first line: defocus and twofold astigmatism
            #second line: coma
            #third line: threefold astigmatism
            raw_kernel = np.pi*4.87e-3*(-aberrations[0]*(x**2+y**2) + np.sqrt(aberrations[1]**2+aberrations[2]**2)*(x**2+y**2)*np.cos(2*(np.arctan2(y,x)-np.arctan2(aberrations[2], aberrations[1]))) \
                + (2.0/3.0)*np.sqrt(aberrations[3]**2+aberrations[4]**2)*4.87e-3*np.sqrt(x**2+y**2)**3*np.cos(np.arctan2(y,x)-np.arctan2(aberrations[4], aberrations[3])) \
                + (2.0/3.0)*np.sqrt(aberrations[5]**2+aberrations[6]**2)*4.87e-3*np.sqrt(x**2+y**2)**3*np.cos(3*(np.arctan2(y,x)-np.arctan2(aberrations[6], aberrations[5]))))
                
            kernel = np.cos(raw_kernel)+1j*np.sin(raw_kernel)
            aperture = np.zeros(kernel.shape)
            #Calculate size of 25 mrad aperture in k-space for 60 keV electrons
            aperturesize = (0.025/kernelsize)*imsize/4.87e-3
            #cv2.circle(aperture, tuple((np.array(kernel.shape)/2).astype('int')), int(kernelsize[0]/4.86), 1, thickness=-1)
            #cv2.circle(aperture, tuple((np.array(kernel.shape)/2).astype('int')), int(np.rint(aperturesize)), 1, thickness=-1)
            draw_circle(aperture, tuple((np.array(kernel.shape)/2).astype('int')), int(np.rint(aperturesize)), color=1)
            
            kernel *= aperture
            kernel = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kernel))))**2
            kernel /= np.sum(kernel)
            #im = cv2.filter2D(im, -1, kernel)
            im = fftconvolve(im, kernel, mode='same')
            shape=im.shape
            im = np.random.poisson(lam=im.flatten(), size=np.size(im)).astype(im.dtype)
            return im.reshape(shape).astype('float32')
            #return kernel

def positive_angle(angle):
    """
    Calculates the angle between 0 and 2pi from an input angle between -pi and pi (all angles in rad)
    """
    if angle < 0:
        return angle  + 2*np.pi
    else:
        return angle

def check_tuning(imagesize, im=None, check_astig=False, average_frames=0, integration_radius=0, save_images=False, savepath=None, \
                process_image=True, return_image = False, dirt_threshold = 0.015, **kwargs):

    global global_aberrations                    
                    
    if kwargs.get('imsize') is None:
        kwargs['imsize'] = imagesize
    else:
        imagesize=kwargs['imsize']
        
    if (process_image or im is None) and average_frames < 2:
        if im is not None and kwargs.get('image') is None:
            kwargs['image'] = im
        im = image_grabber(**kwargs)
            
    if average_frames > 1:
        im = []
        #Acquire only one image in the first place to avoid "stacking" of aberations
        single_image = image_grabber(**kwargs)
        im.append(single_image)
        
        #remove aberrations from kwargs fore next frames
        kwargs2 = kwargs.copy()
        if kwargs.get('relative_aberrations'):
                if not kwargs.get('reset_aberrations'):
                    keys = ['EHTFocus', 'C12_a', 'C12_b', 'C21_a', 'C21_b', 'C23_a', 'C23_b']
                    for key in keys:
                        kwargs2.pop(key, 0)
        
        for i in range(average_frames-1):
            im.append(image_grabber(**kwargs2))
            
#Apply dirt detection, but only when real images are used
    #Averaging is only done with real images
    mask = None
    if average_frames > 1:
        for image in im:
            mask = dirt_detector(image, threshold=0.015)
            if np.sum(mask) > 0.5*np.prod(np.array(np.shape(image))):
                raise DirtError('Cannot check tuning of images with more than 50% dirt coverage.')
    #If no image is provided or just a tuning check without real or simulated acquisition should be done it is real data
    elif kwargs.get('image') is None or not process_image:
        mask = dirt_detector(im, threshold=0.015)
        if np.sum(mask) > 0.5*np.prod(np.array(np.shape(im))):
            raise DirtError('Cannot check tuning of images with more than 50% dirt coverage.')
    
    if save_images:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        name = str(int(time.time()*100))+'.tif'
        logfile = open(savepath+'log.txt', 'a')
#        kwargs2 = kwargs.copy()
#        kwargs2.pop('image', 0)
#        logfile.write(name+': '+str(kwargs2)+'\n')
        logfile.write(name+': '+str(global_aberrations)+'\n')
        logfile.close()
        if average_frames < 2:
            tifffile.imsave(savepath+name, im.astype('float32'))
        else:
            tifffile.imsave(savepath+name, im[0].astype('float32'))
            
    try:
        peaks_first, peaks_second = find_peaks(im, imagesize, integration_radius=integration_radius, second_order=True, position_tolerance=9)
    except RuntimeError as detail:
        raise RuntimeError('Tuning check failed. Reason: '+ str(detail))

    intensities = []
    for peak in peaks_first:
        intensities.append(peak[3])
    for peak in peaks_second:
        intensities.append(peak[3])
    intensities=np.array(intensities)
    
    if return_image:
        return (intensities, (im if average_frames < 2 else im[0]) , mask)
    else:        
        return intensities        

def optimize_tuning(imsize, im=None, astig_stepsize=0.1, focus_stepsize=1.0, tune_astig=False, save_iterations=False):
    #Take a focus series to find correct astigmatism 
    astig_step = astig_stepsize
    focus_step = 0.0
    angle_difference = 0.0
    counter = 0
    best_focus = 0.0
    overfocus = 0.0
    #list of all the images already investigated
    focus_tunings = []
    focus_values = []
    while np.abs(angle_difference) < np.pi/2.4 and counter < 6:
        counter += 1
        focus_step += focus_stepsize
        #appending new values to list, sorted by defocus
        focus_tunings.insert(0, check_tuning(imsize, defocus=-focus_step, im=im, check_astig=True))
        focus_values.insert(0,-focus_step)
        focus_tunings.append(check_tuning(imsize, defocus=focus_step, im=im, check_astig=True))
        focus_values.append(focus_step)
        #check if angle could not be determined in one of the last steps
        if focus_tunings[-1] == 0 or focus_tunings[0] == 0:
            #if more 2 or more angles were already found, look for a bigger angle in the existing list
            if len(focus_tunings)-focus_tunings.count(0) > 1:
                for i in range(len(focus_tunings)-1):
                    if focus_tunings[i] == 0 or focus_tunings[i+1] == 0:
                        continue
                    elif np.abs(focus_tunings[i]-focus_tunings[i+1]) > np.abs(angle_difference):
                        angle_difference = focus_tunings[i][2] - focus_tunings[i+1][2]
                        best_focus = np.mean((focus_values[i],focus_values[i+1]))
                        overfocus = i+1
        else:
            angle_difference = focus_tunings[0][2] - focus_tunings[-1][2]
            best_focus = 0.0
            overfocus = len(focus_values)
    
    #if no angle of astigmatism could be determined, adjust only focus
    if counter >= 6:
        logging.warn('No angle of astigmatism could be found. Adjusting only focus.')
        return optimize_focus(imsize, im=im)
    
    #If more than two angles were found, look again for the defocus were the angle "flips"
    if len(focus_tunings)-focus_tunings.count(0) > 2:
        for i in range(len(focus_tunings)-1):
            if focus_tunings[i] == 0 or focus_tunings[i+1] == 0:
                continue
            elif np.abs(focus_tunings[i]-focus_tunings[i+1]) >= np.pi/2.4:
                angle_difference = focus_tunings[i][2] - focus_tunings[i+1][2]
                best_focus = np.mean((focus_values[i],focus_values[i+1]))
                print(best_focus)
                overfocus = i+1
                break
        
    #Now start tuning the astigmatism while focus is set to overfocus
    astig_tunings = []
    astig_values = []
    #Find direction of astigmatism correction
    astig_tunings.append(check_tuning(imsize, defocus=overfocus, astig=[astig_step, astig_step*np.tan(focus_tunings[overfocus][2])]))
    astig_values.append(astig_step)
    astig_tunings.insert(0, check_tuning(imsize, defocus=overfocus, astig=[-astig_step, -astig_step*np.tan(focus_tunings[overfocus][2])]))
    astig_values.insert(0, -astig_step)
    
    current = np.var(focus_tunings[overfocus][0])/np.sum(focus_tunings[overfocus][0])
    plus = np.var(astig_tunings[1][0])/np.sum(astig_tunings[1][0])
    minus = np.var(astig_tunings[0][0])/np.sum(astig_tunings[0][0])
    
    if plus < minus and plus < current:
        pass
        
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
    
def gaussian2D(xdata, x0, y0, x_std, y_std, amplitude, offset):
    x0, y0, x_std, y_std, amplitude, offset = float(x0), float(y0), float(x_std), float(y_std), float(amplitude), float(offset)
    return (amplitude*np.exp( -0.5*( ((xdata[1]-x0)/x_std)**2 + ((xdata[0]-y0)/y_std)**2 ) ) + offset)#.ravel()

def hyperbola1D(xdata, a, offset):
    a, offset = float(a), float(offset)
    return np.abs(1.0/(a*xdata))+offset

def find_peaks(im, imsize, half_line_thickness=5, position_tolerance=5, integration_radius=0, second_order=False, debug_mode=False, mode='magnitude'):
    """
        This function can find the 6 first-order peaks in the FFT of an atomic-resolution image of graphene.
        Input:
                im: Image as a numpy array or any type that can be simply casted to a numpy array.
                imsize: Size of the input image in nm.
                return_type: 'magnitude', 'phase' or 'amplitude', depending on the values of the fft you want to use.
                            When chosing 'amplitude' or 'phase', the abolutes of the respective parts of the fft are summed
                            (Otherwise all values would be close to zero, because there is a sign change at the peak location).
        Output:
                List of tuples that contain the coordinates of the reflections. The tuples have the form (y, x, intensity_of_peak_maximum)
                If no peaks were found the return value will be None.
                Note that the returned intesities might be smaller than that of the raw fft because of the processing done in the function.
    """

    shape = np.shape(im)
    fft = np.fft.fftshift(np.fft.fft2(im))
    #If more than one image are passed to find_peaks, compute average of their fft's before going on
    if len(shape) > 2:
        fft  = np.mean(fft, axis=0)
        shape = shape[1:]
            
    center = np.array(np.array(shape)/2, dtype='int')
    
    if mode.lower() == 'phase':
        fft_raw = np.abs(np.imag(fft))
    elif mode.lower() == 'amplitude':
        fft_raw = np.abs(np.real(fft))
    else:    
        fft_raw = np.abs(fft)
    
    fft = np.abs(fft)
    
    first_order = imsize/0.213
    second_order_peaks = imsize/0.123
    
    #make sure that areas of first and second_order peaks don't overlap
    if position_tolerance > (second_order_peaks-first_order)/np.sqrt(2)-1:
        position_tolerance = int(np.rint((second_order_peaks-first_order)/np.sqrt(2)-1))
    
    #print('center: '+str(center)+', first_order: '+str(first_order))
    #blank out bright spot in center of fft
    #cv2.circle(fft, tuple(center), int(np.rint(first_order/2.0)), -1, -1)
    draw_circle(fft, tuple(center), int(np.rint(first_order/2.0)))
    
    #prevent infinite values when cross would be calculated until central pixel because of too high half line thickness
    if half_line_thickness > int(np.rint(first_order/2.0))-1:
        half_line_thickness = int(np.rint(first_order/2.0))-1

    #std_dev_fft = np.std(fft[fft>-1])
    mean_fft = np.mean(fft[fft>-1])
    #Fit horizontal and vertical lines with hyperbola
    cross = np.zeros(shape)
    for i in range(-half_line_thickness, half_line_thickness+1):
        horizontal = fft[center[0]+i,:]
        vertical = fft[:, center[1]+i]
        xdata = np.mgrid[:shape[1]][horizontal>-1] - center[1]
        ydata = np.mgrid[:shape[0]][vertical>-1] - center[0]
        horizontal = horizontal[horizontal>-1]
        vertical = vertical[vertical>-1]
        horiz_a = 1.0/((np.mean(horizontal[int(len(horizontal)*0.6)-3:int(len(horizontal)*0.6)+4])-np.mean(horizontal[int(len(horizontal)*0.7)-3:int(len(horizontal)*0.7)+4]))*2.0*xdata[int(len(horizontal)*0.6)])
        vert_a = 1.0/((np.mean(vertical[int(len(vertical)*0.6)-3:int(len(vertical)*0.6)+4])-np.mean(vertical[int(len(vertical)*0.7)-3:int(len(vertical)*0.7)+4]))*2.0*ydata[int(len(vertical)*0.6)])
        horizontal_popt, horizontal_pcov = scipy.optimize.curve_fit(hyperbola1D, xdata[:len(xdata)/2], horizontal[:len(xdata)/2], p0=(horiz_a, 0))
        vertical_popt, vertical_pcov = scipy.optimize.curve_fit(hyperbola1D, ydata[:len(ydata)/2], vertical[:len(ydata)/2], p0=(vert_a, 0))
        
        cross[center[0]+i, xdata+center[1]] = hyperbola1D(xdata, *horizontal_popt)-1.5*mean_fft
        cross[ydata+center[0], center[1]+i] = hyperbola1D(ydata, *vertical_popt)-1.5*mean_fft
    
    fft-=cross
    
    if (4*int(first_order) < center).all():
        fft[center[0]-4*int(first_order):center[0]+4*int(first_order)+1, center[1]-4*int(first_order):center[1]+4*int(first_order)+1] *= gaussian2D(np.mgrid[center[0]-4*int(first_order):center[0]+4*int(first_order)+1, center[1]-4*int(first_order):center[1]+4*int(first_order)+1], shape[1]/2, shape[0]/2, 0.75*first_order, 0.75*first_order, -1, 1)#.reshape(np.array(shape)/2)
    else:
        fft *= gaussian2D(np.mgrid[:shape[0], :shape[1]], shape[1]/2, shape[0]/2, 0.75*first_order, 0.75*first_order, -1, 1)
    #find peaks
    success = False
    counter = 0
    
    while success is False:
        counter += 1
        if counter > np.sqrt(shape[0]):
            #raise RuntimeError('No peaks could be found in the FFT of im.')
            break
        if second_order:
            peaks = np.zeros((2,6,4))
        else:
            peaks = np.zeros((6,4))
        #peaks = []
        first_peak = np.unravel_index(np.argmax(fft), shape)+(np.amax(fft), )
        area_first_peak = fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1]

        if first_peak[2] < np.mean(area_first_peak)+6*np.std(area_first_peak):
            fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 1
        elif np.sqrt(np.sum((np.array(first_peak[0:2])-center)**2)) < first_order*0.6667 or np.sqrt(np.sum((np.array(first_peak[0:2])-center)**2)) > first_order*1.5:
            fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 2
        else:
            try:            
                #peaks.append(first_peak+(np.sum(fft_raw[first_peak[0]-integration_radius:first_peak[0]+integration_radius+1, first_peak[1]-integration_radius:first_peak[1]+integration_radius+1]),))
                if second_order:
                    peaks[0,0] = np.array(first_peak+(np.sum(fft_raw[first_peak[0]-integration_radius:first_peak[0]+integration_radius+1, first_peak[1]-integration_radius:first_peak[1]+integration_radius+1]),))
                else:
                    peaks[0] = np.array(first_peak+(np.sum(fft_raw[first_peak[0]-integration_radius:first_peak[0]+integration_radius+1, first_peak[1]-integration_radius:first_peak[1]+integration_radius+1]),))
                
                for i in range(1,6):
                    rotation_matrix = np.array( ( (np.cos(i*np.pi/3), -np.sin(i*np.pi/3)), (np.sin(i*np.pi/3), np.cos(i*np.pi/3)) ) )
                    if second_order:
                        next_peak = np.rint(np.dot( rotation_matrix , peaks[0,0,0:2]-center ) + center).astype(int)
                    else:
                        next_peak = np.rint(np.dot( rotation_matrix , peaks[0,0:2]-center ) + center).astype(int)
                    area_next_peak = fft[next_peak[0]-position_tolerance:next_peak[0]+position_tolerance+1, next_peak[1]-position_tolerance:next_peak[1]+position_tolerance+1]
                    max_next_peak = np.amax(area_next_peak)
                   #if  max_next_peak > mean_fft + 5.0*std_dev_fft:#peaks[0][2]/4:
                    if max_next_peak > np.mean(area_next_peak)+5*np.std(area_next_peak):
                        next_peak += np.array( np.unravel_index( np.argmax(area_next_peak), np.shape(area_next_peak) ) ) - position_tolerance
                        #peaks.append(tuple(next_peak)+(max_next_peak,np.sum(fft_raw[next_peak[0]-integration_radius:next_peak[0]+integration_radius+1, next_peak[1]-integration_radius:next_peak[1]+integration_radius+1])))
                        if second_order:
                            peaks[0,i] = np.array(tuple(next_peak)+(max_next_peak,np.sum(fft_raw[next_peak[0]-integration_radius:next_peak[0]+integration_radius+1, next_peak[1]-integration_radius:next_peak[1]+integration_radius+1])))
                        else:
                            peaks[i] = np.array(tuple(next_peak)+(max_next_peak,np.sum(fft_raw[next_peak[0]-integration_radius:next_peak[0]+integration_radius+1, next_peak[1]-integration_radius:next_peak[1]+integration_radius+1])))
                
                if second_order:
                    #peaks = (peaks, [])
                    org_pos_tol = position_tolerance
                    position_tolerance = int(np.rint(position_tolerance*np.sqrt(3)))
                    
                    #make sure that areas of first and second_order peaks don't overlap
                    if position_tolerance >= (second_order_peaks-first_order)/np.sqrt(2)-1:
                        position_tolerance = int(np.rint((second_order_peaks-first_order)/np.sqrt(2)-1))

                    for i in range(6):
                        rotation_matrix = np.array( ( (np.cos(i*np.pi/3+np.pi/6), -np.sin(i*np.pi/3+np.pi/6)), (np.sin(i*np.pi/3+np.pi/6), np.cos(i*np.pi/3+np.pi/6)) ) )
                        next_peak = np.rint(np.dot( rotation_matrix , (peaks[0,0,0:2]-center)*(0.213/0.123) ) + center).astype(int)
                        area_next_peak = fft[next_peak[0]-position_tolerance:next_peak[0]+position_tolerance+1, next_peak[1]-position_tolerance:next_peak[1]+position_tolerance+1]
                        max_next_peak = np.amax(area_next_peak)
                        #if  max_next_peak > mean_fft + 4.0*std_dev_fft:#peaks[0][2]/4:
                        if max_next_peak > np.mean(area_next_peak)+4*np.std(area_next_peak):
                            next_peak += np.array( np.unravel_index( np.argmax(area_next_peak), np.shape(area_next_peak) ) ) - position_tolerance
                            #peaks[1].append(tuple(next_peak)+(max_next_peak,np.sum(fft_raw[next_peak[0]-integration_radius:next_peak[0]+integration_radius+1, next_peak[1]-integration_radius:next_peak[1]+integration_radius+1])))
                            peaks[1,i] = np.array(tuple(next_peak)+(max_next_peak,np.sum(fft_raw[next_peak[0]-integration_radius:next_peak[0]+integration_radius+1, next_peak[1]-integration_radius:next_peak[1]+integration_radius+1])))
                    position_tolerance = org_pos_tol
                success = True
            except Exception as detail:
                fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 3
                print(str(detail))
    
    if debug_mode:    
        if second_order:
            for i in range(len(peaks)):
                if i == 1:
                    position_tolerance = int(np.rint(position_tolerance*np.sqrt(3)))
                for coord in peaks[i]:
                    fft[coord[0]-position_tolerance:coord[0]+position_tolerance+1, coord[1]-position_tolerance:coord[1]+position_tolerance+1] *= 4.0
        else:    
            for coord in peaks:
                fft[coord[0]-position_tolerance:coord[0]+position_tolerance+1, coord[1]-position_tolerance:coord[1]+position_tolerance+1] *= 4.0    
        return (peaks, fft)
    else:
        return peaks
        
def draw_circle(image, center, radius, color=-1, thickness=-1):
    
    subarray = image[center[0]-radius:center[0]+radius+1, center[1]-radius:center[1]+radius+1]
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    distances = np.sqrt(x**2+y**2)
    
    if thickness < 0:
        subarray[distances <= radius] = color
    elif thickness == 0:
        subarray[(distances < radius+0.5) * (distances > radius-0.5)] = color
    else:
        subarray[(distances < radius+thickness+1) * (distances > radius-thickness)] = color