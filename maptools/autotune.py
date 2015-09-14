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
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import fftconvolve
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


class Imaging(object):
    
    def __init__(self, **kwargs):
        self._image = kwargs.get('image')
        self._shape = kwargs.get('shape')
        self.imsize = kwargs.get('imsize')
        self._online = kwargs.get('online')
        self.dirt_threshold = kwargs.get('dirt_threshold')
        self._mask = kwargs.get('mask')
        self.frame_parameters = kwargs.get('frame_parameters', {})
        self.record_parameters = kwargs.get('record_parameters')
        self.detectors = kwargs.get('detectors', {'HAADF': False, 'MAADF': True})
        self.aberrations = kwargs.get('aberrations', {})
        self.superscan = kwargs.get('superscan')
        self.as2 = kwargs.get('as2')
        self.delta_graphene = None
        
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, image):
        self._image = image
        self._shape = np.shape(image)
        self._mask = None
        
    @property
    def shape(self):
        if self._shape is None:
            assert self.image is not None, 'No image was found for shape determination. Set Imaging.image first.'
            self._shape = np.shape(self._image)
        return self._shape
    
    @shape.setter
    def shape(self, shape):
        self._shape = shape
        
    @property
    def online(self):
        if self._online is None:
            if self.as2 is not None or self.superscan is not None:
                self._online = True
            else:
                logging.info('Going to offline mode because no instance of as2 or superscan was provided.')
                self._online = False
        return self.online
    
    @online.setter
    def online(self, online):
        self._online = online
    
    @property
    def mask(self):
        if self._mask == None:
            assert self.image is not None, 'No image was found for which a mask could be computed.'
            self._mask = self.dirt_detector()
        return self._mask
    
    @mask.setter
    def mask(self, mask):
        self._mask = mask
        
    def create_record_parameters(self, frame_parameters=None, detectors=None):
        """
        Returns the frame parameters in a form that they can be used in the record and view functions.
        (e.g. superscan.record(**record_parameters), if record_parameters was created by this function.)
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
        if frame_parameters is None:
            frame_parameters = self.frame_parameters
        if detectors is None:
            detectors = self.detectors
            
        if frame_parameters is not None:
            parameters = self.superscan.get_default_frame_parameters()
            
            if frame_parameters.get('size_pixels') is not None:
                parameters['size'] = list(frame_parameters['size_pixels'])   
            if frame_parameters.get('center') is not None:
                parameters['center_nm'] = list(frame_parameters['center']) 
            if frame_parameters.get('pixeltime') is not None:
                parameters['pixel_time_us'] = frame_parameters['pixeltime']  
            if frame_parameters.get('fov') is not None:
                parameters['fov_nm'] = frame_parameters['fov']  
            if frame_parameters.get('rotation') is not None:
                parameters['rotation_rad'] = frame_parameters['rotation']
        else:
            parameters = None
            
        if detectors is not None:
            channels_enabled = [detectors['HAADF'], detectors['MAADF'], False, False]
        else:
            channels_enabled = [False, True, False, False]
            
        #self.record_parameters = {'frame_parameters': parameters, 'channels_enabled': channels_enabled}
        #return self.record_parameters
        return {'frame_parameters': parameters, 'channels_enabled': channels_enabled}
        
    def dirt_detector(self, median_blur_diam=59, gaussian_blur_radius=3, **kwargs):
        """
        Returns a mask with the same shape as "image" that is 1 where there is dirt and 0 otherwise
        """
        # check for optional input arguments that can update instance variables
        if kwargs.get('image') is not None:
            self.image = kwargs.get('image')
        if kwargs.get('dirt_threshold') is not None:
            self.dirt_threshold = kwargs.get('dirt_threshold')
        # if no dirt_threshold is available, find it automatically
        if self.dirt_threshold is None:
            pass
        
        #apply Gaussian Blur to improve dirt detection
        if gaussian_blur_radius > 0:
            self.image = gaussian_filter(self.image, gaussian_blur_radius)
        #create mask
        mask = np.zeros(self.shape)
        mask[self.image>self.dirt_threshold] = 1
        #apply median blur to mask to remove noise influence
        if median_blur_diam % 2==0:
            median_blur_diam+=1

        #self.mask = np.rint(uniform_filter(self.mask, median_blur_diam)).astype('uint8')
        return np.rint(uniform_filter(self.mask, median_blur_diam)).astype('uint8')

    def distribute_intensity(self, x, y):
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

    def find_biggest_clean_spot(self, image):
        pass
        
        
    def find_dirt_threshold(self, **kwargs):
        """
        Returns the correct dirt threshold for an image to use with dirt_detector.
        For possible keyword arguments check function dirt_detector.
        """
        # check for optional input arguments
        if kwargs.pop('debug_mode', False):
            debug_mode = True
        else:
            debug_mode = False
        # check for optional input arguments that can update instance variables
        if kwargs.get('image') is not None:
            self.image = kwargs.pop('image')
        
        # set up the search range
        search_range = np.mgrid[0:2*np.mean(self.image):30j]
        mask_sizes = []
        dirt_start = None
        dirt_end = None
        # go through list of thresholds and determine the amount of dirt with this threshold
        for threshold in search_range:
            mask_size = np.sum(self.dirt_detector(**kwargs)) / np.prod(self.shape)
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
        else:
        # if distance between dirt_start and dirt_end is longer, set threshold to a value 
        # 10% smaller than mean to prevent missing dirt that is actually there in the image
            threshold = (dirt_end + dirt_start) * 0.45
        
        #self.dirt_threshold = threshold
        
        if debug_mode:
            return (self.dirt_threshold, search_range, np.array(mask_sizes))
        else:
            return self.dirt_threshold
            
    def graphene_generator(self, imsize, impix, rotation):
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
                    pixelvalues = self.distribute_intensity(x,y)
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
                    pixelvalues = self.distribute_intensity(x,y)
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
    
    def image_grabber(self, acquire_image=True, **kwargs):
        """
        acquire_image defines if an image is taken and returned or if just the correctors are updated.
        
        kwargs contains all possible values for the correctors : 
            These are all lens aberrations up to threefold astigmatism. If an image is given, the function will simulate
            aberrations to this image and add poisson noise to it. If not, an image with the current frame parameters
            and the corrector parameters given in kwargs is taken.
        
        Possible Parameters
        -------------------
        
        aberrations : dictionary
            e.g. {'EHTFocus': 0, 'C12_a': 0, 'C12_b': 0, 'C21_a': 0, 'C21_b': 0, 'C23_a': 0,  'C23_b': 0} (all in nm)
                    
        image : 
            (as numpy array)
                
        relative_aberrations : True/False
                If 'relative_aberrations' is included and set to True, image_grabber will get the current value for
                each control first and add the given value for the respective aberration to the current value.
                Otherwise, each aberration in kwargs is just set to the value given there.    
                
        reset_aberrations : True/False    
            If 'reset_aberrations' is included and set to True, image_grabber will set each aberration back to its
            original value after acquiring an image. This is a good choice if you want to try new values for the
            aberration correctors bur are not sure you want to keep them.
        
        frame_parameters : dictionary
            Contains the frame parameters for acquisition. See function create_record_parameters() for details.
            
        detectors : dictionary
            Contains the dectectors used for acquisition. See function create_record_parameters() for details.
            
        Example call of image_grabber:
        ------------------------------
        
        result = image_grabber(EHTFocus=1, C12_a=0.5, image=graphene_lattice, imsize=10)
        
        Note that the Poisson noise is added modulatory, e.g. each pixel value is replaced by a random number from a
        Poisson distribution that has the original pixel value as its mean. That means you can control the noise level
        by changing the mean intensity in your image.
        """
        # Check input for additinal parameters that override instance variables
        if kwargs.get('image') is not None:
            self.image = kwargs.get('image')
        if kwargs.get('frame_parameters') is not None:
            self.frame_parameters = kwargs.get('frame_parameters')
        if kwargs.get('detectors') is not None:
            self.detectors = kwargs.get('detectors')
        
        if self.frame_parameters.get('imsize') is not None:
                self.imsize = self.frame_parameters.get('imsize')
        if self.frame_parameters.get('impix') is not None:
                self.shape = self.frame_parameters.get('impix')
            
        # Set parameters for dealing with aberrration settings and apply correct defaults
        relative_aberrations = kwargs.get('relative_aberrations', True)
        reset_aberrations = kwargs.get('reset_aberrations', False)
        # Keys to check for aberrations in aberrations dictionary
        keys = ['EHTFocus', 'C12_a', 'C12_b', 'C21_a', 'C21_b', 'C23_a', 'C23_b']
        return_image = None
           
#                #if aberrations should not be reset, change global_aberrations
#                    if not kwargs.get('reset_aberrations'):
#                        global_aberrations[key] = aberrations[i]
#                else:
#                    aberrations[i] = global_aberrations[keys[i]]
        
#        
#        if relative_aberrations:
#                        self.aberrations[key] += kwargs['aberrations'].get(key, 0)
#                    else:
#                        self.aberrations[key] = kwargs['aberrations'].get(key, 0)
        # Check if all required parameters are there
        if self.online:
            controls = {'EHTFocus': 'EHTFocus', 'C12_a': 'C12.a', 'C12_b': 'C12.b', 'C21_a': 'C21.a',
                        'C21_b': 'C21.b', 'C23_a': 'C23.a', 'C23_b': 'C23.b'}
            originals = {}
            
            if kwargs.get('aberrations') is not None or len(self.aberrations) > 0:
                assert self.as2 is not None, 'You have to provide an instance of as2 to perform as2-related operations.'
            if kwargs.get('aberrations') is not None:
                for key in keys:
                    if kwargs['aberrations'].get(key):
                        if relative_aberrations:
                            self.aberrations[key] = vt.as2_get_control(self.as2, controls[key]) + \
                                                    kwargs['aberrations'].get(key)
                        else:
                            self.aberrations[key] = kwargs['aberrations'].get(key)
                        
                        if reset_aberrations:
                            originals[controls[key]] = vt.as2_get_control(self.as2, controls[key])
            # Apply corrector values to the Hardware
            for key in self.aberrations.keys():
                vt.as2_set_control(controls[key], self.aberrations[key])
            
            if acquire_image:
                assert self.superscan is not None, \
                       'You have to provide an instance of superscan to perform superscan-related operations.'
                self.record_parameters = self.create_record_parameters(self.superscan, self.frame_parameters, self.detectors)
                im = self.superscan.record(**self.record_parameters)
                if len(im) > 1:
                    return_image = []
                    for entry in im:
                        return_image.append(entry.data)
                else:
                    return_image = im[0].data
            # reset all corrector values to the original ones
            for key in originals.keys():
                vt.as2_set_control(key, originals[key])
        
        # e.g. offline mode
        else:
            global global_aberrations
            
            assert self.imsize is not None, \
                   'You have to input the size (in nm) for the generated image in order to use the offline mode.'
            
            if self.delta_graphene is None:
                assert self.shape is not None, \
                       'You have to input the shape for the generated image in order to use the offline mode.'
                self.delta_graphene = self.graphene_generator()
                
            # Update aberrations dictionary with the values passed to this function
            if kwargs.get('aberrations') is not None:
                for key in keys:
                    if kwargs['aberrations'].get(key):
                        # Relative aberrations is here relative to global_aberrations, in online mode its relative to
                        # the values already set in as2
                        if relative_aberrations:
                            self.aberrations[key] = global_aberrations[key] + kwargs['aberrations'].get(key)
                        else:
                            self.aberrations[key] = kwargs['aberrations'].get(key)
            # Write current values to global aberrations if they should be kept (which is similar to applying them to
            # the hardware in online mode)
            if not reset_aberrations:
                for key in self.aberrations.keys():
                    global_aberrations[key] = self.aberrations[key]

            if acquire_image:
                # Create x and y coordinates such that resulting beam has the same scale as the image.
                # The size of the kernel which is used for image convolution is chosen to be "1/kernelsize"
                # of the image size (in pixels)
                kernelsize = 2
                kernelpixel = int(self.shape[0]/kernelsize)
                frequencies = np.matrix(np.fft.fftshift(np.fft.fftfreq(kernelpixel, self.imsize/self.shape[0])))
                x = np.array(np.tile(frequencies, np.size(frequencies)).reshape((kernelpixel,kernelpixel)))
                y = np.array(np.tile(frequencies.T, np.size(frequencies)).reshape((kernelpixel,kernelpixel)))
                
                # compute aberration function up to threefold astigmatism
                # formula taken from "Advanced Computing in Electron Microscopy",
                # Earl J. Kirkland, 2nd edition, 2010, p. 18
                # wavelength for 60 keV electrons: 4.87e-3 nm
                raw_kernel = (
                              (-self.aberrations.get('EHTFocus', 0) * (x**2 + y**2) +       
                              
                               np.sqrt(self.aberrations.get('C12_a', 0)**2 + self.aberrations.get('C12_b', 0)**2) * 
                               (x**2 + y**2) * np.cos(2 * (np.arctan2(y,x) -
                                                      np.arctan2(self.aberrations.get('C12_b', 0),
                                                                 self.aberrations.get('C12_a', 0)))) +
                               (2.0/3.0) *
                               np.sqrt(self.aberrations.get('C21_a', 0)**2 + self.aberrations.get('C21_b', 0)**2) *
                               4.87e-3 *
                               np.sqrt(x**2 + y**2)**3 * np.cos(np.arctan2(y,x) -
                                                                np.arctan2(self.aberrations.get('C21_b', 0),
                                                                           self.aberrations.get('C21_a', 0)[3])) + 
                               (2.0/3.0) *
                               np.sqrt(self.aberrations.get('C23_a', 0)**2 + self.aberrations.get('C23_a', 0)**2) *
                               4.87e-3 * 
                               np.sqrt(x**2 + y**2)**3 * np.cos(3 * (np.arctan2(y,x) - 
                                                                np.arctan2(self.aberrations.get('C23_b', 0),
                                                                           self.aberrations.get('C23_b', 0))))) * 
                               np.pi * 4.87e-3
                              )
                    
                kernel = np.cos(raw_kernel)+1j*np.sin(raw_kernel)
                aperture = np.zeros(kernel.shape)
                # Calculate size of 25 mrad aperture in k-space for 60 keV electrons
                aperturesize = (0.025/kernelsize)*self.imsize/4.87e-3
                # "Apply" aperture
                draw_circle(aperture, tuple((np.array(kernel.shape)/2).astype('int')),
                            int(np.rint(aperturesize)), color=1)
                
                kernel *= aperture
                kernel = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(kernel))))**2
                kernel /= np.sum(kernel)
                #im = cv2.filter2D(im, -1, kernel)
                im = fftconvolve(self.delta_graphene, kernel, mode='same')
                im = np.random.poisson(lam=im.flatten(), size=np.size(im)).astype(im.dtype)
                
                return im.reshape(self.shape).astype('float32')
                #return kernel    
            
            
class Peaking(Imaging):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fft = kwargs.get('fft')
        self.peaks = kwargs.get('peaks')
        self._center = kwargs.get('center')
        self.integration_radius = kwargs.get('integration_radius')
    
    @Imaging.image.setter
    def image(self, image):
        self._image = image
        self._shape = np.shape(image)
        self._center = tuple((np.array(np.shape(image))/2).astype(np.int))
        self.fft = None
        self._mask = None
        
    @property
    def center(self):
        if self._center is None:
            self._center = tuple((np.array(self.shape)/2).astype(np.int))
        return self._center
    
    @center.setter
    def center(self, center):
        self._center = center
    
    @property
    def fft(self):
        if self._fft == None:
            assert self.image is not None, 'Can not calculate the fft because no image is given.'
            self._fft = np.fft.fftshift(np.fft.fft2(self.image))
    
    def find_peaks(self, half_line_thickness=5, position_tolerance=5, second_order=False, debug_mode=False, **kwargs):
        """
            This function can find the 6 first-order peaks in the FFT of an atomic-resolution image of graphene.
            Input:
                    im: Image as a numpy array or any type that can be simply casted to a numpy array.
                    imsize: Size of the input image in nm.

            Output:
                    List of tuples that contain the coordinates of the reflections. The tuples have the form 
                    (y, x, intensity_of_peak_maximum)
                    If no peaks were found the return value will be None.
                    Note that the returned intesities might be smaller than that of the raw fft because of the
                    processing done in the function.
        """
        # Check kwargs for entrys that override class variables
        if kwargs.get('image') is not None:
            self.image = kwargs['image']
        if kwargs.get('imsize') is not None:
            self.imsize = kwargs['imsize']

        fft = np.abs(self.fft)
        fft_raw = fft.copy()
        
        first_order = self.imsize/0.213
        second_order_peaks = self.imsize/0.123
        
        # make sure that areas of first and second_order peaks don't overlap
        if position_tolerance > (second_order_peaks-first_order)/np.sqrt(2)-1:
            position_tolerance = int(np.rint((second_order_peaks-first_order)/np.sqrt(2)-1))
        
        # blank out bright spot in center of fft
        draw_circle(fft, self.center, int(np.rint(first_order/2.0)))
        
        # prevent infinite values when cross would be calculated until central pixel because of too high half
        # line thickness
        if half_line_thickness > int(np.rint(first_order/2.0))-1:
            half_line_thickness = int(np.rint(first_order/2.0))-1
    
        mean_fft = np.mean(fft[fft>-1])
        # Fit horizontal and vertical lines with hyperbola
        cross = np.zeros(self.shape)
        for i in range(-half_line_thickness, half_line_thickness+1):
            horizontal = fft[self.center[0]+i,:]
            vertical = fft[:, self.center[1]+i]
            xdata = np.mgrid[:self.shape[1]][horizontal>-1] - self.center[1]
            ydata = np.mgrid[:self.shape[0]][vertical>-1] - self.center[0]
            horizontal = horizontal[horizontal>-1]
            vertical = vertical[vertical>-1]
            horiz_a = 1.0 / ((np.mean(horizontal[int(len(horizontal) * 0.6) - 3 : int(len(horizontal) * 0.6) + 4]) - 
                              np.mean(horizontal[int(len(horizontal) * 0.7) - 3 : int(len(horizontal) * 0.7) + 4])) * 
                              2.0 * xdata[int(len(horizontal) * 0.6)])
            vert_a = 1.0 / ((np.mean(vertical[int(len(vertical) * 0.6) - 3 : int(len(vertical) * 0.6) + 4]) -
                             np.mean(vertical[int(len(vertical) * 0.7) - 3 : int(len(vertical) * 0.7) + 4])) *
                             2.0 * ydata[int(len(vertical) * 0.6)])
            horizontal_popt, horizontal_pcov = scipy.optimize.curve_fit(hyperbola1D, xdata[:len(xdata)/2],
                                                                        horizontal[:len(xdata)/2], p0=(horiz_a, 0))
            vertical_popt, vertical_pcov = scipy.optimize.curve_fit(hyperbola1D, ydata[:len(ydata)/2],
                                                                    vertical[:len(ydata)/2], p0=(vert_a, 0))
            
            cross[self.center[0] + i, xdata + self.center[1]] = hyperbola1D(xdata, *horizontal_popt) - 1.5 * mean_fft
            cross[ydata + self.center[0], self.center[1] + i] = hyperbola1D(ydata, *vertical_popt) - 1.5 * mean_fft
        
        fft-=cross
        
        if (4*int(first_order) < self.center).all():
            fft[self.center[0]-4*int(first_order):self.center[0]+4*int(first_order)+1,
                self.center[1]-4*int(first_order):self.center[1]+4*int(first_order)+1] *= \
            gaussian2D(np.mgrid[self.center[0]-4*int(first_order):self.center[0]+4*int(first_order)+1,
                                self.center[1]-4*int(first_order):self.center[1]+4*int(first_order)+1],
                       self.shape[1]/2, self.shape[0]/2, 0.75*first_order, 0.75*first_order, -1, 1)
        else:
            fft *= gaussian2D(np.mgrid[:self.shape[0], :self.shape[1]], self.shape[1]/2, self.shape[0]/2,
                              0.75*first_order, 0.75*first_order, -1, 1)
        #find peaks
        success = False
        counter = 0
        
        while success is False:
            counter += 1
            if counter > np.sqrt(self.shape[0]):
                raise RuntimeError('No peaks could be found in the FFT of im.')
                #break
            if second_order:
                peaks = np.zeros((2,6,4))
            else:
                peaks = np.zeros((6,4))
                
            first_peak = np.unravel_index(np.argmax(fft), self.shape)+(np.amax(fft), )
            area_first_peak = fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1,
                                  first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1]
    
            if first_peak[2] < np.mean(area_first_peak)+6*np.std(area_first_peak):
                fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1,
                    first_peak[1]-position_tolerance:first_peak[1]+position_tolerance+1] = 1
            elif np.sqrt(np.sum((np.array(first_peak[0:2])-self.center)**2)) < first_order * 0.6667 or \
                 np.sqrt(np.sum((np.array(first_peak[0:2])-self.center)**2)) > first_order * 1.5:
                fft[first_peak[0]-position_tolerance:first_peak[0]+position_tolerance+1, first_peak[1] - 
                    position_tolerance:first_peak[1]+position_tolerance+1] = 2
            else:
                try:            
                    if second_order:
                        peaks[0,0] = np.array(first_peak + (np.sum(fft_raw[first_peak[0] - self.integration_radius:
                                              first_peak[0] + self.integration_radius + 1, first_peak[1] -
                                              self.integration_radius:first_peak[1] + self.integration_radius + 1]),))
                    else:
                        peaks[0] = np.array(first_peak + (np.sum(fft_raw[first_peak[0] - self.integration_radius:
                                            first_peak[0] + self.integration_radius + 1, first_peak[1] -
                                            self.integration_radius:first_peak[1] + self.integration_radius + 1]),))
                    
                    for i in range(1,6):
                        rotation_matrix = np.array( ( (np.cos(i*np.pi/3), -np.sin(i*np.pi/3)), (np.sin(i*np.pi/3),
                                                       np.cos(i*np.pi/3)) ) )
                        if second_order:
                            next_peak = np.rint(np.dot( rotation_matrix , peaks[0,0,0:2] - self.center ) +
                                                self.center).astype(int)
                        else:
                            next_peak = np.rint(np.dot( rotation_matrix , peaks[0,0:2] - self.center ) +
                                                self.center).astype(int)
                        area_next_peak = fft[next_peak[0] - position_tolerance:next_peak[0] + position_tolerance+1,
                                             next_peak[1] - position_tolerance:next_peak[1] + position_tolerance+1]
                        max_next_peak = np.amax(area_next_peak)

                        if max_next_peak > np.mean(area_next_peak)+5*np.std(area_next_peak):
                            next_peak += np.array(np.unravel_index(np.argmax(area_next_peak),
                                                                   np.shape(area_next_peak))) - position_tolerance
                            if second_order:
                                peaks[0,i] = np.array(tuple(next_peak) + 
                                                      (max_next_peak,np.sum(fft_raw[next_peak[0] -
                                                      self.integration_radius:next_peak[0]+self.integration_radius+1,
                                                      next_peak[1] - self.integration_radius:next_peak[1] +
                                                      self.integration_radius+1])))
                            else:
                                peaks[i] = np.array(tuple(next_peak) + 
                                                    (max_next_peak,np.sum(fft_raw[next_peak[0] -
                                                    self.integration_radius:next_peak[0] + self.integration_radius + 1,
                                                    next_peak[1] - self.integration_radius:next_peak[1] +
                                                    self.integration_radius + 1])))
                    
                    if second_order:
                        #peaks = (peaks, [])
                        org_pos_tol = position_tolerance
                        position_tolerance = int(np.rint(position_tolerance*np.sqrt(3)))
                        
                        #make sure that areas of first and second_order peaks don't overlap
                        if position_tolerance >= (second_order_peaks-first_order)/np.sqrt(2)-1:
                            position_tolerance = int(np.rint((second_order_peaks-first_order)/np.sqrt(2)-1))
    
                        for i in range(6):
                            rotation_matrix = np.array(((np.cos(i*np.pi/3+np.pi/6), -np.sin(i*np.pi/3+np.pi/6)),
                                                        (np.sin(i*np.pi/3+np.pi/6), np.cos(i*np.pi/3+np.pi/6))))
                            next_peak = np.rint(np.dot(rotation_matrix , (peaks[0,0,0:2]-self.center)*(0.213/0.123)) +
                                                self.center).astype(int)
                            area_next_peak = fft[next_peak[0]-position_tolerance:next_peak[0]+position_tolerance+1,
                                                 next_peak[1]-position_tolerance:next_peak[1]+position_tolerance+1]
                            max_next_peak = np.amax(area_next_peak)
                            #if  max_next_peak > mean_fft + 4.0*std_dev_fft:#peaks[0][2]/4:
                            if max_next_peak > np.mean(area_next_peak)+4*np.std(area_next_peak):
                                next_peak += np.array(np.unravel_index(np.argmax(area_next_peak),
                                                                       np.shape(area_next_peak))) - position_tolerance
                                peaks[1,i] = np.array(tuple(next_peak) +
                                                      (max_next_peak,
                                                       np.sum(fft_raw[next_peak[0] - self.integration_radius:
                                                              next_peak[0] + self.integration_radius + 1,
                                                              next_peak[1] - self.integration_radius:next_peak[1] +
                                                              self.integration_radius+1])))
                        position_tolerance = org_pos_tol
                    success = True
                except Exception as detail:
                    fft[first_peak[0] - position_tolerance:first_peak[0] + position_tolerance+1,
                        first_peak[1] - position_tolerance:first_peak[1]+position_tolerance+1] = 3
                    print(str(detail))
        
        if debug_mode:    
            if second_order:
                for i in range(len(peaks)):
                    if i == 1:
                        position_tolerance = int(np.rint(position_tolerance * np.sqrt(3)))
                    for coord in peaks[i]:
                        fft[coord[0]-position_tolerance:coord[0]+position_tolerance+1,
                            coord[1]-position_tolerance:coord[1]+position_tolerance+1] *= 4.0
            else:    
                for coord in peaks:
                    fft[coord[0]-position_tolerance:coord[0]+position_tolerance+1,
                        coord[1]-position_tolerance:coord[1]+position_tolerance+1] *= 4.0    
            return (peaks, fft)
        else:
            return peaks
        
    def fourier_filter(self, filter_radius=7, **kwargs):
        peaks = self.find_peaks(second_order=True, half_line_thickness=3, **kwargs)
        xdata = np.mgrid[-filter_radius:filter_radius+1, -filter_radius:filter_radius+1]
        mask = gaussian2D(xdata, 0, 0, filter_radius/2, filter_radius/2, 1, 0)
        maskradius = int(np.shape(mask)[0]/2)
        fft_masked = np.zeros(self.shape, dtype=self.fft.dtype)
        for order in peaks:
            for peak in order:
                if np.count_nonzero(peak) > 0:
                    fft_masked[peak[0]-maskradius:peak[0]+maskradius+1, peak[1]-maskradius:peak[1]+maskradius+1] += \
                    self.fft[peak[0]-maskradius:peak[0]+maskradius+1, peak[1]-maskradius:peak[1]+maskradius+1]*mask    
        
        return np.real(np.fft.ifft2(np.fft.fftshift(fft_masked)))

            
class Tuning(Peaking):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = kwargs.get('steps')
        self.keys = kwargs.get('keys')
        self.event = kwargs.get('event')
        self.document_controller = kwargs.get('document_controller')
        self.save_images = kwargs.get('save_images', False)
        self.savepath = kwargs.get('savepath')
        self.average_frames = kwargs.get('average_frames')
        self.aberrations_tracklist = []
        
    def tuning_merit(imsize, average_frames, integration_radius, save_images, savepath, dirt_threshold, kwargs):
        intensities, image, mask = check_tuning(imsize, average_frames=average_frames, integration_radius=integration_radius, \
                            save_images=save_images, savepath=savepath, return_image = True, dirt_threshold=dirt_threshold, **kwargs)
        
        symmetry = symmetry_merit(image, imsize, mask=mask)
        
        print('sum intensities: ' + str(np.sum(intensities)) + '\tvar intensities: ' + str(np.std(intensities)/np.sum(intensities)) + '\tsymmetry: ' + str(symmetry))
        #return 1.0/(np.sum(intensities/1e6) + np.sum(symmetry) + np.count_nonzero(intensities)/10.0)
        return 1.0/(np.sum(intensities)/1e6 + np.sum(symmetry))
        
    
    def kill_aberrations(superscan=None, as2=None, document_controller=None, average_frames=3, integration_radius=1, image=None, \
                        imsize=None, only_focus=False, save_images=False, savepath=None, event=None, dirt_threshold=None, \
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
        
        # If no dirt threshold was provided and in online mode, find the correct dirt threshold
        if online and dirt_threshold == None:
            image = image_grabber(**kwargs)
            dirt_threshold = find_dirt_threshold(image)
        
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
    
    
    
    
    def check_tuning(self, imagesize, im=None, check_astig=False, average_frames=0, integration_radius=0, save_images=False, savepath=None, \
                    process_image=True, return_image = False, dirt_threshold = 0.015, **kwargs):
    
        global global_aberrations                    
                        
        if kwargs.get('imsize') is None:
            kwargs['imsize'] = imagesize
        else:
            imagesize=kwargs['imsize']
            
        if (process_image or im is None) and average_frames < 2:
            if im is not None and kwargs.get('image') is None:
                kwargs['image'] = im
            im = self.image_grabber(**kwargs)
                
        if average_frames > 1:
            im = []
            #Acquire only one image in the first place to avoid "stacking" of aberations
            single_image = self.image_grabber(**kwargs)
            im.append(single_image)
            
            #remove aberrations from kwargs fore next frames
            kwargs2 = kwargs.copy()
            if kwargs.get('relative_aberrations'):
                    if not kwargs.get('reset_aberrations'):
                        keys = ['EHTFocus', 'C12_a', 'C12_b', 'C21_a', 'C21_b', 'C23_a', 'C23_b']
                        for key in keys:
                            kwargs2.pop(key, 0)
            
            for i in range(average_frames-1):
                im.append(self.image_grabber(**kwargs2))
                
    #Apply dirt detection, but only when real images are used
        #Averaging is only done with real images
        mask = None
        if average_frames > 1:
            for image in im:
                mask = self.dirt_detector(image, threshold=0.015)
                if np.sum(mask) > 0.5*np.prod(np.array(np.shape(image))):
                    raise DirtError('Cannot check tuning of images with more than 50% dirt coverage.')
        #If no image is provided or just a tuning check without real or simulated acquisition should be done it is real data
        elif kwargs.get('image') is None or not process_image:
            mask = self.dirt_detector(im, threshold=0.015)
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
            peaks_first, peaks_second = self.find_peaks(im, imagesize, integration_radius=integration_radius, second_order=True, position_tolerance=9)
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
            
    def measure_symmetry(image, imsize):
        point_mirrored = np.flipud(np.fliplr(image)) 
        return autoalign.find_shift(image, point_mirrored, ratio=0.142/imsize/2)
        
    def symmetry_merit(self, image, imsize, mask=None):
        if mask is None:
            mean = np.mean(image)
        else:
            mean = np.mean(image[mask==0])
            
        ffil = self.fourier_filter(image, imsize)
        
        if mask is None:
            return (self.measure_symmetry(ffil, imsize)[1], np.var(ffil)/mean**2*100)
        else:
            return (self.measure_symmetry(ffil, imsize)[1]*(1.0-np.sum(mask)/np.size(mask)), np.var(ffil[mask==0]/mean**2*50))

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
        
def gaussian2D(xdata, x0, y0, x_std, y_std, amplitude, offset):
    x0, y0, x_std, y_std, amplitude, offset = float(x0), float(y0), float(x_std), float(y_std), float(amplitude), float(offset)
    return (amplitude*np.exp( -0.5*( ((xdata[1]-x0)/x_std)**2 + ((xdata[0]-y0)/y_std)**2 ) ) + offset)#.ravel()

def hyperbola1D(xdata, a, offset):
    a, offset = float(a), float(offset)
    return np.abs(1.0/(a*xdata))+offset
    
def positive_angle(angle):
    """
    Calculates the angle between 0 and 2pi from an input angle between -pi and pi (all angles in rad)
    """
    if angle < 0:
        return angle  + 2*np.pi
    else:
        return angle
    
        
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