# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:54:54 2015

@author: mittelberger
"""

import logging
import time
#import os
#import warnings
#from threading import Event

import numpy as np
import scipy.optimize
from scipy.ndimage import gaussian_filter, uniform_filter, distance_transform_cdt
from scipy.signal import fftconvolve
#import cv2
#try:
#    import cv2
#except:
#    logging.warn('Could not import opencv')
#import matplotlib as plt

#try:
#with warnings.catch_warnings():
#    warnings.simplefilter('ignore')
#    #import ViennaTools.ViennaTools as vt
#    from . import tifffile
#except:
#    try:
#        import ViennaTools as vt
#        from ViennaTools import tifffile
#    except:
#        logging.warn('Could not import Vienna tools!')

#try:
from . import autoalign
#except:
#    try:
#        import autoalign
#    except:
#        pass

#try:
#    from superscan import SuperScanPy as ss
#except:
#    logging.warn('Could not import SuperScanPy. Maybe you are running in offline mode.')
#global variable to store aberrations when simulating them (see function image_grabber() for details)
#global_aberrations = {'EHTFocus': 0, 'C12_a': 5, 'C12_b': 0, 'C21_a': 801.0, 'C21_b': 0, 'C23_a': -500, 'C23_b': 0}
global_aberrations = {'EHTFocus': 0, 'C12_a': 0, 'C12_b': 0, 'C21_a': 0, 'C21_b': 0, 'C23_a': 0, 'C23_b': 0}



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
        self._frame_parameters = kwargs.get('frame_parameters', {})
        self.detectors = kwargs.get('detectors', {'HAADF': False, 'MAADF': True})
        self.aberrations = kwargs.get('aberrations', {})
        self.superscan = kwargs.get('superscan')
        self.as2 = kwargs.get('as2')
        self.document_controller = kwargs.get('document_controller')
        self.delta_graphene = None
        self.live_data_item_MAADF = None
        self.live_data_item_HAADF = None

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
        return self._online

    @online.setter
    def online(self, online):
        self._online = online

    @property
    def mask(self):
        if self._mask is None:
            assert self.image is not None, 'No image was found for which a mask could be computed.'
            self._mask = self.dirt_detector()
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    @property
    def frame_parameters(self):
        return self._frame_parameters

    @frame_parameters.setter
    def frame_parameters(self, frame_parameters):
        for key in frame_parameters.keys():
            self._frame_parameters[key] = frame_parameters[key]
        self.delta_graphene = None

    @frame_parameters.deleter
    def frame_parameters(self):
        self._frame_parameters = {}

    def create_record_parameters(self, frame_parameters=None, detectors=None):
        """
        Returns the frame parameters in a form that they can be used in the record and view functions.
        (e.g. superscan.record(**record_parameters), if record_parameters was created by this function.)
        Parameters
        -----------

        frame_parameters : dictionary
            Frame parameters to set in the microscope. Possible keys are:

            - size_pixels: Number of pixels in x- and y-direction of the acquired frame as tuple (x,y)
            - pixeltime: Time per pixel (us)
            - fov: Field-of-view of the acquired frame (nm)
            - rotation: Scan rotation (deg)
            - center: Center of the scan (y, x, nm)

        detectors : optional, dictionary
            If None (default), the detector settings saved in this object are used.
            Dictionary has to be in the form:
            {'HAADF': False, 'MAADF': True}

        Returns
        --------
        record_parameters : dictionary
            It has the form: {'frame_parameters': frame_parameters, 'channels_enabled': [HAADF, MAADF, False, False]}
        """
        if frame_parameters is None:
            frame_parameters = self.frame_parameters

        parameters = self.superscan.get_record_frame_parameters()

        if frame_parameters is not None:
            if frame_parameters.get('size_pixels') is not None:
                parameters['size'] = tuple(frame_parameters['size_pixels'])
            if frame_parameters.get('pixeltime') is not None:
                parameters['pixel_time_us'] = frame_parameters['pixeltime']
            if frame_parameters.get('fov') is not None:
                parameters['fov_nm'] = frame_parameters['fov']
            if frame_parameters.get('rotation') is not None:
                parameters['rotation_rad'] = frame_parameters['rotation']/180.0*np.pi
            if frame_parameters.get('center') is not None:
                parameters['center_nm'] = frame_parameters['center']

        detector_list = [False, False, False, False]

        if detectors is None:
            detectors = self.detectors

        if 'HAADF' in detectors:
            detector_list[0] = detectors['HAADF']
        if 'MAADF' in detectors:
            detector_list[1] = detectors['MAADF']

        if not True in detector_list:
            detector_list = None

        return {'frame_parameters': parameters, 'channels_enabled': detector_list}

    def dirt_detector(self, median_blur_diam=59, gaussian_blur_radius=3, **kwargs):
        """
        Returns a mask with the same shape as "image" that is 1 where there is dirt and 0 otherwise
        Possible keyword arguments are (all optional):
            - image : ndarray
                image data where dirt should be found. If not given, the image stored in the instance variable of
                Imaging class is used.
            - dirt_threshold : float
                threshold which lies in between the brightness of the dirt and the brightness of the underlying
                graphene. If not given, the number stored in the instance variable of Imaging class is used.
                If this is also not set, the threshold is determined automatically.
        """
        # check for optional input arguments that can update instance variables
        if kwargs.get('image') is not None:
            self.image = kwargs.get('image')
        if kwargs.get('dirt_threshold') is not None:
            self.dirt_threshold = kwargs.get('dirt_threshold')
        # if no dirt_threshold is available, find it automatically
        if self.dirt_threshold is None:
            self.dirt_threshold = self.find_dirt_threshold()

        #apply Gaussian Blur to improve dirt detection
        if gaussian_blur_radius > 0:
            image = gaussian_filter(self.image, gaussian_blur_radius)
        #create mask
        mask = np.zeros(self.shape)
        mask[image > self.dirt_threshold] = 1
        #apply median blur to mask to remove noise influence
        if median_blur_diam % 2 == 0:
            median_blur_diam += 1

        #self.mask = np.rint(uniform_filter(self.mask, median_blur_diam)).astype('uint8')
        return np.rint(uniform_filter(mask, median_blur_diam)).astype('uint8')

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

    def find_biggest_clean_spot(self, **kwargs):
        """
        Applies a distance transform to the dirt mask of an image and returns the position and height of the maximum
        in the distance transform. All keyword arguments are directly passed to dirt_detector.
        """
        if self.mask is None:
            self.mask = self.dirt_detector(**kwargs)

        dist_mask = distance_transform_cdt(self.mask*-1+1)

        biggest_spot = np.unravel_index(np.argmax(dist_mask), np.shape(self.mask))
        max_distance = np.amax(dist_mask)

        return (np.array(biggest_spot), max_distance)

    def find_clean_spots(self, size=3, overlap=0.1, debug_mode=False, **kwargs):
        """
        Finds clean spots of the given size in an image. For this to work an image with a bigger FOV has to be there
        as an instance variable or passed to the function (remember to also set or pass the FOV or correct frame
        parameters).
        """
        if kwargs.get('imsize') is not None:
            self.imsize = kwargs.pop('imsize')
        if kwargs.get('frame_parameters') is not None:
            self.frame_parameters = kwargs.pop('frame_parameters')
        if kwargs.get('image') is not None:
            self.image = kwargs.pop('image')
        if kwargs.get('mask') is not None:
            self.mask = kwargs.pop('mask')
        if self.mask is None:
            self.dirt_detector(**kwargs)
        if self.imsize is None and self.frame_parameters.get('fov'):
            self.imsize = self.frame_parameters['fov']

        if self.imsize is None or self.image is None:
            raise RuntimeError('An image and its size has to be there in order to find clean spots.')
        if self.imsize < size:
            raise RuntimeError('Can not find clean spots that are larger than the image size.')

        counter = 0
        mask = self.mask.copy()
        size_pixels = size/self.imsize*self.shape[0]
        radius_pixels = size_pixels/2
        radius_overlap = radius_pixels * (1-overlap)
        size_pixels = int(np.rint(size_pixels))
        radius_pixels = int(np.rint(radius_pixels))
        radius_overlap = int(np.rint(radius_overlap))
        radius_left = radius_right = radius_top = radius_bottom = radius_overlap
        clean_spots = []
        dist = distance_transform_cdt(mask*-1+1)
        dist[:radius_pixels] = 0
        dist[-radius_pixels:] = 0
        dist[:,:radius_pixels] = 0
        dist[:, -radius_pixels:] = 0
        while counter < 100:
            counter +=1
            maxi = np.unravel_index(np.argmax(dist), dist.shape)
            radius_left = np.amin((maxi[1], 2*radius_overlap))
            radius_right = np.amin((dist.shape[1]-maxi[1], 2*radius_overlap))
            radius_top = np.amin((maxi[0], 2*radius_overlap))
            radius_bottom = np.amin((dist.shape[0]-maxi[0], 2*radius_overlap))
            if dist[maxi] > radius_overlap:
                clean_spots.append(np.array(maxi))

                dist[maxi[0]-radius_top:maxi[0]+radius_bottom,
                     maxi[1]-radius_left:maxi[1]+radius_right] = 0
            else:
                self.logwrite('Finished because all distances are too small')
                break
        else:
            self.logwrite('Finished because of maximium number of iterations was exceeded')

        if debug_mode:
            for clean_spot in clean_spots:
                mask[clean_spot[0]-radius_pixels:clean_spot[0]+radius_pixels,
                     clean_spot[1]-radius_pixels:clean_spot[1]+radius_pixels] += 2
            return (clean_spots, mask)
        else:
            return clean_spots

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
        search_range = np.mgrid[0:np.mean(self.image):30j]
        mask_sizes = []
        dirt_start = None
        dirt_end = None
        while dirt_end is None :
            search_range *= 2
        # go through list of thresholds and determine the amount of dirt with this threshold
            for threshold in search_range:
                mask_size = np.sum(self.dirt_detector(dirt_threshold = threshold, **kwargs)) / np.prod(self.shape)
                # remember value where the mask started to shrink
                if mask_size < 0.99 and dirt_start is None:
                    dirt_start = threshold
                # remember value where the mask is almost zero and end search
                if mask_size < 0.01:
                    dirt_end = threshold
                    break

                mask_sizes.append(mask_size)

        # determine if there was really dirt present and return an appropriate threshold
        if dirt_end-dirt_start < 3*search_range[1]:
        # if distance between maximum and minimum mask size is very small, no dirt is present
        # set threshold to a value 25% over dirt_end
            threshold = dirt_end * 1.25
        else:
        # if distance between dirt_start and dirt_end is longer, set threshold to a value
        # 16% smaller than mean to prevent missing dirt that is actually there in the image
            threshold = (dirt_end + dirt_start) * 0.42

        #self.dirt_threshold = threshold

        if debug_mode:
            return (threshold, search_range, np.array(mask_sizes), dirt_start, dirt_end)
        else:
            return threshold

    def graphene_generator(self, imsize, impix, rotation):
        rotation = rotation*np.pi/180

        #increase size of initially generated image by 20% to avoid missing atoms at the edges (image will be cropped
        #to actual size before returning it)
        image = np.zeros((int(impix*1.2), int(impix*1.2)))
        rotation_matrix = np.array( ( (np.cos(2.0/3.0*np.pi), np.sin(2.0/3.0*np.pi)), (-np.sin(2.0/3.0*np.pi),
                                       np.cos(2.0/3.0*np.pi)) ) )
        #define basis vectors of unit cell, a1 and a2
        basis_length = 0.142 * np.sqrt(3) * impix/float(imsize)
        a1 = np.array((np.cos(rotation), np.sin(rotation))) * basis_length
        a2 = np.dot(a1, rotation_matrix)
        #print(a1)
        #print(a2)
        a1position = np.array((0.0, 0.0))
        a2position = np.array((0.0, 0.0))
        a2direction = 1.0


        while (a1position < impix*2.4).all():
            success = True

            while success:
                firsta2 = a2position.copy()
                cellposition = a1position + a2position
                #print(str(a1position) + ', '  + str(a2position))
                #print(cellposition)

                #place atoms
                if (cellposition+a1/3.0+a2*(2.0/3.0) < impix*1.2).all() and \
                (cellposition+a1/3.0+a2*(2.0/3.0) >= 0).all():
                    success = True
                    y, x = cellposition + a1/3.0 + a2*(2.0/3.0)
                    pixelvalues = self.distribute_intensity(x, y)
                    pixelpositions = [(0, 0), (0, 1), (1, 1), (1, 0)]

                    for i in range(len(pixelvalues)):
                        try:
                            image[np.floor(y)+pixelpositions[i][0], np.floor(x)+pixelpositions[i][1]] = pixelvalues[i]
                        except IndexError:
                            pass
                else:
                    success = False

                if (cellposition+a2/3.0+a1*(2.0/3.0) < impix*1.2).all() and \
                   (cellposition+a2/3.0+a1*(2.0/3.0) >= 0).all():
                    success = True
                    y, x = cellposition + a2/3.0 + a1*(2.0/3.0)
                    pixelvalues = self.distribute_intensity(x, y)
                    pixelpositions = [(0, 0), (0, 1), (1, 1), (1, 0)]

                    for i in range(len(pixelvalues)):
                        try:
                            image[np.floor(y) + pixelpositions[i][0], np.floor(x) +
                                  pixelpositions[i][1]] = pixelvalues[i]
                        except IndexError:
                            pass
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

    def image_grabber(self, acquire_image=True, debug_mode=False, show_live_image=False, **kwargs):
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
        if kwargs.get('delta_graphene') is not None:
            self.delta_graphene = kwargs.get('delta_graphene')
        if kwargs.get('frame_parameters') is not None:
            self.frame_parameters = kwargs.get('frame_parameters')
        if kwargs.get('detectors') is not None:
            self.detectors = kwargs.get('detectors')
        if kwargs.get('imsize') is not None:
            self.imsize = kwargs['imsize']
        if kwargs.get('shape') is not None:
            self.shape = kwargs['shape']
        if kwargs.get('impix') is not None:
            self.shape = (kwargs['impix'], kwargs['impix'])

        if self.frame_parameters.get('fov') is not None:
            self.imsize = self.frame_parameters.get('fov')
        if self.frame_parameters.get('size_pixels') is not None:
            self.shape = self.frame_parameters.get('size_pixels')

        # Set parameters for dealing with aberrration settings and apply correct defaults
        relative_aberrations = kwargs.get('relative_aberrations', True)
        reset_aberrations = kwargs.get('reset_aberrations', False)
        # Keys to check for aberrations in aberrations dictionary
        keys = ['EHTFocus', 'C12_a', 'C12_b', 'C21_a', 'C21_b', 'C23_a', 'C23_b']
        return_image = None
        # Check if all required parameters are there
        if self.online:
            controls = {'EHTFocus': 'EHTFocus', 'C12_a': 'C12.a', 'C12_b': 'C12.b', 'C21_a': 'C21.a',
                        'C21_b': 'C21.b', 'C23_a': 'C23.a', 'C23_b': 'C23.b'}
            originals = {}

            if kwargs.get('aberrations') is not None or len(self.aberrations) > 0:
                assert self.as2 is not None,'You have to provide an instance of as2 to perform as2-related operations.'
            if kwargs.get('aberrations') is not None:
                for key in keys:
                    if relative_aberrations:
                        self.aberrations[key] = self.as2.get_property_as_float(controls[key]) * 1e9 + \
                                                kwargs['aberrations'].get(key, 0)
                    else:
                        self.aberrations[key] = \
                            kwargs['aberrations'].get(key, self.as2.get_property_as_float(controls[key]) * 1e9)

                    if reset_aberrations:
                        originals[key] = self.as2.get_property_as_float(controls[key]) * 1e9
            # Apply corrector values to the Hardware
            for key in self.aberrations.keys():
                self.as2.set_property_as_float(controls[key], self.aberrations[key] * 1e-9)

            if acquire_image:
                assert self.superscan is not None, \
                    'You have to provide an instance of superscan in order to perform superscan-related operations.'
                record_parameters = self.create_record_parameters()
                #self.superscan.set_frame_parameters(**self.record_parameters)
                #if self.superscan.is_playing:
                #    self.superscan.stop_playing()
#                channels_enabled = [False, False]
#                if self.detectors['HAADF']:
#                    channels_enabled[0] = True
#                if self.detectors['MAADF']:
#                    channels_enabled[1] = True
                if self.frame_parameters.get('center') is not None:
                    rot = self.frame_parameters.get('rotation', 0) * 180 / np.pi
                    rotated_center = np.dot(np.array(((np.cos(rot), -np.sin(rot)), (np.sin(rot), np.cos(rot)))),
                                            np.array(self.frame_parameters.get('center')) * 1e-9)

                    self.as2.set_property_as_float('CSH.y', rotated_center[0])
                    self.as2.set_property_as_float('CSH.x', rotated_center[1])
                    time.sleep(0.1)

                im = None
                import threading
                image_finished = threading.Event()
                def get_image():
                    nonlocal im
                    im = self.superscan.record(**record_parameters)
                    image_finished.set()
                self.document_controller.queue_task(get_image)
                image_finished.wait()
                image_finished.clear()

                return_image = [data_and_metadata.data for data_and_metadata in im]


#                self.document_controller.queue_task(lambda:
#                                        self.superscan._HardwareSource__hardware_source.set_selected_profile_index(1))

#                default_params = ss.SS_Functions_SS_GetFrameParamsForProfile2(1)
#                ss.SS_Functions_SS_SetFrameParamsForProfile(1,
#                                                  self.frame_parameters.get('size_pixels', (default_params[0],
#                                                                                            default_params[0]))[1],
#                                                  self.frame_parameters.get('size_pixels', (default_params[1],
#                                                                                            default_params[1]))[0],
#                                                  self.frame_parameters.get('center', (default_params[2],
#                                                                                       default_params[2]))[1],
#                                                  self.frame_parameters.get('center', (default_params[3],
#                                                                                       default_params[3]))[0],
#                                                  self.frame_parameters.get('pixeltime', default_params[4]),
#                                                  self.frame_parameters.get('fov', default_params[5]),
#                                                  self.frame_parameters.get('rotation',
#                                                                            default_params[6]*180/np.pi)/180*np.pi,
#                                                  0)
#
#                ss.SS_Functions_SS_SetFrameParams(
#                                                  self.frame_parameters.get('size_pixels', (default_params[0],
#                                                                                            default_params[0]))[1],
#                                                  self.frame_parameters.get('size_pixels', (default_params[1],
#                                                                                            default_params[1]))[0],
#                                                  self.frame_parameters.get('center', (default_params[2],
#                                                                                       default_params[2]))[1],
#                                                  self.frame_parameters.get('center', (default_params[3],
#                                                                                       default_params[3]))[0],
#                                                  self.frame_parameters.get('pixeltime', default_params[4]),
#                                                  self.frame_parameters.get('fov', default_params[5]),
#                                                  self.frame_parameters.get('rotation',
#                                                                            default_params[6]*180/np.pi)/180*np.pi,
#                                                  0)
#
#                acchannels = 0
#                if self.detectors['HAADF']:
#                    acchannels += 1
#                if self.detectors['MAADF']:
#                    acchannels +=2
#                ss.SS_Functions_SS_SetAcquisitionChannels(acchannels)
##                self.document_controller.queue_task(lambda:
##                    self.superscan.abort_playing())
##                frame_nr = ss.SS_Functions_SS_StartFrame2(False, 1)
#                frame_nr = ss.SS_Functions_SS_StartFrameFromProfile(False, 1)
#                ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
#                while not ss.SS_Functions_SS_GetRemainingPixelsForFrame(frame_nr) == -1:
#                    self.logwrite('Waiting for Frame to finish.')
#                    time.sleep(0.01)
#                return_image = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
#                startwaittime = time.time()
#                while (return_image[-1] == 0).all():
#                    if time.time() - startwaittime > 1:
#                        self.logwrite('Exceeded maximum waiting time for frame data.')
#                        break
#                    self.logwrite('Waiting for frame to be fully transfered.')
#                    return_image = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
#                    time.sleep(0.01)
#
                if self.frame_parameters.get('center') is not None:
                    self.as2.set_property_as_float('CSH.y', 0.0)
                    self.as2.set_property_as_float('CSH.x', 0.0)
#
#                if self.detectors['HAADF'] and self.detectors['MAADF']:
#                    data = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 1))
#                    return_image = [return_image, data]
#
#                if show_live_image:
#                    self.show_live_image(return_image)

                #im = self.superscan.grab_next_to_start(channels_enabled=channels_enabled)
#                if len(im) > 1:
#                    return_image = []
#                    for entry in im:
#                        return_image.append(entry.data)
#                else:
#                    return_image = im[0].data
            # reset all corrector values to the original ones
            for key in originals.keys():
                self.as2.set_property_as_float(controls[key], originals[key] * 1e-9)
                #vt.as2_set_control(controls[key], originals[key] * 1e-9)
                self.aberrations[key] = originals[key]

        # e.g. offline mode
        else:
            global global_aberrations

            assert self.imsize is not None, \
                   'You have to input the size (in nm) for the generated image in order to use the offline mode.'

            if self.delta_graphene is None:
                assert self.shape is not None, \
                       'You have to input the shape for the generated image in order to use the offline mode.'

            # Update aberrations dictionary with the values passed to this function
            if kwargs.get('aberrations') is not None:
                for key in keys:
                    # Relative aberrations is here relative to global_aberrations, in online mode its relative to
                    # the values already set in as2
                    if relative_aberrations:
                        self.aberrations[key] = global_aberrations.get(key, 0) + kwargs['aberrations'].get(key, 0)
                    else:
                        self.aberrations[key] = kwargs['aberrations'].get(key, global_aberrations.get(key, 0))

            # Write current values to global aberrations if they should be kept (which is similar to applying them to
            # the hardware in online mode)
            if not reset_aberrations:
                for key in self.aberrations.keys():
                    global_aberrations[key] = self.aberrations[key]

            print(self.aberrations)

            if acquire_image:
                # Create x and y coordinates such that resulting beam has the same scale as the image.
                # The size of the kernel which is used for image convolution is chosen to be "1/kernelsize"
                # of the image size (in pixels)
                kernelsize = 2
                kernelpixel = int(self.shape[0]/kernelsize)

                if self.delta_graphene is None:
                    impix = self.shape[0]+kernelpixel-1
                    imsize = impix/self.shape[0]*self.imsize
                    rotation = self.frame_parameters.get('rotation', 0)
                    self.delta_graphene = self.graphene_generator(imsize, impix, rotation)

                frequencies = np.matrix(np.fft.fftshift(np.fft.fftfreq(kernelpixel, self.imsize/self.shape[0])))
                x = np.array(np.tile(frequencies, np.size(frequencies)).reshape((kernelpixel,kernelpixel)))
                y = np.array(np.tile(frequencies.T, np.size(frequencies)).reshape((kernelpixel,kernelpixel)))

                # compute aberration function up to threefold astigmatism
                # formula taken from "Advanced Computing in Electron Microscopy",
                # Earl J. Kirkland, 2nd edition, 2010, p. 18
                # wavelength for 60 keV electrons: 4.87e-3 nm
                raw_kernel = (-self.aberrations.get('EHTFocus', 0) * (x**2 + y**2) +

                              np.sqrt(self.aberrations.get('C12_a', 0)**2 + self.aberrations.get('C12_b', 0)**2) *
                              (x**2 + y**2) * np.cos(2 * (np.arctan2(y,x) -
                                                     np.arctan2(self.aberrations.get('C12_b', 0),
                                                                self.aberrations.get('C12_a', 0)))) +
                              (2.0/3.0) *
                              np.sqrt(self.aberrations.get('C21_a', 0)**2 + self.aberrations.get('C21_b', 0)**2) *
                              4.87e-3 *
                              np.sqrt(x**2 + y**2)**3 * np.cos(np.arctan2(y,x) -
                                                               np.arctan2(self.aberrations.get('C21_b', 0),
                                                                          self.aberrations.get('C21_a', 0))) +
                              (2.0/3.0) *
                              np.sqrt(self.aberrations.get('C23_a', 0)**2 + self.aberrations.get('C23_b', 0)**2) *
                              4.87e-3 *
                              np.sqrt(x**2 + y**2)**3 * np.cos(3 * (np.arctan2(y,x) -
                                                               np.arctan2(self.aberrations.get('C23_b', 0),
                                                                          self.aberrations.get('C23_a', 0))))) * \
                              np.pi * 4.87e-3

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
                if self.frame_parameters.get('pixeltime', 0) < 0:
                    multiplicator = 1
                else:
                    multiplicator = self.frame_parameters.get('pixeltime', 0)*100+1
                im = fftconvolve((self.delta_graphene+0.002)*multiplicator, kernel, mode='valid')
                if self.frame_parameters.get('pixeltime', 0) >= 0:
                    im = np.random.poisson(lam=im.flatten(), size=np.size(im)).astype(im.dtype)

                if debug_mode:
                    return_image = [im.reshape(self.shape).astype('float32'), kernel]
                else:
                    return_image = [im.reshape(self.shape).astype('float32')]
        #print(self.aberrations)
        return return_image

    def logwrite(self, msg, level='info'):
        if self.document_controller is None:
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
                self.document_controller.queue_task(lambda: logging.info(str(msg)))
            elif level.lower() == 'warn':
                self.document_controller.queue_task(lambda: logging.warn(str(msg)))
            elif level.lower() == 'error':
                self.document_controller.queue_task(lambda: logging.error(str(msg)))
            else:
                self.document_controller.queue_task(lambda: logging.debug(str(msg)))

    def show_live_image(self, image):
        assert self.document_controller is not None, 'Cannot create a data item without a document controller instance'

        if self.live_data_item_MAADF is None and self.detectors['MAADF']:
            self.live_data_item_MAADF = self.document_controller.library.create_data_item('Live (MAADF)')
        if self.live_data_item_HAADF is None and self.detectors['HAADF']:
            self.live_data_item_HAADF = self.document_controller.library.create_data_item('Live (HAADF)')

        if self.detectors['HAADF'] and self.detectors['MAADF']:
            self.live_data_item_HAADF.set_data(image[0])
            self.live_data_item_MAADF.set_data(image[1])
        elif self.detectors['HAADF']:
            self.live_data_item_HAADF.set_data(image)
        elif self.detectors['MAADF']:
            self.live_data_item_MAADF.set_data(image)

class Peaking(Imaging):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fft = kwargs.get('fft')
        self.peaks = kwargs.get('peaks')
        self._center = kwargs.get('center')
        self.integration_radius = kwargs.get('integration_radius', 1)

    @Imaging.image.setter
    def image(self, image):
        self._image = image
        self._shape = np.shape(image)
        self._center = tuple((np.array(np.shape(image))/2).astype(np.int))
        self.fft = None
        self._mask = None
        self.peaks = None

    @property
    def center(self):
        if self._center is None:
            self._center = (np.array(self.shape)/2).astype(np.int)
        return np.array(self._center).astype(np.int)

    @center.setter
    def center(self, center):
        self._center = center

    @property
    def fft(self):
        if self._fft is None:
            assert self.image is not None, 'Can not calculate the fft because no image is given.'
            self._fft = np.fft.fftshift(np.fft.fft2(self.image))
        return self._fft

    @fft.setter
    def fft(self, fft):
        self._fft = fft

    def analyze_fft(self, full_output=False, **kwargs):
        coords = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        coords = coords.astype(np.float)
        coords[0] -= self.center[0]
        coords[1] -= self.center[1]
#        radii = np.sqrt(np.sum(coords**2, axis=0))
#        coords /= radii
#        coords[:, tuple(self.center)] = 0
        # preprocessing of fft
        if 'fft' in kwargs:
            fft = kwargs['fft']
        else:
            fft = np.abs(self.fft)
            center = np.zeros(fft.shape)
            draw_circle(center, self.center, np.rint(self.imsize/8) or 1, color=1)
            center = fft * center
            # This has to be with the default color (-1) in order to make remove_edge_effects work correctly
            draw_circle(fft, self.center, np.rint(self.imsize/8) or 1)
            fft = self.remove_edge_effects(fft, half_line_thickness=1)
            fft += center
            #draw_circle(fft, self.center, np.rint(self.imsize/8) or 1, color=2*np.mean(fft))
            #inner_filter = gaussian2D(np.mgrid[0:self.shape[0], 0:self.shape[1]], self.shape[0]/2, self.shape[1]/2, 4, 4, -1, 1)
            fft = (np.real(np.fft.ifft2(scipy.ndimage.fourier_gaussian(np.fft.fft2(fft), 2))))**2
            # cut the image off at 10 x pixel size
            outer_filter = gaussian2D(np.mgrid[0:self.shape[0], 0:self.shape[1]], self.center[0], self.center[1],
                                      self.shape[0]/10, self.shape[1]/10, 1, 0)
            fft = fft*outer_filter
        #fft[:, self.center[1]] = 0
        #fft[self.center[0], :] = 0
        nu00 = np.sum(fft)
        nu11 = np.sum(coords[0]*coords[1]*fft)/nu00
        nu02 = np.sum(coords[0]**2*fft)/nu00
        nu20 = np.sum(coords[1]**2*fft)/nu00
        nu04 = np.sum(coords[0]**4*fft)/nu00
        nu40 = np.sum(coords[1]**4*fft)/nu00
        # Find image orientation
        # Formula taken from https://en.wikipedia.org/wiki/Image_moment
        covmat = np.array(((nu20,nu11), (nu11,nu02)))
        eigval, eigvec = np.linalg.eig(covmat)
        angle = np.arctan2(*eigvec[:, np.argmax(eigval)])
        excent = np.sqrt(1-np.amin(eigval)**2/np.amax(eigval)**2)
        # Standard deviation in polar coordinates
        # Formula taken from http://stackoverflow.com/questions/13894631/image-skewness-kurtosis-in-python
        stddev_mag = np.sqrt(nu02 + nu20)
        stddev_angle = np.arctan2(np.sqrt(nu02), np.sqrt(nu20))
        # Kurtosis in polar koordinates
        kurtosis_mag = np.sqrt(nu04**2/nu02**4 + nu40**2/nu20**4)
        kurtosis_angle = np.arctan2(nu04/nu02**2, nu40/nu20**2)
        if self.peaks is None:
            self.peaks = fft
        # rotate result by 90 degrees to get angle from x-axis
        if full_output:
            return (positive_angle(angle+np.pi/2), excent,
                    positive_angle(stddev_angle+np.pi/2), stddev_mag,
                    positive_angle(kurtosis_angle+np.pi/2), kurtosis_mag,
                    fft)
        else:
            return (positive_angle(angle+np.pi/2), excent,
                    positive_angle(stddev_angle+np.pi/2), stddev_mag,
                    positive_angle(kurtosis_angle+np.pi/2), kurtosis_mag)

    def find_peaks(self, half_line_thickness=3, position_tolerance=5, second_order=False, debug_mode=False, **kwargs):
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
        if kwargs.get('integration_radius') is not None:
            self.integration_radius = kwargs['integration_radius']

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

        fft = self.remove_edge_effects(fft, half_line_thickness=half_line_thickness)

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
                if debug_mode:
                    break
                else:
                    raise RuntimeError('No peaks could be found in the FFT of im.')
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
                except IndexError as detail:
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

    def find_peaks_orientation(self, **kwargs):
        if self.peaks is None:
            self.peaks = self.find_peaks(**kwargs)
        if len(self.peaks.shape) == 3:
            peaks = self.peaks[0].copy()
        else:
            peaks = self.peaks.copy()

        peaks[:,:2] -= self.center
#        radii = np.sqrt(peaks[:, 0]**2 + peaks[:, 1]**2)
#        peaks[:,0] /= radii
#        peaks[:,1] /= radii
        nu00 = np.sum(peaks[:, 3])
        nu11 = np.sum(peaks[:, 0] * peaks[:, 1] * peaks[:, 3])/nu00
        nu02 = np.sum(peaks[:, 0]**2 * peaks[:, 3])/nu00
        nu20 = np.sum(peaks[:, 1]**2 * peaks[:, 3])/nu00
        nu04 = np.sum(peaks[:, 0]**4 * peaks[:, 3])/nu00
        nu40 = np.sum(peaks[:, 1]**4 * peaks[:, 3])/nu00
        # Formula taken from https://en.wikipedia.org/wiki/Image_moment
        covmat = np.array(((nu20,nu11), (nu11,nu02)))
        eigval, eigvec = np.linalg.eig(covmat)

        angle = np.arctan2(*eigvec[:, np.argmax(eigval)])
        excent = np.sqrt(1-np.amin(eigval)**2/np.amax(eigval)**2)
        # Standard deviation in polar coordinates
        # Formula taken from http://stackoverflow.com/questions/13894631/image-skewness-kurtosis-in-python
        stddev_mag = np.sqrt(nu02 + nu20)
        stddev_angle = np.arctan2(np.sqrt(nu02), np.sqrt(nu20))
        # Kurtosis in polar koordinates
        kurtosis_mag = np.sqrt(nu04**2/nu02**4 + nu40**2/nu20**4)
        kurtosis_angle = np.arctan2(nu04/nu02**2, nu40/nu20**2)
        # rotate result by 90 degrees to get angle from x-axis
        return (positive_angle(angle+np.pi/2), excent,
                positive_angle(stddev_angle+np.pi/2), stddev_mag,
                positive_angle(kurtosis_angle+np.pi/2), kurtosis_mag)

    def fourier_filter(self, filter_radius=10, **kwargs):
        # check if peaks are already saved and if second order is there (if a new image is provided also recalculate)
        if len(np.shape(self.peaks)) < 3 or kwargs.get('image') is not None:
            self.peaks = self.find_peaks(second_order=True, **kwargs)
        xdata = np.mgrid[-filter_radius:filter_radius+1, -filter_radius:filter_radius+1]
        mask = gaussian2D(xdata, 0, 0, filter_radius/2, filter_radius/2, 1, 0)
        maskradius = int(np.shape(mask)[0]/2)
        fft_masked = np.zeros(self.shape, dtype=self.fft.dtype)
        for order in self.peaks:
            for peak in order:
                if np.count_nonzero(peak) > 0:
                    fft_masked[peak[0]-maskradius:peak[0]+maskradius+1, peak[1]-maskradius:peak[1]+maskradius+1] += \
                    self.fft[peak[0]-maskradius:peak[0]+maskradius+1, peak[1]-maskradius:peak[1]+maskradius+1]*mask

        return np.real(np.fft.ifft2(np.fft.fftshift(fft_masked)))

    def remove_edge_effects(self, fft, half_line_thickness=3):
        mean_fft = np.mean(fft[fft>-1])
        if half_line_thickness > 0:
            # Fit horizontal and vertical lines with hyperbola
            cross = np.zeros(self.shape)
            for i in range(-half_line_thickness, half_line_thickness+1):
                horizontal = fft[self.center[0]+i,:]
                vertical = fft[:, self.center[1]+i]
                xdata = np.mgrid[:self.shape[1]][horizontal>-1] - self.center[1]
                ydata = np.mgrid[:self.shape[0]][vertical>-1] - self.center[0]
                horizontal = horizontal[horizontal>-1]
                vertical = vertical[vertical>-1]
                horiz_a = ((np.mean(horizontal[int(len(horizontal) * 0.6) - 3 : int(len(horizontal) * 0.6) + 4]) -
                            np.mean(horizontal[int(len(horizontal) * 0.7) - 3 : int(len(horizontal) * 0.7) + 4])) *
                            2.0 * xdata[int(len(horizontal) * 0.6)])
                vert_a = ((np.mean(vertical[int(len(vertical) * 0.6) - 3 : int(len(vertical) * 0.6) + 4]) -
                           np.mean(vertical[int(len(vertical) * 0.7) - 3 : int(len(vertical) * 0.7) + 4])) *
                           2.0 * ydata[int(len(vertical) * 0.6)])
                horizontal_popt, horizontal_pcov = scipy.optimize.curve_fit(hyperbola1D, xdata[:len(xdata)/2],
                                                                            horizontal[:len(xdata)/2], p0=(horiz_a, mean_fft))
                vertical_popt, vertical_pcov = scipy.optimize.curve_fit(hyperbola1D, ydata[:len(ydata)/2],
                                                                        vertical[:len(ydata)/2], p0=(vert_a, mean_fft))
                vertical_perr = np.sqrt(np.diag(vertical_pcov))
                horizontal_perr = np.sqrt(np.diag(horizontal_pcov))
                print(vertical_popt, vertical_perr, horizontal_popt, horizontal_perr)
                if (np.abs(horizontal_popt) > 2*horizontal_perr).any():
                    cross[self.center[0] + i, xdata + self.center[1]] = hyperbola1D(xdata, *horizontal_popt) - 1.5 * mean_fft
                if (np.abs(vertical_popt) > 2*vertical_perr).any():
                    cross[ydata + self.center[0], self.center[1] + i] = hyperbola1D(ydata, *vertical_popt) - 1.5 * mean_fft

            fft-=cross
        return fft


class Tuning(Peaking):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._merits = {'peaks': self.astig_2f, 'symmetry': self.astig_3f, 'combined': self.combined,
                        'astig_2f': self.astig_2f, 'astig_3f': self.astig_3f, 'coma': self.coma,
                        'intensity': self.coma, 'judge_fft': self.judge_fft,
                        }
#        self._merit_lookup = {'EHTFocus': 'intensity', 'C12_a': 'astig_2f', 'C21_a': 'intensity', 'C23_a': 'astig_3f',
#                              'C12_b': 'astig_2f', 'C21_b': 'intensity', 'C23_b': 'astig_3f'}
        self._merit_lookup = {'EHTFocus': 'intensity', 'C12_a': 'astig_2f', 'C21_a': 'intensity', 'C23_a': 'astig_3f',
                              'C12_b': 'astig_2f', 'C21_b': 'intensity', 'C23_b': 'astig_3f'}
        self._analysis_methods = {'graphene': self.find_peaks_orientation, 'general': self.analyze_fft}
        self.steps = kwargs.get('steps')
        self.keys = kwargs.get('keys')
        self.event = kwargs.get('event')
        self.save_images = kwargs.get('save_images', False)
        self.savepath = kwargs.get('savepath')
        self.average_frames = kwargs.get('average_frames')
        self.aberrations_tracklist = []
        self.merit_history = {}
        self.analysis_results = []
        for key in self._merits.keys():
            self.merit_history[key] = []
        self.run_history = {}
        for key in self._merits.keys():
            self.run_history[key] = []
        self.focus = None
        self.number_corrections = {'EHTFocus': 0, 'astig_2f': 0, 'coma': 0, 'astig_3f': 0}
        self.method = kwargs.get('method', 'graphene')

    @property
    def merit_lookup(self):
        return self._merit_lookup

    @property
    def merits(self):
        return self._merits
    
    @property
    def analysis_methods(self):
        return self._analysis_methods

    def append_merit(self, merit_dict, location=None):
        if location is None:
            location = self.merit_history
        if not isinstance(location, (list, tuple)):
            location = (location,)
        for element in location:
            for key, value in merit_dict.items():
                element[key].append(value)

    def calculate_merit(self):
        result = {}
        for key in self.keys:
            if result.get(self.merit_lookup[key]) is None:
                result[self.merit_lookup[key]] = self.merits[self.merit_lookup[key]]()

        if result.get('intensity') is None:
            result['intensity'] = self.merits['intensity']()

        return result

    def find_direction(self, key, dirt_detection=True, merit='astig_2f', merit_tolerance=0.0):
        #step_multiplicators = [1, 0.5, 2]
        step_multiplicators = [1]
        #step_multiplicators.sort(key=lambda a: np.random.rand())
        #step_multiplicator = None
        print(self.merit_history)
        current = {merit: self.merit_history[merit][-1], 'intensity': self.merit_history['intensity'][-1]}

        for step_multiplicator in step_multiplicators:
            self.logwrite('Finding direction of ' + key + ' with stepsize ' + \
                          str(self.steps[key]*step_multiplicator) + '.')
            #changes = 0.0
            aberrations = {key: self.steps[key]*step_multiplicator}
            #changes += self.steps[key]*step_multiplicator
            self.image = self.image_grabber(aberrations=aberrations, show_live_image=True)[0]
            self.mask = self.dirt_detector() if dirt_detection else None
            try:
                #plus = self.merits[merit]()
                plus = self.calculate_merit()
            except RuntimeError:
                plus = {merit: 1e5}
            except DirtError:
                if self.online:
                    self.aberrations = self.aberrations_tracklist[-1].copy()
                    self.image_grabber(acquire_image=False)[0]

                self.logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                raise

            #passing 2xstepsize to image_grabber to get from +1 to -1
            aberrations = {key: -2.0*self.steps[key]*step_multiplicator}
            #changes += -2.0*self.steps[key]*step_multiplicator
            self.image = self.image_grabber(aberrations=aberrations, show_live_image=True)[0]
            self.mask = self.dirt_detector() if dirt_detection else None
            try:
                #minus = self._merits[merit]()
                minus = self.calculate_merit()
            except RuntimeError:
                minus = {merit: 1e5}
            except DirtError:
                if self.online:
                    self.aberrations = self.aberrations_tracklist[-1].copy()
                    self.image_grabber(acquire_image=False)

                self.logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                raise

            if (minus[merit] < plus[merit] and minus[merit] < current[merit]*(1+merit_tolerance) and
                minus['intensity'] < plus['intensity'] and
                minus['intensity'] < current['intensity']*(1+merit_tolerance)):

                direction = -1
                current = minus
                #setting the stepsize to new value
                self.steps[key] *= step_multiplicator
                # save best tuning
                #self.merit_history[merit].append(current)
                self.append_merit(current)
                # append aberrations
                self.aberrations_tracklist.append(self.aberrations.copy())
                # Save new configuration
                #self.aberrations_tracklist.append(self.aberrations.copy())
                break

            elif (plus[merit] < minus[merit] and plus[merit] < current[merit]*(1+merit_tolerance) and
                  plus['intensity'] < minus['intensity'] and
                  plus['intensity'] < current['intensity']*(1+merit_tolerance)):

                direction = 1
                current = plus
                #setting the stepsize to new value
                self.steps[key] *= step_multiplicator
                #Setting aberrations to values of 'plus' which where the best so far
                aberrations = {key: 2.0*self.steps[key]*step_multiplicator}
                #changes += 2.0*self.steps[key]*step_multiplicator
                #update hardware
                self.image_grabber(acquire_image=False, aberrations=aberrations)
                # save best tuning
                #self.merit_history[merit].append(current)
                self.append_merit(current)
                # append aberrations
                self.aberrations_tracklist.append(self.aberrations.copy())
                # Save new configuration
                #self.aberrations_tracklist.append(self.aberrations.copy())
                break
            else:
                self.aberrations = self.aberrations_tracklist[-1].copy()
                self.image_grabber(acquire_image=False)
                #self.logwrite('Changing the stepsize of ' + key + '.')
                #step_multiplicator *= 2
        # This 'else' belongs to the while loop. It is executed when the loop ends 'normally', e.g. not through
        # break or continue
        else:
            direction = 0
            #self.aberrations = self.aberrations_tracklist[-1].copy()
            #self.image_grabber(acquire_image=False)
            #reduce stepsize for next iteration
            #self.steps[key] *= 0.5

        self.logwrite('Latest ' + merit + ' merit: ' + str(current))
        return direction

    def find_focus(self, stepsize=3, range=9, **kwargs):
        if kwargs.get('method') is not None:
            self.method = kwargs.pop('method')

        self.analysis_results = []
        for i in np.arange(-range, range+stepsize, stepsize):
            aberrations = {'EHTFocus': i}
            self.image = self.image_grabber(aberrations=aberrations, reset_aberrations=True, show_live_image=True)[0]
            try:
                res = self.analysis_methods[self.method]()
            except RuntimeError:
                self.logwrite('No peaks could be found for defocus {:.0f} nm.'.format(i))
                continue
            else:
                self.analysis_results.append(((i, np.sum(self.peaks)) + res))

        analysis_results = np.array(self.analysis_results)
        _has_kurtosis = analysis_results.shape[1] > 7 and self.method == 'general'
        if len(self.analysis_results) < 1:
            raise RuntimeError('Could not find focus.')

        best_focus = np.argmin(analysis_results[:, 7]) if _has_kurtosis else np.argmax(analysis_results[:, 1])

        while best_focus == 0 or best_focus == len(analysis_results)-1:
            if best_focus == 0:
                aberrations = {'EHTFocus': analysis_results[0, 0] - stepsize}
            else:
                aberrations = {'EHTFocus': analysis_results[-1, 0] + stepsize}

            self.image = self.image_grabber(aberrations=aberrations, reset_aberrations=True, show_live_image=True)[0]
            try:
                res = self.analysis_methods[self.method]()
            except RuntimeError:
                self.logwrite('No peaks could be found for defocus {:.0f} nm.'.format(aberrations['EHTFocus']))
                break
            else:
                if best_focus == 0:
                    self.analysis_results.insert(0, (aberrations['EHTFocus'] , np.sum(self.peaks)) + res)
                else:
                    self.analysis_results.append((aberrations['EHTFocus'], np.sum(self.peaks)) + res)

            analysis_results = np.array(self.analysis_results)
            best_focus = np.argmin(analysis_results[:, 7]) if _has_kurtosis else np.argmax(analysis_results[:, 1])


        if len(analysis_results) < 3:
            self.logwrite('Could only detect peaks in less than 3 images ({:.0f}). ' +
                          'Assuming focus at maximum intensity.'.format(len(analysis_results)))
            return (best_focus, analysis_results[best_focus, 0])

        # Only do fit in reasonable range around best focus
        lower_limit = 0 if best_focus - 3 < 0 else best_focus - 3
        upper_limit = None if best_focus + 3 > len(analysis_results) -1 else best_focus + 3
        ind = 7 if _has_kurtosis else 1
        b0 = analysis_results[best_focus, 0]
        y0 = analysis_results[best_focus, ind]
        x1 = analysis_results[best_focus - 1, 0]
        y1 = np.mean((analysis_results[best_focus-1, ind], analysis_results[best_focus+1, ind]))
        a0 = (y1 - y0) / (x1 - b0)
        popt, pcov = scipy.optimize.curve_fit(parabola_1D,
                                              analysis_results[lower_limit:upper_limit , 0],
                                              analysis_results[lower_limit:upper_limit, ind],
                                              p0 = (a0, b0, y0), maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        print(popt, perr)
        return (popt, perr)

    def get_keys(self, **kwargs):
        """
        kwargs are directly passed to find_focus and has_astig. Check them for possible arguments.
        """
        try:
            res = self.find_focus(**kwargs)
        except RuntimeError:
            self.logwrite('Could not find focus.', level='warn')
            return

        self.logwrite('Found focus at {:.2f} +- {:.2f} nm.'.format(res[0][1], res[1][1]))
        self.focus = res[0][1]
        astig = self.has_astig(**kwargs)

        if astig[0]:
            self.logwrite('Detected astigmatism as dominant aberration (angle change: {:.1f} deg).'
                          .format(astig[1]*180/np.pi))
            self.number_corrections['astig_2f'] += 1
            return ['C12_a', 'C12_b']
        elif astig[1]:
            self.logwrite('Detected no dominant astigmatism (angle change: {:.1f} deg).'.format(astig[1]*180/np.pi))
            self.number_corrections['coma'] += 1
            self.number_corrections['astig_3f'] += 1
            return ['C21_a', 'C21_b', 'C23_a', 'C23_b']
        else:
            self.logwrite('Could not measure astigmatism.')
            if self.number_corrections['coma'] > self.number_corrections['astig_2f'] + 1:
                self.logwrite('Still correcting astigmatism because it was not corrected during the last runs.')
                self.number_corrections['astig_2f'] += 1
                return ['C12_a', 'C12_b']
            else:
                self.logwrite('Assuming that astigmatism is not the dominant aberration.')
                self.number_corrections['coma'] += 1
                self.number_corrections['astig_3f'] += 1
                return ['C21_a', 'C21_b', 'C23_a', 'C23_b']

    def get_analysis_result_for_defocus(self, defocus=-5, tolerance=2, **kwargs):
        assert self.focus is not None, 'Focus must be found before this!'
        if kwargs.get('method') is not None:
            self.method = kwargs.pop('method')

        if np.iterable(defocus):
            pass
        else:
            defocus = np.resize(defocus, 1)

        if np.iterable(tolerance):
            assert np.size(defocus) == np.size(tolerance), 'Defocus and tolerance must have the same length.'
        else:
            tolerance = np.resize(tolerance, np.size(defocus))

        results_at_defoci = [None]*len(defocus)

        # find the closest value to the requested defoci
        for result in self.analysis_results:
            absolute_focus = result[0] - self.focus

            for i in range(len(results_at_defoci)):
                if np.abs(absolute_focus - defocus[i]) <= tolerance[i]:
                    if (results_at_defoci[i] is None or np.abs(absolute_focus - defocus[i]) <
                                                        np.abs(results_at_defoci[i][0] - self.focus - defocus[i])):
                        results_at_defoci[i] = result

        # acquire images and analyze them for defoci where no valid result was already stored
        for i in range(len(results_at_defoci)):
            if results_at_defoci[i] is None:
                aberrations = {'EHTFocus': self.focus + defocus[i]}
                self.image = self.image_grabber(aberrations=aberrations, reset_aberrations=True)[0]
                try:
                    res = self.analysis_methods[self.method]()
                except RuntimeError:
                    self.logwrite('No peaks could be found for defocus {:.0f} nm.'.format(aberrations['EHTFocus']))
                else:
                    results_at_defoci[i] = (aberrations['EHTFocus'], np.sum(self.peaks)) + res

        return results_at_defoci

    def has_astig(self, defocus=5, tolerance=2, **kwargs):
        if len(self.analysis_results) < 3:
            self.logwrite('Could only analyze less than 3 images ({:.0f}). Checking for astigmatism not possible.'
                          .format(len(self.analysis_results)))
            return (False, None)

        assert self.focus is not None, 'Focus must be found before measuring astigmatism.'

        negative_defocus, positive_defocus = self.get_analysis_result_for_defocus(defocus=[-defocus, defocus],
                                                                                  tolerance=tolerance, **kwargs)

        if None in (negative_defocus, positive_defocus):
            self.logwrite('Could not measure astigmatism because one of the defocused images could not be analyzed.')
            return (False, None)

        angle_change = angle_difference(negative_defocus[2], positive_defocus[2])
        print(negative_defocus[2], positive_defocus[2])

        if 2*np.pi/3 > angle_change > np.pi/3:
            return (True, angle_change)
        else:
            return (False, angle_change)

    def kill_aberrations(self, dirt_detection=True, merit = 'intensity', max_run_number = 5, **kwargs):
        # Backup original frame parameters
        original_frame_parameters = self.frame_parameters.copy()
        auto_keys = False
        # Check input for arguments that override class variables
        if kwargs.get('steps') is not None:
            self.steps = kwargs['steps']
        if kwargs.get('keys') is not None:
            self.keys = kwargs['keys']
        if kwargs.get('method') is not None:
            self.method = kwargs.pop('method')

        if self.keys is 'auto':
            auto_keys = True
            self.keys = None
        if kwargs.get('frame_parameters') is not None:
            self.frame_parameters = kwargs['frame_parameters']
        else:
            self.frame_parameters = {'size_pixels': (512, 512), 'center': (0,0), 'pixeltime': 8, 'fov': 4}

        # Apply default values if one required parameter is not set
        if self.steps is None:
            self.steps = {'EHTFocus': 1, 'C12_a': 1, 'C12_b': 1, 'C21_a': 150, 'C21_b': 150, 'C23_a': 75, 'C23_b': 75}
        if self.keys is None:
            #self.keys = ['EHTFocus', 'C12_a', 'C21_a', 'C23_a', 'C12_b', 'C21_b', 'C23_b']
            self.keys = ['EHTFocus', 'C21_a', 'C21_b', 'C23_a', 'C23_b', 'C12_a', 'C12_b']
        if auto_keys:
            all_keys = ['EHTFocus', 'C21_a', 'C21_b', 'C23_a', 'C23_b', 'C12_a', 'C12_b']
        # Check if merit should be adapted automatically to current aberration
        auto_merit = False
        if merit == 'auto':
            merit = 'intensity'
            auto_merit = True
        else:
            for key in self.keys:
                self._merit_lookup[key] = merit

        step_originals = self.steps.copy()
        self.aberrations_tracklist = []
        self.merit_history = {}
        for key in self._merits.keys():
            self.merit_history[key] = []
        self.run_history = {}
        for key in self._merits.keys():
            self.run_history[key] = []

        counter = 0
        self.imsize = self.frame_parameters['fov']

        self.image = self.image_grabber(aberrations={}, show_live_image=True)[0]
        self.mask = self.dirt_detector() if dirt_detection else None

        try:
            #current = self._merits[merit]()
            current = self.calculate_merit()
            print(current)
        except RuntimeError as detail:
            #current = 1e5
            current = {}
            for key in self.keys:
                if current.get(self.merit_lookup[key]) is None:
                    current[self.merit_lookup[key]] = 1e5
            print(str(detail))
        except DirtError:
            self.frame_parameters = original_frame_parameters.copy()
            self.logwrite('Tuning ended because of too high dirt coverage.', level='warn')
            raise

        #self.merit_history[merit].append(current)
        #self.run_history[merit].append(current)
        self.append_merit(current, location=(self.merit_history, self.run_history))

        # append current corrector configuration to aberrations_tracklist
        self.aberrations_tracklist.append(self.aberrations.copy())

        #total_tunings.append(current)
        self.logwrite('Appending start value: ' + str(current))

        while counter < max_run_number:
            if self.event is not None and self.event.is_set():
                break
            start_time = time.time()
            if counter > 0 and len(self.run_history[merit]) < counter+1:
                self.logwrite('Finished tuning because no improvements could be found anymore.')
                break

            if len(self.run_history[merit]) > 1:
                self.logwrite('Improved tuning by ' +
                              str(np.abs((self.run_history['intensity'][-2] - self.run_history['intensity'][-1]) /
                              ((self.run_history['intensity'][-2]+self.run_history['intensity'][-1])*0.5)*100)) + '%.')

            if len(self.run_history['intensity']) > 1:
                if np.abs((self.run_history['intensity'][-2] - self.run_history['intensity'][-1]) /
                          ((self.run_history['intensity'][-2] + self.run_history['intensity'][-1])*0.5)) < 0.005:
                    self.logwrite('Finished tuning successfully after %d runs.' %(counter))
                    break

            self.logwrite('Starting run number '+str(counter+1))
            #part_tunings = []

            if auto_keys:
                self.keys = self.get_keys(**kwargs)
                if self.keys is None:
                    self.keys = ['EHTFocus', 'C12_a', 'C12_b', 'C21_a', 'C21_b', 'C23_a', 'C23_b']
                else:
                    aberrations={'EHTFocus': self.focus}
                    C12 = self.measure_astig()
                    if C12 is not None:
                        self.keys = ['C21_a', 'C21_b', 'C23_a', 'C23_b']
                        aberrations['C12_b'] = C12[0]
                        aberration['C12_a'] = C12[1]
                    self.image_grabber(acquire_image=False, aberrations=aberrations)

            for key in self.keys:
                if self.event is not None and self.event.is_set():
                    break

                self.logwrite('Working on: '+ key)
                if auto_merit:
                    merit = self.merit_lookup[key]
                try:
                    direction = self.find_direction(key, dirt_detection=dirt_detection, merit=merit)
                except DirtError:
                    raise

                if direction == 0:
                    self.logwrite('Could not find a direction to improve ' + key + '. Going to next aberration.')
                    continue
                else:
                    stringed_direction = 'positive' if direction > 0 else 'negative'
                    self.logwrite('Trying to improve ' + key + ' with stepsize ' +
                                  str(self.steps[key]) + ' in ' + stringed_direction  + ' direction.')
                    #current = self.merit_history[merit][-1]
                    current = {}
                    for key2 in self.keys:
                        if current.get(self.merit_lookup[key2]) is None:
                            current[self.merit_lookup[key2]] = self.merit_history[self.merit_lookup[key2]][-1]
                    if current.get('intensity') is None:
                        current['intensity'] = self.merit_history['intensity'][-1]

                small_counter = 1
                while True:
                    small_counter+=1
                    aberrations = {key: direction*self.steps[key]}
                    #changes += direction*self.steps[key]
                    self.image = self.image_grabber(aberrations=aberrations, show_live_image=True)[0]
                    self.mask = self.dirt_detector() if dirt_detection else None
                    try:
                        #next_frame = self._merits[merit]()
                        next_frame = self.calculate_merit()
                    except RuntimeError:
                        aberrations = {key: -direction*self.steps[key]}
                        #changes -= direction*self.steps[key]
                        #update hardware
                        self.image_grabber(acquire_image=False, aberrations=aberrations)
                        break
                    except DirtError:
                        if self.online:
                            self.aberrations = self.aberrations_tracklist[-1].copy()
                            self.image_grabber(acquire_image=False)
                        self.frame_parameters = original_frame_parameters.copy()
                        self.logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                        raise

                    if next_frame[merit] > current[merit] or next_frame['intensity'] > current['intensity']:
                        aberrations = {key: -direction*self.steps[key]}
                        #changes -= direction*self.steps[key]
                        #update hardware
                        self.image_grabber(acquire_image=False, aberrations=aberrations)
                        #part_tunings.append(merit(current))
                        #part_tunings.append(current)
                        #part_lens.append(np.count_nonzero(current))
                        break
                    current = next_frame

                #only keep changes if they improve the overall tuning
                if False:#len(self.merit_history[merit]) > 0:
                    if current[merit] > np.amin(self.merit_history[merit]):
                        self.image = self.image_grabber(show_live_image=True)[0]
                        self.mask = self.dirt_detector() if dirt_detection else None
                        try:
                            #current = self._merits[merit]()
                            current = self.calculate_merit()
                        except DirtError:
                            self.frame_parameters = original_frame_parameters.copy()
                            self.logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                            raise
                        if current[merit] > np.amin(self.merit_history[merit]):
                            self.aberrations = self.aberrations_tracklist[-1].copy()
                            self.image = self.image_grabber(show_live_image=True)[0]
                            self.mask = self.dirt_detector() if dirt_detection else None
                            try:
                                #current = self._merits[merit]()
                                current = self.calculate_merit()
                            except DirtError:
                                self.frame_parameters = original_frame_parameters.copy()
                                self.logwrite('Tuning ended because of too high dirt coverage.', level='warn')
                                raise
                            self.logwrite('Dismissed changes at '+ key)
                        else:
                            self.logwrite('Kept changes at '+ key + ' after measuring again.')
                            self.logwrite('Found new best tuning with ' + merit  + ' merit: '  + str(current) +
                                      ' by changing ' + key + ' to ' + str(self.aberrations[key]) + '.')
                            #self.merit_history[merit].append(current)
                            self.append_merit(current)
                            self.aberrations_tracklist.append(self.aberrations.copy())
                    else:
                        self.logwrite('Found new best tuning with ' + merit  + ' merit: '  + str(current) +
                                      ' by changing ' + key + ' to ' + str(self.aberrations[key]) + '.')
                        #self.merit_history[merit].append(current)
                        self.append_merit(current)
                        self.aberrations_tracklist.append(self.aberrations.copy())
                else:
                    self.logwrite('Found new best tuning with ' + merit  + ' merit: '  + str(current) +
                                  ' by changing ' + key + ' to ' + str(self.aberrations[key]) + '.')
                    #self.merit_history[merit].append(current)
                    # make sure once per run the merit for all keys is calculated to prevent index_out_of_range errors
                    if auto_keys:
                        self.keys = all_keys
                        current = self.calculate_merit()

                    self.append_merit(current)
                    self.aberrations_tracklist.append(self.aberrations.copy())
                #reduce stepsize for next iteration
                #self.steps[key] *= 0.5
                # append current corrector configuration to aberrations_tracklist


#            if len(part_tunings) > 0:
#                self.logwrite('Appending best value of this run to total_tunings: '+str(np.amin(part_tunings)))
#                self.merit_history[merit].append(np.amin(part_tunings))
#                #total_lens.append(np.amax(part_lens))
            #self.run_history[merit].append(self.merit_history[merit][-1])
            for key2, value in self.merit_history.items():
                if len(value) > 0:
                    self.run_history[key2].append(value[-1])
            self.logwrite('Finished run number '+str(counter+1)+' in '+str(time.time()-start_time)+' s.')
            counter += 1
            #self.keys.sort(key=lambda a: np.random.rand())
        # This else belongs to the while loop. It is executed when the loop ends 'normally', e.g not through
        # break.
        else:
            self.logwrite('Finished tuning because maximum number of runs was exceeded.')

#        if save_images:
#            try:
#                tuning_merit(frame_parameters['fov'], average_frames, integration_radius, save_images, savepath,
#                             dirt_threshold, kwargs)
#            except DirtError:
#                if online:
#                    for key, value in global_aberrations.items():
#                        kwargs[key] = aberrations_last_run[key]-value
#                    image_grabber(acquire_image=False, **kwargs)
#                logwrite('Tuning ended because of too high dirt coverage.', level='warn')
#                raise
#            except:
#                pass
#        else:
#            image_grabber(acquire_image=False, **kwargs)
        self.steps = step_originals.copy()
        self.frame_parameters = original_frame_parameters.copy()

    def measure_astig(self, **kwargs):
        assert self.focus is not None, 'Focus must be found before measuring astigmatism!'
        if kwargs.get('method') is not None:
            self.method = kwargs.pop('method')

        has_astig = self.has_astig(**kwargs)
        if not has_astig[0]:
            print(has_astig)
            self.logwrite('No dominant astigmatism found in this measurement.')
            return None
        analysis_results = np.array(self.analysis_results)
        normalized_excent = analysis_results[:, 3]/analysis_results[:, 7]
        astig_defocus = np.argmax(normalized_excent)
        print('Defocus: ' + str(astig_defocus))
        if astig_defocus == 0:
            astig_defocus += 1
        print('Defocus: ' + str(astig_defocus))
        if astig_defocus == len(analysis_results) - 1:
            astig_defocus -= 1
        print('Defocus: ' + str(astig_defocus))
        astig_defocus = parabola_through_three_points((normalized_excent[astig_defocus-1], analysis_results[astig_defocus-1, 0]),
                                                      (normalized_excent[astig_defocus], analysis_results[astig_defocus, 0]),
                                                      (normalized_excent[astig_defocus+1], analysis_results[astig_defocus+1, 0]))[1]
        print('Defocus: ' + str(astig_defocus))
        astig_defocus -= self.focus
        astig_angle = self.get_analysis_result_for_defocus(defocus=astig_defocus, **kwargs)[0][2]
        print('Defocus: ' + str(astig_defocus))
        print('self.focus:'  + str(self.focus))
        self.logwrite('Found maximum excentricity at {:.1f} nm defocus. Angle: {:.1f} deg.'.format(astig_defocus,
                                                                                           astig_angle*180/np.pi))
        #astig_angle -= np.pi/2 if astig_defocus > 0 else 0
        #shear_angle = np.pi/4
        # Calculate astigmatism in weird coordinates of the corrector from polar coordinates
        C12 = np.array((np.sqrt(2) * np.sin(np.abs(np.arcsin(np.sin(astig_angle))) - np.pi/4),
                        np.sqrt(2) * np.sin(np.abs(np.arcsin(np.sin(astig_angle + np.pi/4))) - np.pi/4)))
        # Normalize it
        C12 /= np.sqrt(np.sum(C12**2))
        # Multiply with defocus to get actual values
        C12 *= astig_defocus
        self.logwrite('Measured astigmatism: C12.a: {:.2f} nm, C12.b: {:.2f}'.format(C12[1], C12[0]))
        return C12

    def astig_2f(self):
        # Check if peaks are already stored and if only first order is there
        if len(np.shape(self.peaks)) != 1:
            try:
                self.peaks = self.find_peaks(second_order=False)
            except RuntimeError as detail:
                print(str(detail))
                return 1000
        #peaks_first, peaks_second = self.peaks
        peaks_first = self.peaks
        intensities = []
        for peak in peaks_first:
            intensities.append(peak[3])
        #for peak in peaks_second:
        #    intensities.append(peak[3])
        self.logwrite('intensity sum: ' + str(np.sum(intensities)) + '\tintensity first var/mean: ' +
                      str(np.std(intensities[:6])/np.mean(intensities[:6])) )#+ '\tintensity second var/mean: ' +
                      #str(np.std(intensities[6:])/np.mean(intensities[6:])))

        #return 1/np.sum(intensities) * 1e3 + np.std(intensities[:6])/np.mean(intensities[:6])
        return np.sqrt((intensities[0] - intensities[1])**2 + (intensities[0] - intensities[2])**2 +
                       (intensities[1] - intensities[2])**2)/np.sum(intensities[0:3])

    def astig_3f(self):
        try:
            ffil = self.fourier_filter()
        except RuntimeError as detail:
            print(str(detail))
            return 1000

        #if self.mask is None:
        ffil = scipy.ndimage.gaussian_filter(ffil, 4)
        res=self.measure_symmetry(ffil)#, np.std(ffil))
        print(res)
        res = (res[1], np.std(ffil))
        return 1/np.sum(res)
        #else:
        #    return 1/np.prod(self.measure_symmetry(ffil)[1]*(1.0-np.sum(self.mask)/mean),
        #            np.std(ffil[self.mask==0])/mean)

    def coma(self):
        # Check if peaks are already stored
        if self.peaks is None:
            try:
                self.peaks = (self.analyze_fft(full_output=True)[-1] if self.method == 'general' else
                              self.find_peaks(second_order=False))
            except RuntimeError as detail:
                print(str(detail))
                return 1000
        #peaks_first, peaks_second = self.peaks
        if self.method == 'general':
            intensities = self.peaks
        else:
            peaks_first = self.peaks
            intensities = []
            for peak in peaks_first:
                intensities.append(peak[3])
        #for peak in peaks_second:
        #    intensities.append(peak[3])
#        self.logwrite('intensity sum: ' + str(np.sum(intensities)) + '\tintensity first var/mean: ' +
#                      str(np.std(intensities[:6])/np.mean(intensities[:6])) + '\tintensity second var/mean: ' +
#                      str(np.std(intensities[6:])/np.mean(intensities[6:])))
        return 1/(np.sum(np.array(intensities))) * 1e5

    def measure_symmetry(self, filtered_image):
        point_mirrored = np.flipud(np.fliplr(filtered_image))
        return autoalign.find_shift(filtered_image[50:-50, 50:-50], point_mirrored[50:-50, 50:-50],
                                    ratio=0.142/self.imsize/2)

    def combined(self, abort_tuning_threshold=0.5):
        if self.mask is not None:
            if np.sum(self.mask) > abort_tuning_threshold*np.prod(self.shape):
                raise DirtError('Cannot tune on images with more than {:.0%} dirt.'.format(abort_tuning_threshold))

        intensities = self.peak_intensity_merit()
        symmetry = self.symmetry_merit()

#        print('sum intensities: ' + str(np.sum(intensities)/1e3) + '\tvar intensities: ' +
#              str(np.std(intensities)/np.sum(intensities)) + '\tsymmetry: ' + str(symmetry))

        return np.sum(intensities, symmetry)

    def judge_fft(self):
#        inner_filter = gaussian2D(np.mgrid[0:self.shape[0], 0:self.shape[1]], self.shape[0]/2, self.shape[1]/2, 4, 4, -1, 1)
#        outer_filter = gaussian2D(np.mgrid[0:self.shape[0], 0:self.shape[1]], self.shape[0]/2, self.shape[1]/2, self.shape[0]/4, self.shape[1]/4, 1, 0)
#        filtered_fft = self.fft*inner_filter*outer_filter
        result = self.analyze_fft(full_output=True)
        self.peaks = result[-1]
        if len(result) > 6 and self.method == 'general':
            return result[5]
        else:
            return 1/np.sum(self.peaks) * 1e6

def draw_circle(image, center, radius, color=-1, thickness=-1):
    subarray = image[center[0]-radius:center[0]+radius+1, center[1]-radius:center[1]+radius+1]
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    distances = np.sqrt(x**2+y**2)
    if thickness < 0:
        subarray[distances < radius + np.sqrt(2)/2] = color
    elif thickness == 0:
        subarray[(distances < radius + np.sqrt(2)/2) * (distances > radius - np.sqrt(2)/2)] = color
    else:
        subarray[(distances < radius+thickness+1) * (distances > radius-thickness)] = color

def gaussian2D(xdata, x0, y0, x_std, y_std, amplitude, offset):
    return (amplitude*np.exp( -0.5*( ((xdata[1]-x0)/x_std)**2 + ((xdata[0]-y0)/y_std)**2 ) ) + offset)

def parabola_1D(xdata, a, b, c):
    """
    Calculates a parabola of the form: a*(x-b)**2 + c
    """
    return a*(xdata-b)**2 + c

def hyperbola1D(xdata, a, offset):
    a, offset = float(a), float(offset)
    return a*np.abs(1.0/xdata**2)+offset

def positive_angle(angle):
    """
    Calculates the angle between 0 and 2pi from an input angle between -pi and pi (all angles in rad)
    """
    if angle < 0:
        return angle  + 2*np.pi
    else:
        return angle

def angle_difference(angle1, angle2):
    """
    Calculates the difference between angle1 and angle2 such that the result is always smaller than pi (all angles in
    rad). The returned difference is always positive.
    """
    diff = np.abs(angle1 - angle2)
    if diff > np.pi:
        diff = 2*np.pi - diff

    return diff

def parabola_through_three_points(p1, p2, p3):
    """
    Calculates the parabola a*(x-b)+c through three points. The points should be given as (y, x) tuples.
    Returns a tuple (a, b, c)
    """
    # formula taken from http://stackoverflow.com/questions/4039039/fastest-way-to-fit-a-parabola-to-set-of-points
    # Avoid division by zero in calculation of s
    if p2[0] == p3[0]:
        temp = p2
        p2 = p1
        p1 = temp

    s = (p1[0]-p2[0])/(p2[0]-p3[0])
    b = (-p1[1]**2 + p2[1]**2 + s*(p2[1]**2 - p3[1]**2)) / (2*(-p1[1] + p2[1] + s*p2[1] - s*p3[1]))
    a = (p1[0] - p2[0]) / ((p1[1] - b)**2 - (p2[1] - b)**2)
    c = p1[0] - a*(p1[1] - b)**2
    return (a, b, c)