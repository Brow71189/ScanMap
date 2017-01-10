# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:03:46 2015

@author: mittelberger
"""

import logging
import time
import os
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    #from ViennaTools import ViennaTools as vt
    from . import tifffile

from .autotune import Imaging, Tuning, DirtError
from scipy.interpolate import Rbf
from .autoalign import align
import threading
import queue
    
class Mapping(object):

    _corners = ['top-left', 'top-right', 'bottom-right', 'bottom-left']

    def __init__(self, **kwargs):
        self.superscan = kwargs.get('superscan')
        self.as2 = kwargs.get('as2')
        self.ccd = kwargs.get('ccd')
        self.document_controller = kwargs.get('document_controller')
        self.coord_dict = kwargs.get('coord_dict')
        # frame_parameters are: rotation, imsize, impix, pixeltime
        self.frame_parameters = kwargs.get('frame_parameters', {})
        self.detectors = kwargs.get('detectors', {'HAADF': False, 'MAADF': True})
        # supported switches are: do_autotuning, use_z_drive, auto_offset, auto_rotation, compensate_stage_error,
        # acquire_overview, blank_beam, tune_at_edges, abort_series_on_dirt, isotope_mapping
        self.switches = kwargs.get('switches', {})
        self.number_of_images = kwargs.get('number_of_images', 1)
        self.dirt_area = kwargs.get('dirt_area', 0.5)
        self.peak_intensity_reference = kwargs.get('peak_intensity_reference')
        if kwargs.get('savepath') is not None:
            self._savepath = os.path.normpath(kwargs.get('savepath'))
        else:
            self._savepath = None
        self.event = kwargs.get('event')
        self.tune_event = kwargs.get('tune_event')
        self.tune_now_event = kwargs.get('tune_now_event')
        self.abort_series_event = kwargs.get('abort_series_event')
        self.foldername = 'map_' + time.strftime('%Y_%m_%d_%H_%M')
        self.offset = kwargs.get('offset', 0)
        self._online = kwargs.get('online')
        self.retuning_mode = kwargs.get('retuning_mode', ['missing_peaks', 'manual'])
        self.gui_communication = {'series_running': False}
        self.missing_peaks = 0
        self.isotope_mapping_settings = kwargs.get('isotope_mapping_settings', {})
        self.average_number = kwargs.get('average_number', 1)
        self.max_align_dist = kwargs.get('max_align_dist', 0.01)
        self.average_data_item_HAADF = None
        self.average_data_item_MAADF = None
        self.last_frames_HAADF = []
        self.last_frames_MAADF = []
        self.sleeptime = kwargs.get('sleeptime', 2)
        self.nion_frame_parameters = {}

    @property
    def online(self):
        if self._online is None:
            if self.as2 is not None and self.superscan is not None:
                self._online = True
            else:
                logging.info('Going to offline mode because no instance of as2 and superscan was provided.')
                self._online = False
        return self._online

    @online.setter
    def online(self, online):
        self._online = online

    @property
    def savepath(self):
        return self._savepath

    @savepath.setter
    def savepath(self, savepath):
        self._savepath = os.path.normpath(savepath)

    def add_to_last_images(self, image, *args, **kwargs):
        if self.detectors['HAADF'] and self.detectors['MAADF']:
            haadfimage = image[0]
            maadfimage = image[1]
        elif self.detectors['HAADF']:
            haadfimage = image
        elif self.detectors['MAADF']:
            maadfimage = image

        if self.detectors['HAADF']:
            if len(self.last_frames_HAADF) >= self.average_number:
                self.last_frames_HAADF.pop(0)

            if self.switches.get('aligned_average') and len(self.last_frames_HAADF) > 1:
                haadfimage = align(self.last_frames_HAADF[0], haadfimage, ratio=self.max_align_dist)
            self.last_frames_HAADF.append(haadfimage)

        if self.detectors['MAADF']:
            if len(self.last_frames_MAADF) >= self.average_number:
                self.last_frames_MAADF.pop(0)

            if self.switches.get('aligned_average') and len(self.last_frames_MAADF) > 1:
                maadfimage = align(self.last_frames_MAADF[0], maadfimage, ratio=self.max_align_dist)
            self.last_frames_MAADF.append(maadfimage)


    def create_map_coordinates(self, compensate_stage_error=False,
                               positionfile='C:/Users/ASUser/repos/ScanMap/positioncollection.npz'):
        imsize = self.frame_parameters['fov']*1e-9
        distance = self.offset*imsize
        self.num_subframes = np.array((int(np.abs(self.rightX-self.leftX)/(imsize+distance))+1,
                                       int(np.abs(self.topY-self.botY)/(imsize+distance))+1))

        map_coords = []
        frame_info = []


        # add additional lines and frames to number of subframes
        if compensate_stage_error:
            try:
                # Load position errors from disk
                data = np.load(os.path.normpath(positionfile))
            except IOError as detail:
                print('Could not load position data from disk. Reason: ' + str(detail))
                print('Compensate_stage_error will be disabled.')
                compensate_stage_error = False
                self.switches['compensate_stage_error'] = False
            else:
                evenlines = data['evenlines']
                firstlines = data['firstlines']
                mapnames = data['mapnames']
                oddlines = data['oddlines']

                # average over all the datasets. This results in 1-D arrays for the different types of coordinates
                xevenline = np.mean(evenlines[1]*np.array([mapnames['pixelsize']]).T, axis=0)
                yevenline = np.mean(evenlines[0]*np.array([mapnames['pixelsize']]).T, axis=0)
                xoddline = np.mean(oddlines[1]*np.array([mapnames['pixelsize']]).T, axis=0)
                yoddline = np.mean(oddlines[0]*np.array([mapnames['pixelsize']]).T, axis=0)
                xfirstline = np.mean(firstlines[1]*np.array([mapnames['pixelsize']]).T, axis=0)
                yfirstline = np.mean(firstlines[0]*np.array([mapnames['pixelsize']]).T, axis=0)
                # Pick the offsets at the appropriate positions for this specific map and convert them to m
                xevenline = xevenline[np.rint(np.mgrid[0:100:self.num_subframes[0]*1j]).astype(np.int)] * 1e-9
                xoddline = xoddline[np.rint(np.mgrid[0:100:self.num_subframes[0]*1j]).astype(np.int)] * 1e-9
                xfirstline = xfirstline[np.rint(np.mgrid[0:100:self.num_subframes[0]*1j]).astype(np.int)] * 1e-9
                yevenline = yevenline[np.rint(np.mgrid[0:100:self.num_subframes[1]*1j]).astype(np.int)] * 1e-9
                yoddline = yoddline[np.rint(np.mgrid[0:100:self.num_subframes[1]*1j]).astype(np.int)] * 1e-9
                yfirstline = yfirstline[np.rint(np.mgrid[0:100:self.num_subframes[1]*1j]).astype(np.int)] * 1e-9

            # Do not use else here to make sure the zero-offset arrays are also created when compensate_stage_error was
            # disabled in the last step.
        if not compensate_stage_error:
            xevenline = xoddline = xfirstline = np.zeros(self.num_subframes[0])
            yevenline = yoddline = yfirstline = np.zeros(self.num_subframes[1])

#            extra_lines = 2  # Number of lines to add at the beginning of the map
#            extra_frames = 5  # Number of extra moves at each end of a line
#            oldm = 0.25  # odd line distance mover (additional offset of odd lines, in fractions of (imsize+distance))
#            eldm = 0  # even line distance mover (additional offset of even lines, in fractions of (imsize+distance))
#            num_subframes = self.num_subframes + np.array((2*extra_frames, extra_lines))
#            leftX = self.leftX - extra_frames*(imsize+distance)
#            for i in range(extra_lines):
#                if i % 2 == 0:  # Odd lines (have even indices because numbering starts with 0), e.g. map from left to right
#                    topY = self.topY + imsize + oldm*distance
#                else:
#                    topY = self.topY + imsize + eldm*distance
#        else:
#            num_subframes = self.num_subframes
#            leftX = self.leftX
#            topY = self.topY
#            oldm = 0
#            eldm = 0
        # make a list of coordinates where images will be aquired.
        # Starting point is the upper-left corner and mapping will proceed to the right. The next line will start
        # at the right and scan towards the left. The next line will again start at the left, and so on. E.g. a "snake shaped"
        # path is chosen for the mapping.
        num_subframes = self.num_subframes
        leftX = self.leftX
        topY = self.topY
        # Focus interpolation will be done live to take changes of the smaple points into account. The map_coords tuples
        # will still have len-4, because the first two coordinates are the position were the stage will move to, the
        # last two are the target positions for the interpolation. In case of uncorrected stage movement they will be
        # the same, otherwise they can differ.
        for j in range(num_subframes[1]):
            for i in range(num_subframes[0]):
                if j == 0:
                    map_coords.append(tuple((leftX + i*(imsize+distance),
                                             topY - j*(imsize+distance))) +
                                      tuple((leftX + i*(imsize+distance) + xfirstline[i],
                                             topY - j*(imsize+distance) - yfirstline[j])))
                    if self.retuning_mode[0] == 'edges' and i == 0:
                        frame_info.append({'number': j*num_subframes[0]+i, 'retune': True, 'corner': 'top-left'})
                    else:
                        frame_info.append({'number': j*num_subframes[0]+i})
                elif j % 2 == 0:  # Odd lines (have even indices because numbering starts with 0), e.g. map from left to right
                    map_coords.append(tuple((leftX + i*(imsize+distance),
                                             topY - j*(imsize+distance))) +
                                      tuple((leftX + i*(imsize+distance) + xoddline[i],
                                             topY - j*(imsize+distance) - yoddline[j])))

                    if self.retuning_mode[0] == 'edges' and i == 0:
                        frame_info.append({'number': j*num_subframes[0]+i, 'retune': True, 'corner': 'top-left'})
                    else:
                        frame_info.append({'number': j*num_subframes[0]+i})
#                    map_coords.append(tuple((leftX+i*(imsize+distance),
#                                             topY-j*(imsize+distance) - oldm*(imsize+distance))) +
#                                      tuple(self.interpolation((leftX + i*(imsize+distance),
#                                            topY-j*(imsize+distance) - oldm*(imsize+distance)))))

                    # Apply correct (continuous) frame numbering for all cases. If no extra positions are added, just
                    # append the correct frame number. Elsewise append the correct frame number if a non-additional
                    # one, else None
#                    if not compensate_stage_error:
#                        frame_info.append(j*num_subframes[0]+i)
#                    elif extra_frames <= i < num_subframes[0]-extra_frames and j >= extra_lines:
#                        frame_info.append((j-extra_lines)*(num_subframes[0]-2*extra_frames)+(i-extra_frames))
#                    else:
#                        frame_info.append(None)

                else: # Even lines, e.g. scan from right to left
                    map_coords.append(tuple((leftX + (num_subframes[0] - (i+1))*(imsize + distance),
                                             topY-j*(imsize + distance))) +
                                      tuple((leftX + (num_subframes[0] - (i+1))*(imsize + distance) + xevenline[i],
                                             topY-j*(imsize + distance) - yevenline[j])))

                    if self.switches.get('focus_at_edges') and num_subframes[0]-(i+1) == 0:
                        frame_info.append({'number': j*num_subframes[0]+(num_subframes[0]-(i+1)), 'retune': True,
                                             'corner': 'top-right'})
                    else:
                        frame_info.append({'number': j*num_subframes[0]+(num_subframes[0]-(i+1))})

#                    map_coords.append(tuple((leftX + (num_subframes[0] - (i+1))*(imsize + distance),
#                                             topY-j*(imsize + distance) - eldm*(imsize + distance))) +
#                                      tuple(self.interpolation(
#                                            (leftX + (num_subframes[0] - (i+1))*(imsize + distance),
#                                             topY-j*(imsize + distance) - eldm*(imsize + distance)))))
#
#                    # Apply correct (continuous) frame numbering for all cases. If no extra positions are added, just
#                    # append the correct frame number. Elsewise append the correct frame number if a non-additional
#                    # one, else None
#                    if not compensate_stage_error:
#                        frame_info.append(j*num_subframes[0]+(num_subframes[0]-(i+1)))
#                    elif extra_frames <= i < num_subframes[0]-extra_frames and j >= extra_lines:
#                        frame_info.append( (j-extra_lines)*(num_subframes[0]-2*extra_frames) +
#                                             ((num_subframes[0]-2*extra_frames)-(i-extra_frames+1)) )
#                    else:
#                        frame_info.append(None)

        return (map_coords, frame_info)

    def tuning_necessary(self, frame_info, message):
        """
        returns a tuple in the form (True/False, message)
        """

        if self.retuning_mode[0] == 'reference':
            self.Tuner.dirt_mask = self.Tuner.dirt_detector()
            if np.sum(self.Tuner.dirt_mask)/(np.shape(self.Tuner.image)[0]*np.shape(self.Tuner.image)[1]) > 0.5:
                message += 'Over 50% dirt coverage. '
                self.Tuner.logwrite('Over 50% dirt coverage in No. ' + str(frame_info['number']))
                return (False, message)
            graphene_mean = np.mean(self.Tuner.image[self.Tuner.dirt_mask==0])
            self.Tuner.image[self.Tuner.dirt_mask==1] = graphene_mean
            try:
                peaks = self.Tuner.find_peaks(half_line_thickness=2, position_tolerance = 10, second_order=True,
                                              integration_radius=1)
            except RuntimeError as detail:
                message += str(detail) + ' '
                self.Tuner.logwrite(message)
                return (False, message)
            else:
                intensities_sum = np.sum(peaks[0][:,-1])+np.sum(peaks[1][:,-1])
            if intensities_sum < 0.4 * self.peak_intensity_reference:
                message += ('Retune because peak intensity sum is only {:.0f} compared to reference ' +
                            '({:.0f}, {:.1%}). ').format(intensities_sum, self.peak_intensity_reference,
                            intensities_sum/self.peak_intensity_reference)
                self.Tuner.logwrite(message)
                return (True, message)

        elif self.retuning_mode[0] == 'missing_peaks':
            self.Tuner.dirt_mask = self.Tuner.dirt_detector()
            if np.sum(self.Tuner.dirt_mask)/(np.shape(self.Tuner.image)[0]*np.shape(self.Tuner.image)[1]) > 0.5:
                message += 'Over 50% dirt coverage. '
                self.Tuner.logwrite('Over 50% dirt coverage in No. ' + str(frame_info['number']))
                return (False, message)
            graphene_mean = np.mean(self.Tuner.image[self.Tuner.dirt_mask==0])
            self.Tuner.image[self.Tuner.dirt_mask==1] = graphene_mean
            try:
                first_order, second_order = self.Tuner.find_peaks(imsize=self.frame_parameters['fov'],
                                                             second_order=True)
            except RuntimeError:
                first_order = second_order = 0
                number_peaks = 0
                message += str(detail) + ' '
                self.Tuner.logwrite(message)
            else:
                number_peaks = np.count_nonzero(first_order[:,-1]) + np.count_nonzero(second_order[:,-1])

            if number_peaks == 12:
                self.missing_peaks = 0
            elif number_peaks < 10:
                message += 'Missing '+str(12 - number_peaks)+' peaks. '
                self.Tuner.logwrite('No. '+str(frame_info['number']) + ': Missing ' + str(12 - number_peaks) +
                                    ' peaks.')
                self.missing_peaks += 12 - number_peaks

            if self.missing_peaks > 12:
                self.Tuner.logwrite('No. '+str(frame_info['number']) + ': Retune because '+str(self.missing_peaks) +
                               ' peaks miss in total.')
                message += 'Retune because '+str(self.missing_peaks)+' peaks miss in total. '
                return (True, message)

        elif self.retuning_mode[0] == 'edges':
            if frame_info.get('retune'):
                return (True, message)

        return (False, message)

    def tuning_successful(self, success, new_point):
        if success:
            counter = 0
            while counter < 10000:
                if not self.coord_dict.get('new_point_{:04d}'.format(counter)):
                    self.coord_dict['new_point_{:04d}'.format(counter)] = new_point
                    if hasattr(self, 'interpolator'):
                        delattr(self, 'interpolator')
                    break
                counter += 1
            else:
                self.Tuner.logwrite('Could not add new point to coord_dict. Continuing with old ones.')

            if self.retuning_mode[0] == 'reference':
                pass
            elif self.retuning_mode[0] == 'missing_peaks':
                self.missing_peaks = 0
            else:
                pass
        else:
            if self.retuning_mode[0] == 'reference':
                pass
            elif self.retuning_mode[0] == 'missing_peaks':
                self.missing_peaks = 0
            else:
                pass

    def handle_retuning(self, frame_coord, frame_info):
        # tests in each frame after aquisition if all 6 reflections in the fft are still there (only for frames where
        # less than 50% of the area are covered with dirt). If not all reflections are visible, autofocus is applied
        # and the result is added as offset to the interpolated focus values. The dirt coverage is calculated by
        # considering all pixels intensities that are higher than 0.02 as dirt
#        Tuner = Tuning(event=self.event, document_controller=self.document_controller, as2=self.as2,
#                       superscan=self.superscan)
        message = '\tTuner: '
        mode = self.retuning_mode[1]

        if self.tune_now_event is not None and self.tune_now_event.is_set():
            return_message = message + 'User initialized retuning. '
            tune = True
            mode = 'manual'
            self.tune_now_event.clear()
        else:
            tune, return_message = self.tuning_necessary(frame_info, message)

        message = return_message
        if not tune:
            return message
        elif mode == 'manual':
            return_message, focused = self.wait_for_focused(message)
            message = return_message
            if focused is not None:
                new_z, newEHTFocus = focused
                self.tuning_successful(True, frame_coord[:2] + (new_z, newEHTFocus))
            else:
                self.tuning_successful(False, None)
        elif mode == 'auto':
            # find place in the image with least dirt to do tuning there
            clean_spot, size = self.Tuner.find_biggest_clean_spot()
            clean_spot_nm = clean_spot * self.frame_parameters['fov'] / self.frame_parameters['size_pixels']
            tune_frame_parameters = {'size_pixels': (512, 512), 'pixeltime': 8, 'fov': 4,
                                     'rotation': self.frame_parameters['rotation'], 'center': clean_spot_nm}
            if self.switches.get('blank_beam'):
                self.verified_unblank()
            try:
                self.Tuner.kill_aberrations(frame_parameters=tune_frame_parameters)
                if self.event is not None and self.event.is_set():
                    return message
            except DirtError:
                self.Tuner.logwrite('No. '+ str(frame_info['number']) +
                                    ': Tuning was aborted because of dirt coming in.')
                message += 'Tuning was aborted because of dirt coming in. '
            else:
                self.Tuner.logwrite('No. '+ str(frame_info['number']) + ': New tuning: '+str(self.Tuner.aberrations))
                message += 'New tuning: '+ str(self.Tuner.aberrations) + '. '

                data_new = self.Tuner.image_grabber(frame_parameters = self.frame_parameters)[0]
                try:
                    first_order_new, second_order_new = self.Tuner.find_peaks(image=data_new,
                                                                         imsize=self.frame_parameters['fov'],
                                                                         second_order=True)
                    number_peaks_new = np.count_nonzero(first_order_new[:,-1]) + \
                                       np.count_nonzero(second_order_new[:,-1])
                except RuntimeError:
                    message += 'Dismissed result because it did not improve tuning: ' + \
                               str(self.Tuner.aberrations) + '. '
                    self.Tuner.logwrite('No. '+ str(frame_info['number']) + \
                                   ': Dismissed result because it did not improve tuning: ' + \
                                   str(self.Tuner.aberrations))
                    #reset aberrations to values before tuning
                    self.Tuner.image_grabber(acquire_image=False, relative_aberrations=False,
                                        aberrations=self.Tuner.aberrations_tracklist[0])
                    self.tuning_successful(False, None)
                else:
                    message += 'Number of peaks after tuning: {:.0f}. '.format(number_peaks_new)
                    self.Tuner.logwrite('No. : Number of peaks after tuning: {:.0f}.'.format(frame_info['number'],
                                                                                             number_peaks_new))
#                else:
#                    if number_peaks_new > number_peaks:
#                        self.tuning_successful(False, frame_coord[:3] + (self.Tuner.aberrations['EHTFocus'] * 1e-9,))
#                    else:
#                        message += 'Dismissed result because it did not improve tuning: ' + \
#                                   str(self.Tuner.aberrations) + '. '
#                        self.Tuner.logwrite('No. ' + str(frame_info['number']) + \
#                                       ': Dismissed result because it did not improve tuning: ' + \
#                                       str(self.Tuner.aberrations))
#                        #reset aberrations to values before tuning
#                        self.Tuner.image_grabber(acquire_image=False, relative_aberrations=False,
#                                        aberrations=self.Tuner.aberrations_tracklist[0])
#                        self.tuning_successful(False, None)
            if self.switches.get('blank_beam'):
                self.as2.set_property_as_float('C_Blank', 1)
        else:
            pass

        return message

    def handle_isotope_mapping(self, frame_coord, frame_info, frame_name, **kwargs):
        message = '\tIsotope mapper: '
        savepath = os.path.join(self.savepath, self.foldername, os.path.splitext(frame_name)[0])
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if self.isotope_mapping_settings.get('overlap') is not None:
            kwargs['overlap'] = self.isotope_mapping_settings['overlap']
        clean_spots = self.Tuner.find_clean_spots(**kwargs)

        if len(clean_spots) < 1:
            message += 'No clean spots found. '
            self.Tuner.logwrite('No. ' + str(frame_info['number']) + ': No clean spots found.')
            return message

        frame_parameters = self.isotope_mapping_settings.get('frame_parameters')

        Imager = Imaging(as2=self.as2, superscan=self.superscan, document_controller=self.document_controller)
#        Imager.image = self.Tuner.image
#        Imager.imsize = self.frame_parameters['fov']
        # Only calculate dirt threshold once per map and pass it to Imager for performance reasons
        Imager.dirt_threshold = self.Tuner.dirt_threshold

        Imager.logwrite('No. ' + str(frame_info['number']) + ': Start ejecting atoms.')

        if self.switches['blank_beam']:
            self.verified_unblank()

        for clean_spot in clean_spots:
            name = 'spot_y_x_' + str(clean_spot[0]) + '_' + str(clean_spot[1]) + '_'
            message += (name + ': ')
            intensity_reference = 0
            clean_spot_nm = clean_spot * frame_parameters['fov'] / frame_parameters['size_pixels']
            Imager.frame_parameters['center'] = clean_spot_nm
            for i in range(self.isotope_mapping_settings.get('max_number_frames', 1)):
                if self.abort_series_event is not None and self.abort_series_event.is_set():
                    self.abort_series_event.clear()
#                    self.gui_communication['series_running'] = False
#                    self.document_controller.queue_task(lambda: self.update_abort_button('Abort map'))
#                    time.sleep(1)
                    break
                Imager.image = Imager.image_grabber(show_live_image=True, frame_parameters=frame_parameters)[0]
                tifffile.imsave(os.path.join(savepath, name + '{:02d}'.format(i) + '.tif'), Imager.image)
                if self.switches.get('show_last_frames_average'):
                    self.add_to_last_images(Imager.image.copy())
                    self.show_average_of_last_frames()
                if np.sum(Imager.dirt_detector()) > 0:
                    message += 'Detected dirt in current frame. Going to next one.'
                    Imager.logwrite('Detected dirt in current frame. Going to next one.')
                    break
                if i == 0:
                    intensity_reference = np.sum(Imager.image)
                elif (self.isotope_mapping_settings.get('intensity_threshold') > 0 and
                      np.sum(Imager.image) < self.isotope_mapping_settings.get('intensity_threshold', 0.8) *
                      intensity_reference):
                    message += 'Found missing atom after {:d} frames '.format(i)
                    Imager.logwrite('Found missing atom after {:d} frames '.format(i))
                    break
            self.last_frames_HAADF = []
            self.last_frames_MAADF = []

#                elif (np.sum(Imager.image) > 3 - 2*self.isotope_mapping_settings.get('intensity_threshold', 0.8) *
#                      intensity_reference):
#                    message += 'Detected dirt coming in after {:d} frames '.format(i)
#                    Imager.logwrite('Detected dirt coming in after {:d} frames '.format(i))
#                    break
        if self.switches['blank_beam']:
            self.as2.set_property_as_float('C_Blank', 1)

        return message

    def interpolation(self, target):
        """
        Bilinear Interpolation between 4 points that do not have to lie on a regular grid.

        Parameters
        -----------
        target : Tuple
            (x,y) coordinates of the point you want to interpolate

        points : List of tuples
            Defines the corners of a quadrilateral. The points in the list
            are supposed to have the order (top-left, top-right, bottom-right, bottom-left)
            The length of the tuples has to be at least 3 and can be as long as necessary.
            The output will always have the length (points[i] - 2).

        Returns
        -------
        interpolated_point : Tuple
            Tuple with the length (points[i] - 2) that contains the interpolated value(s) at target (i is
            a number iterating over the list entries).
        """
        result = tuple()
        points = []
        if len(self.coord_dict) > 4:
            closest_points = find_nearest_neighbors(4, target, list(self.coord_dict.values()))
            raw_closest_points = []
            for point in closest_points:
                raw_closest_points.append(point[1:])
            coord_dict = self.sort_quadrangle(raw_closest_points)
        else:
            coord_dict = self.coord_dict
        print(coord_dict)
        for corner in self._corners:
            points.append(coord_dict[corner])
        # Bilinear interpolation within 4 points that are not lying on a regular grid.
        m = (target[0]-points[0][0]) / (points[1][0]-points[0][0])
        n = (target[0]-points[3][0]) / (points[2][0]-points[3][0])

        Q1 = np.array(points[0]) + m*(np.array(points[1])-np.array(points[0]))
        Q2 = np.array(points[3]) + n*(np.array(points[2])-np.array(points[3]))

        l = (target[1]-Q1[1]) / (Q2[1]-Q1[1])

        T = Q1 + l*(Q2-Q1)

        for j in range(len(points[0])-2):
            result += (T[j+2],)

    #Interpolation with Inverse distance weighting with a Thin-PLate-Spline Kernel
    #    for j in range(len(points[0])-2):
    #        interpolated = 0
    #        sum_weights = 0
    #        for i in range(len(points)):
    #            distance = np.sqrt((target[0]-points[i][0])**2 + (target[1]-points[i][1])**2)
    #            weight = 1/(distance**2*np.log(distance))
    #            interpolated += weight * points[i][j+2]
    #            sum_weights += weight
    #        result += (interpolated/sum_weights,)
        return result

    def interpolation_rbf(self, target):
        if not hasattr(self, 'interpolator'):
            x = []
            y = []
            z = []
            focus = []

            for value in self.coord_dict.values():
                x.append(value[0])
                y.append(value[1])
                z.append(value[2])
                focus.append(value[3])

            self.interpolator = []
            self.interpolator.append(Rbf(x, y, z, function='inverse'))
            self.interpolator.append(Rbf(x, y, focus, function='inverse'))

        return (self.interpolator[0](*target), self.interpolator[1](*target))

    def load_mapping_config(self, path):
        #config_file = open(os.path.normpath(path))
        #counter = 0
        # make sure function is not caught in an endless loop if 'end' is missing at the end
        #while counter < 1e3:
        with open(os.path.normpath(path)) as config_file:
            for line in config_file:
            #counter += 1
            #line = config_file.readline().strip()
                line = line.strip()

            #if line == 'end':
            #    break
                if line.startswith('#'):
                    continue
                elif line.startswith('{'):
                    line = line[1:].strip()
                    self.fill_dicts(line, config_file)
                elif len(line.split(':', maxsplit=1)) == 2:
                    line = line.split(':', maxsplit=1)
                    setattr(self, line[0].strip(), eval(line[1].strip()))
                else:
                    continue

        #config_file.close()

    def fill_dicts(self, line, file):
        if hasattr(self, line):
            if getattr(self, line) is None:
                setattr(self, line, {})
            counter = 0
            # Make sure function is not caught in an endless loop if '}' is missing
            while counter < 100:
                counter += 1
                subline = file.readline().strip()
                if subline.startswith('}'):
                    break
                elif subline.startswith('#'):
                    continue
#                elif subline.endswith('}'):
#                    subline = subline[:-1]
#                    subline = subline.split(':')
#                    getattr(self, line)[subline[0].strip()] = eval(subline[1].strip())
#                    break
                else:
                    subline = subline.split(':', maxsplit=1)
                    getattr(self, line)[subline[0].strip()] = eval(subline[1].strip())

    def save_mapping_config(self, path=None):
        if path is None:
            path = os.path.join(self.savepath, self.foldername)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, 'configs_map.txt')

        #config_file = open(path, mode='w+')
        with open(path, mode='w+') as config_file:

            config_file.write('# Configurations for ' + self.foldername + '.\n')
            config_file.write('# This file can be loaded to resume the mapping process with the exact same parameters.\n')
            config_file.write('# Only edit this file if you know what you do. ')
            config_file.write('Otherwise the loading process can fail.\n\n')
            config_file.write('{ switches\n')
            for key, value in self.switches.items():
                config_file.write('\t' + str(key) + ': ' + str(value) + '\n')
            config_file.write('}\n\n{ detectors\n')
            for key, value in self.detectors.items():
                config_file.write('\t' + str(key) + ': ' + str(value) + '\n')
            config_file.write('}\n\n{ coord_dict\n')
            for key, value in self.coord_dict.items():
                config_file.write('\t' + str(key) + ': ' + str(value) + '\n')
            config_file.write('}\n\n{ frame_parameters\n')
            for key, value in self.frame_parameters.items():
                config_file.write('\t' + str(key) + ': ' + str(value) + '\n')
            config_file.write('}\n')
            if self.switches.get('isotope_mapping'):
                config_file.write('\n{ isotope_mapping_settings\n')
                for key, value in self.isotope_mapping_settings.items():
                    config_file.write('\t' + str(key) + ': ' + str(value) + '\n')
                config_file.write('}\n')
            config_file.write('\n# Other parameters\n')
            config_file.write('savepath: ' + repr(self.savepath) + '\n')
    #        config_file.write('foldername: ' + repr(self.foldername) + '\n')
            config_file.write('number_of_images: ' + str(self.number_of_images) + '\n')
            config_file.write('offset: ' + str(self.offset) + '\n')
            config_file.write('retuning_mode: ' + str(self.retuning_mode) + '\n')
            config_file.write('dirt_area: ' + str(self.dirt_area) + '\n')
            config_file.write('sleeptime: ' + str(self.sleeptime) + '\n')
            config_file.write('average_number: ' + str(self.average_number) + '\n')
            config_file.write('max_align_dist: ' + str(self.max_align_dist))

            #config_file.write('\nend')

        #config_file.close()

    def show_average_of_last_frames(self, *args, **kwargs):
        assert self.document_controller is not None, 'Cannot create a data item without a document controller instance'
        if self.detectors['HAADF']:
            assert len(self.last_frames_HAADF) > 0, 'No HAADF data to average.'
        if self.detectors['MAADF']:
            assert len(self.last_frames_MAADF) > 0 is not None, 'No MAADF data to average.'

        if self.average_data_item_HAADF is None and self.detectors['HAADF']:
            self.average_data_item_HAADF = self.document_controller.library.create_data_item(
                                           'Average of last {:.0f} frames (HAADF)'.format(self.average_number))
        if self.average_data_item_MAADF is None and self.detectors['MAADF']:
            self.average_data_item_MAADF = self.document_controller.library.create_data_item(
                                           'Average of last {:.0f} frames (MAADF)'.format(self.average_number))

        if self.detectors['HAADF']:
            self.average_data_item_HAADF.set_data(np.mean(self.last_frames_HAADF, axis=0))
        elif self.detectors['MAADF']:
            self.average_data_item_MAADF.set_data(np.mean(self.last_frames_MAADF, axis=0))

    def sort_quadrangle(self, *args):
        """
        Brings 4 points in the correct order to form a quadrangle.

        Parameters
        ----------
        coord_dict : dictionary
            4 points that are supposed to form a quadrangle.

        Returns
        ------
        coord_dict_sorted : dictionary
            Keys are called 'top-left', 'top-right', 'bottom-right' and 'bottom-left'. They contain the respective input tuple.
            Axis of the result are in standard directions, e.g. x points to the right and y to the top.
        """
        result = {}
        if len(args) > 0:
            points = args[0]
        else:
            points = []
            for corner in self._corners:
                points.append(self.coord_dict[corner])

        points.sort()

        if points[0][1] >= points[1][1]:
            result['top-left'] = points[0]
            result['bottom-left'] = points[1]
        elif points[0][1] < points[1][1]:
            result['top-left'] = points[1]
            result['bottom-left'] = points[0]

        if points[2][1] >= points[3][1]:
            result['top-right'] = points[2]
            result['bottom-right'] = points[3]
        elif points[2][1] < points[3][1]:
            result['top-right'] = points[3]
            result['bottom-right'] = points[2]

        return result

    def verified_unblank(self, timeout=1):
        assert self.as2 is not None, 'Cannot do unblank beam without an instance of as2.'
        if not self.ccd:
            self.Tuner.logwrite('Cannot check if beam is unblanked without ccd.')
            self.as2.set_property_as_float('C_Blank', 0)
            return
        self.ccd.set_property_as_float('exposure_ms', 50)
        if not self.ccd.is_playing:
            self.ccd.start_playing()
        reference = np.mean(self.ccd.grab_next_to_finish()[0].data)
        value = 0
        counter = 0
        self.as2.set_property_as_float('C_Blank', 0)
        starttime = time.time()
        while value < 5*reference:
            counter += 1
            if time.time()-starttime > timeout:
                self.Tuner.logwrite('A timeout occured during waiting for beam unblanking. Make sure the CCD is in.')
                break
            value = np.mean(self.ccd.grab_next_to_finish()[0].data)
            time.sleep(0.02)
        else:
            print(str(counter) + ' steps until full unblank.')
        time.sleep(0.02)

    def SuperScan_mapping(self, **kwargs):
        """
            This function will take a series of STEM images (subframes) to map a large rectangular sample area.
            coord_dict is a dictionary that has to consist of at least 4 tuples that hold stage coordinates in x,y,z - direction
            and the fine focus value, e.g. the tuples have to have the form: (x,y,z,focus). The keys for the coordinates of the 4 corners
            have to be called 'top-left', 'top-right', 'bottom-right' and 'bottom-left'.
            The user has to set the correct focus in all 4 corners. The function will then adjust the focus continuosly during mapping.
            Optionally, automatic focus adjustment is applied (default = off).
            Files will be saved to disk in a folder specified by the user. For each map a subfolder with current date and time is created.

        """
        # check for entries in kwargs that override instance variables
        if kwargs.get('event') is not None:
            self.event = kwargs.get('event')
        if kwargs.get('switches') is not None:
            self.switches = kwargs.get('switches')
        if kwargs.get('frame_parameters') is not None:
            self.frame_parameters = kwargs.get('frame_parameters')
        if kwargs.get('detectors') is not None:
            self.detectors = kwargs.get('detectors')
        if kwargs.get('coord_dict') is not None:
            self.coord_dict = kwargs.get('coord_dict')
        if kwargs.get('offset') is not None:
            self.offset = kwargs.get('offset')
        if kwargs.get('number_of_images') is not None:
            self.number_of_images = kwargs.get('number_of_images')
        if kwargs.get('savepath') is not None:
            self.savepath = kwargs.get('savepath')
        if kwargs.get('foldername') is not None:
            self.foldername = kwargs.get('foldername')

        if np.size(self.frame_parameters.get('pixeltime')) > 1 and \
           np.size(self.frame_parameters.get('pixeltime')) != self.number_of_images:
            raise ValueError('The number of given pixeltimes does not match the given number of frames that should ' +
                             'be recorded per location. You can either input one number or a list with a matching ' +
                             'length.')
        pixeltimes = None
        if np.size(self.frame_parameters.get('pixeltime')) > 1:
            pixeltimes = self.frame_parameters.get('pixeltime')
            self.frame_parameters['pixeltime'] = pixeltimes[0]

        self.save_mapping_config()

        self.document_controller.queue_task(lambda: self.update_button('analyze_button', 'Retune now'))

        self.Tuner = Tuning(frame_parameters=self.frame_parameters.copy(), detectors=self.detectors, event=self.event,
                     online=self.online, document_controller=self.document_controller, as2=self.as2,
                     superscan=self.superscan)

        # Sort coordinates in case they were not in the right order
#        self.coord_dict = self.sort_quadrangle()
        # Find bounding rectangle of the four points given by the user
        self.leftX = np.amin((self.coord_dict['top-left'][0], self.coord_dict['bottom-left'][0]))
        self.rightX = np.amax((self.coord_dict['top-right'][0], self.coord_dict['bottom-right'][0]))
        self.topY = np.amax((self.coord_dict['top-left'][1], self.coord_dict['top-right'][1]))
        self.botY = np.amin((self.coord_dict['bottom-left'][1], self.coord_dict['bottom-right'][1]))

        map_coords, map_infos = self.create_map_coordinates(compensate_stage_error=
                                                               self.switches['compensate_stage_error'])
        # create output folder:
        self.store = os.path.join(self.savepath, self.foldername)
        if not os.path.exists(self.store):
            os.makedirs(self.store)

        logfile = open(os.path.join(self.store, 'log.txt'), mode='w')
        test_map = []
        counter = 0
        self.write_map_info_file()
        # Now go to each position in "map_coords" and take a snapshot
        for i in range(len(map_coords)):
            if self.switches.get('isotope_mapping') or self.number_of_images > 1:
                self.gui_communication['series_running'] = True
                self.document_controller.queue_task(lambda: self.update_button('abort_button', 'Abort series'))
            frame_coord = map_coords[i]
            frame_info = map_infos[i]
            if self.event is not None and self.event.is_set():
                break
            counter += 1
            stagex, stagey, stagex_corrected, stagey_corrected = frame_coord
            stagez, fine_focus = self.interpolation_rbf((stagex, stagey))
            self.Tuner.logwrite(str(counter) + '/' + str(len(map_coords)) + ': (No. ' +
                         str(frame_info['number']) + ') x: ' +str((stagex_corrected)) + ', y: ' +
                         str((stagey_corrected)) + ', z: ' + str((stagez)) + ', focus: ' + str((fine_focus)))
            logfile.write(str(counter) + '/' + str(len(map_coords)) + ': (No. ' +
                         str(frame_info['number']) + ') x: ' +str((stagex_corrected)) + ', y: ' +
                         str((stagey_corrected)) + ', z: ' + str((stagez)) + ', focus: ' + str((fine_focus)) + ':\n')
            # only do hardware operations when online
            if self.online:
                if self.switches.get('blank_beam'):
                    self.as2.set_property_as_float('C_Blank', 1)

                self.as2.set_property_as_float('StageOutX', stagex_corrected)
                self.as2.set_property_as_float('StageOutY', stagey_corrected)
                if self.switches['use_z_drive']:
                    self.as2.set_property_as_float('StageOutZ', stagez)
                self.as2.set_property_as_float('EHTFocus', fine_focus)

                # Wait until movement of stage is done (wait longer time before first frame)
                if counter == 1:
                    time.sleep(10) # time in seconds
                else:
                    time.sleep(self.sleeptime)

                name = str('%.4d_%.3f_%.3f.tif' % (frame_info['number'], stagex_corrected*1e6,
                                                   stagey_corrected*1e6))

                    # Take frame and save it to disk
                if self.number_of_images < 2:
                    if self.switches.get('blank_beam'):
                        self.verified_unblank()
                    self.Tuner.image = self.Tuner.image_grabber(show_live_image=True)[0]
                    tifffile.imsave(os.path.join(self.store, name), self.Tuner.image)
                else:
                    if self.switches.get('blank_beam'):
                        self.verified_unblank()
                    splitname = os.path.splitext(name)
                    for k in range(self.number_of_images):
                        if self.abort_series_event is not None and self.abort_series_event.is_set():
                            self.abort_series_event.clear()
#                            self.gui_communication['series_running'] = False
#                            self.document_controller.queue_task(lambda: self.update_abort_button('Abort map'))
#                            time.sleep(1)
                            break
                        if pixeltimes is not None:
                            self.frame_parameters['pixeltime'] = pixeltimes[k]
                        self.Tuner.image = self.Tuner.image_grabber(frame_parameters=self.frame_parameters,
                                                        show_live_image=True)[0]
                        new_name = splitname[0] + ('_{:0'+str(len(str(self.number_of_images)))+'d}'
                                                   ).format(k) + splitname[1]
                        tifffile.imsave(os.path.join(self.store, new_name), self.Tuner.image)

                        if self.switches.get('show_last_frames_average') and not self.switches.get('isotope_mapping'):
                            self.add_to_last_images(self.Tuner.image.copy())
                            self.show_average_of_last_frames()

                        if self.switches.get('abort_series_on_dirt'):
                            dirt_mask = self.Tuner.dirt_detector()
                            if np.sum(dirt_mask)/np.prod(dirt_mask.shape) > self.dirt_area:
                                self.Tuner.logwrite('Series was aborted because of more than ' +
                                             str(int(self.dirt_area*100)) + '% dirt coverage.')
                                break
                    self.last_frames_HAADF = []
                    self.last_frames_MAADF = []

                if self.switches.get('blank_beam'):
                    self.as2.set_property_as_float('C_Blank', 1)

                if self.switches.get('isotope_mapping'):
                    message = self.handle_isotope_mapping(frame_coord, frame_info, name)
                    logfile.write(message + '\n')

                if self.tune_now_event is not None and self.tune_now_event.is_set():
                    message = self.handle_retuning(frame_coord, frame_info)
                    logfile.write(message + '\n')
                elif self.switches.get('do_retuning'):
                    message = self.handle_retuning(frame_coord, frame_info)
                    logfile.write(message + '\n')

            test_map.append(frame_coord + (stagez, fine_focus))

        if self.switches.get('blank_beam'):
            self.as2.set_property_as_float('C_Blank', 0)

        #acquire overview image if desired
        if self.online and self.switches['acquire_overview']:
            self.Tuner.logwrite('Acquiring overview...')
            #Use longest edge as image size
            if abs(self.rightX-self.leftX) < abs(self.topY-self.botY):
                over_size = abs(self.topY-self.botY)*1e9 + 6*self.frame_parameters['fov']
            else:
                over_size = abs(self.rightX-self.leftX)*1e9 + 6*self.frame_parameters['fov']

            #Find center of mapped area:
            map_center = ((self.leftX+self.rightX)/2, (self.topY+self.botY)/2)
            #Goto center
            self.as2.set_property_as_float('StageOutX', map_center[0])
            self.as2.set_property_as_float('StageOutY', map_center[1])
            time.sleep(5)
            #acquire image and save it
            overview_parameters = {'size_pixels': (4096, 4096), 'center': (0,0), 'pixeltime': 4, \
                                'fov': over_size, 'rotation': self.frame_parameters['rotation']}
            self.Tuner.image = self.Tuner.image_grabber(frame_parameters=overview_parameters, show_live_image=True)[0]
            tifffile.imsave(os.path.join(self.store, 'Overview_{:.0f}_nm.tif'.format(over_size)), self.Tuner.image)

        if self.event is None or not self.event.is_set():
            x_map = np.zeros((self.num_subframes[1], self.num_subframes[0]))
            y_map = np.zeros((self.num_subframes[1], self.num_subframes[0]))
            x_corrected_map = np.zeros((self.num_subframes[1], self.num_subframes[0]))
            y_corrected_map = np.zeros((self.num_subframes[1], self.num_subframes[0]))
            z_map = np.zeros((self.num_subframes[1], self.num_subframes[0]))
            focus_map = np.zeros((self.num_subframes[1], self.num_subframes[0]))
            for j in range(self.num_subframes[1]):
                for i in range(self.num_subframes[0]):
                    if j%2 == 0: #Odd lines, e.g. map from left to right
                        x_map[j,i] = test_map[i+j*self.num_subframes[0]][0]
                        y_map[j,i] = test_map[i+j*self.num_subframes[0]][1]
                        x_corrected_map[j,i] = test_map[i+j*self.num_subframes[0]][2]
                        y_corrected_map[j,i] = test_map[i+j*self.num_subframes[0]][3]
                        z_map[j,i] = test_map[i+j*self.num_subframes[0]][4]
                        focus_map[j,i] = test_map[i+j*self.num_subframes[0]][5]
                    else: #Even lines, e.g. scan from right to left
                        x_map[j,(self.num_subframes[0]-(i+1))] = test_map[i+j*self.num_subframes[0]][0]
                        y_map[j,(self.num_subframes[0]-(i+1))] = test_map[i+j*self.num_subframes[0]][1]
                        x_corrected_map[j,(self.num_subframes[0]-(i+1))] = test_map[i+j*self.num_subframes[0]][2]
                        y_corrected_map[j,(self.num_subframes[0]-(i+1))] = test_map[i+j*self.num_subframes[0]][3]
                        z_map[j,(self.num_subframes[0]-(i+1))] = test_map[i+j*self.num_subframes[0]][4]
                        focus_map[j,(self.num_subframes[0]-(i+1))] = test_map[i+j*self.num_subframes[0]][5]


            tifffile.imsave(os.path.join(self.store, 'x_map.tif'), np.asarray(x_map, dtype='float32'))
            tifffile.imsave(os.path.join(self.store, 'y_map.tif'), np.asarray(y_map, dtype='float32'))
            tifffile.imsave(os.path.join(self.store, 'x_corrected_map.tif'), np.asarray(x_corrected_map, dtype='float32'))
            tifffile.imsave(os.path.join(self.store, 'y_corrected_map.tif'), np.asarray(y_corrected_map, dtype='float32'))
            tifffile.imsave(os.path.join(self.store, 'z_map.tif'), np.asarray(z_map, dtype='float32'))
            tifffile.imsave(os.path.join(self.store, 'focus_map.tif'), np.asarray(focus_map, dtype='float32'))

        logfile.write('\nDONE')
        logfile.close()
        self.document_controller.queue_task(lambda: self.update_button('abort_button', 'Abort map'))
        self.document_controller.queue_task(lambda: self.update_button('analyze_button', 'Analyze image'))
        self.Tuner.logwrite('\nDONE\n')

    def update_button(self, button, text):
        if self.gui_communication.get(button) is not None:
            self.gui_communication[button].text = text

    def wait_for_focused(self, message, timeout=360, accept_timeout=30):

        self.document_controller.queue_task(lambda: self.update_button('done_button', 'Done tuning'))
        self.tune_event.set()

        accepted = False
        def was_accepted():
            nonlocal accepted
            accepted = True
        self.document_controller.queue_task(lambda:
        self.document_controller.show_confirmation_message_box('Please focus now at the current position. Do not ' +
                                                               'change the stage position! If you are done, ' +
                                                               'press "Done" and the mapping process will continue.' +
                                                               'Please confirm this message within ' +
                                                               str(accept_timeout) +  ' seconds, otherwise the map ' +
                                                               'will continue with the old values.',
                                                               was_accepted)
        )
        starttime = time.time()
        while time.time() - starttime < accept_timeout:
            if accepted:
                break
            time.sleep(0.1)
        else:
            message += 'Timeout during waiting for confirmation. Continuing with old values. '
            self.Tuner.logwrite(message, level='warn')
            self.superscan.stop_playing()
            self.tune_event.clear()
            return (message, None)

        if self.switches.get('blank_beam'):
            self.as2.set_property_as_float('C_Blank', 0)

        def set_profile_to_puma():
            self.superscan.profile_index = 0
        self.document_controller.queue_task(set_profile_to_puma)
        self.superscan.start_playing()
        starttime = time.time()
        while time.time() - starttime < timeout:
            if not self.tune_event.is_set():
                break
            time.sleep(0.1)
        else:
            message += 'Timeout during waiting for new focus. Keeping old value. '
            self.Tuner.logwrite(message, level='warn')
            self.superscan.abort_playing()
            self.tune_event.clear()
            return (message, None)

        self.superscan.abort_playing()
        if self.switches.get('blank_beam'):
            self.as2.set_property_as_float('C_Blank', 1)

        return (message, (self.gui_communication.pop('new_z'), self.gui_communication.pop('new_EHTFocus')))

    def write_map_info_file(self):

            def translator(switch_state):
                if switch_state:
                    return 'ON'
                else:
                    return 'OFF'

            config_file = open(os.path.join(self.store, 'map_info.txt'), 'w')
            config_file.write('#This file contains all parameters used for the mapping.\n\n')
            config_file.write('#Map parameters:\n')
            map_paras = {'Autofocus': translator(self.switches.get('do_autotuning')),
                         'Auto Rotation': translator(self.switches.get('auto_rotation')),
                         'Auto Offset': translator(self.switches.get('auto_offset')),
                         'Z Drive': translator(self.switches.get('use_z_drive')),
                         'Acquire_Overview': translator(self.switches.get('acquire_overview')),
                         'Number of frames': str(self.num_subframes[0])+'x'+str(self.num_subframes[1]),
                         'Compensate stage error': translator(self.switches.get('compensate_stage_error'))}
            for key, value in map_paras.items():
                config_file.write('{0:18}{1:}\n'.format(key+':', value))

            config_file.write('\n#Scan parameters:\n')
            scan_paras = {'SuperScan FOV value': str(self.frame_parameters.get('fov')) + ' nm',
                          'Image size': str(self.frame_parameters.get('size_pixels'))+' px',
                          'Pixel time': str(self.frame_parameters.get('pixeltime'))+' us',
                          'Offset between images': str(self.offset)+' x image size',
                          'Scan rotation': str(self.frame_parameters.get('rotation')) +' deg',
                          'Detectors': str(self.detectors)}
            for key, value in scan_paras.items():
                config_file.write('{0:25}{1:}\n'.format(key+':', value))

            config_file.close()

class AcquisitionLoop(object):
    """
    This class grabs images from SuperScan and pushes them into a buffer. The items in the buffer will be dictionaries,
    with the data in the key "data". The acquisition function adds additional info to each image that is retrieved by
    calling "get_info_dict". If this function is used it must be callable and take no arguments. It must return a
    dictionary whose items will be added to the buffer item.
    """
    def __init__(self, **kwargs):
        self.buffer = kwargs.get('buffer', Buffer(maxsize=200))
        self.superscan = kwargs.get('superscan')
        self.nion_frame_parameters = kwargs.get('nion_frame_parameters')
        self._n = -1
        self.buffer_timeout = None
        self._pause_timeout = None
        self._pause_event = threading.Event()
        self._abort_event = threading.Event()
        self._acquisition_finished_event = threading.Event()
        self._t = None
        self.get_info_dict = kwargs.get('get_info_dict')
        
    @property
    def is_acquiring(self):
        return not self._acquisition_finished_event.is_set()
        
    def start(self, n=-1):
        if self._t is not None and self._t.is_alive():
            return
        self._pause_event.set()
        self._abort_event.clear()
        self._acquisition_finished_event.clear()
        if n > 0:
            self._n = n
        else:
            self._n = -1
        self._t = threading.Thread(target=self._acquisition_thread, daemon=True)
        self._t.start()     
    
    def pause(self, time=None):
        self._pause_event.clear()
        self.superscan.stop_playing()
    
    def unpause(self):
        self.superscan.set_frame_parameters(**self.nion_frame_parameters)
        self.superscan.start_playing()
        self._pause_event.set()
    
    def abort(self):
        self._abort_event.set()
        self.superscan.stop_playing()
    
    def wait_for_acquisition(self, timeout=None):
        return self._acquisition_finished_event.wait(timeout=timeout)
    
    def _acquisition_thread(self):
        counter = 0
        with self.superscan.create_view_task(**self.nion_frame_parameters):
            self.superscan.start_playing()
            while self._n < 0 or counter < self._n:
                image = {}
                if counter == self._n - 1:
                    self.superscan.stop_playing()
                    image['is_last'] = True
                self._acquisition_finished_event.clear()
                image['data'] = self.superscan.grab_next_to_finish()
                self.buffer.put(image, timeout=self.buffer_timeout)
                if callable(self.get_info_dict):
                    info_dict = self.get_info_dict()
                    image.update(info_dict)
                if not self._pause_event.is_set() or self._abort_event.is_set():
                    self._acquisition_finished_event.set()
                    if self._abort_event.is_set():
                        break
                self._pause_event.wait(timeout=self._pause_timeout)
                counter += 1
        self.superscan.stop_playing()
        self._acquisition_finished_event.set()
        self._n = -1
    
    def close(self):
        self.abort()
    
class Buffer(queue.Queue):
    """
    A buffer for acquired images
    """
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)

    def get(self, block=True, timeout=None):
        obj = super().get(block=block, timeout=timeout)
        self.task_done()
        return obj
    
class ProcessingLoop(object):
    """
    This class will process data from a buffer and notify the main thread about important events found during processing.
    Processing in will run over a list of tasks. A task is a dictionary that needs to contain at least the key "function"
    which must be a valid function that can be run by this code. Further entries in a "task" dictionary are:
    args: tuple/list, positional arguments passed to the respective funtion
    kwargs: dictionary, keyword arguments passed to the respective function
    """
    
    def __init__(self, buffer, **kwargs):
        self.buffer = buffer
        self.tasks = kwargs.get('tasks', [])
        self.buffer_timeout = None
        self._pause_timeout = None
        self._pause_event = threading.Event()
        self._abort_event = threading.Event()
        self._t = None
        self.on_found_something = None
        
    def start(self):
        if self._t is not None and self._t.is_alive():
            return
        self._pause_event.set()
        self._abort_event.clear()
        self._t = threading.Thread(target=self._processing_thread, daemon=True)
    
    def pause(self, time=None):
        self._pause_event.clear()
    
    def unpause(self):
        self._pause_event.set()
    
    def abort(self):
        self._abort_event.set()
    
    def _processing_thread(self):
        while not self._abort_event.is_set():
            image = self.buffer.get(timeout=self.buffer_timeout)
            data = image.pop('data')
            skip_tasks = image.pop('skip_tasks', list())
            for task in self.tasks:
                if task in skip_tasks:
                    continue
                function = task['function']
                args = task.get('args', tuple())
                kwargs = task.get('kwargs', dict())
                kwargs.update(image)
                res = function(data, *args, **kwargs)
                if res is not None and callable(self.on_found_something):
                    self.on_found_something(function.__name__, res)
            self._pause_event.wait(timeout=self._pause_timeout)
    
    def close(self):
        self.buffer.join()
        self.abort()

class MappingLoop(object):
    """
    This class will iterate over a coordinate list and move the stage to each position. It also takes care of focus
    interpolation. After the "start" method is called it will move to the first position and block until the stabilize
    timeout is over. For each of the following elements its "next" method has to be called, which also blocks until the
    stabilize timeout for each position is over.
    The attribute "coordinate_info" can be set to a list of the same length as "coordinate_list". If it is given,
    "start" and "next" will return the corresponding info to each coordinate.
    """
    
    def __init__(self, coordinate_list, **kwargs):
        self._coordinate_list = coordinate_list
        self.as2 = kwargs.get('as2')
        self.switches = kwargs.get('switches', dict())
        self.interpolation = kwargs.get('interpolation')
        self.coordinate_info = kwargs.get('coordinate_info')
        self.first_wait_time = kwargs.get('first_wait_time', 10)
        self.wait_time = kwargs.get('wait_time', 3)        
        self.counter = 0
        self._current_position = None
    
    @property
    def current_position(self):
        return self._current_position
        
    def start(self):
        self._coordinate_iterator = iter(self._coordinate_list)
        self._coordinate_info_iterator = iter(self.coordinate_info) if self.coordinate_info is not None else None
        return self._next(self.first_wait_time)
        
    def next(self):
        return self._next(self.wait_time)
    
    def _next(self, wait_time):
        self.counter += 1
        self._current_position = next(self._coordinate_iterator)
        stagex, stagey, stagex_corrected, stagey_corrected = self.current_position
        stagez, fine_focus = self.interpolation((stagex, stagey))
        self.as2.set_property_as_float('StageOutX', stagex_corrected)
        self.as2.set_property_as_float('StageOutY', stagey_corrected)
        if self.switches.get('use_z_drive'):
            self.as2.set_property_as_float('StageOutZ', stagez)
        self.as2.set_property_as_float('EHTFocus', fine_focus)
        time.sleep(wait_time)
        return_value = (stagex, stagey, stagex_corrected, stagey_corrected, stagez, fine_focus, self.counter)
        if self._coordinate_info_iterator is not None:
            return_value += (next(self._coordinate_info_iterator), )
        return return_value

class SuperScanMapper(Mapping):
    """
    This class is the main part of a mapping process. It takes care of live logging, saving log files and coordinating
    actions between the different working threads.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffer = Buffer(maxsize=200)
        self.processing_loop = ProcessingLoop(self.buffer)
        self.mapping_loop = None
        self.acquisition_loop = None
        self._abort_event = threading.Event()
        self._pause_event = threading.Event()
        self._processing_finished_event = threading.Event()
        self._t = None
        self.tasks = []
        
    def start(self):
        if self._t is not None and self._t.is_alive():
            return
        self._abort_event.clear()
        self._pause_event.set()
        
        self.save_mapping_config()
        self.document_controller.queue_task(lambda: self.update_button('analyze_button', 'Retune now'))
        self.Tuner = Tuning(frame_parameters=self.frame_parameters.copy(), detectors=self.detectors, event=self.event,
                     online=self.online, document_controller=self.document_controller, as2=self.as2,
                     superscan=self.superscan)
        self.create_nion_frame_parameters()
        # Find rectangle inside the four points given by the user
        self.leftX = np.amax((self.coord_dict['top-left'][0], self.coord_dict['bottom-left'][0]))
        self.rightX = np.amin((self.coord_dict['top-right'][0], self.coord_dict['bottom-right'][0]))
        self.topY = np.amin((self.coord_dict['top-left'][1], self.coord_dict['top-right'][1]))
        self.botY = np.amax((self.coord_dict['bottom-left'][1], self.coord_dict['bottom-right'][1]))
        self.map_coords, self.map_infos = self.create_map_coordinates(compensate_stage_error=
                                                            self.switches['compensate_stage_error'])
        self.mapping_loop = MappingLoop(self.map_coords, coordinate_info=self.map_infos, as2=self.as2,
                                        switches=self.switches, interpolation = self.interpolation_rbf)
        # create output folder:
        self.store = os.path.join(self.savepath, self.foldername)
        if not os.path.exists(self.store):
            os.makedirs(self.store)
        self.logfile = open(os.path.join(self.store, 'log.txt'), mode='w')

        self.test_map = []
        self.write_map_info_file()
        if self.switches.get('save'):
            self.tasks.append({'function': self.save_image})
        if self.switches.get('show_last_frames_average'):
            self.tasks.append({'function': self.add_to_last_images})
            self.tasks.append({'function': self.show_average_of_last_frames})
        if self.switches.get('abort_series_on_dirt'):
            self.tasks.append({'function': self.Tuner.dirt_detector})
        if self.switches.get('do_returning'):
            self.tasks.append({'function': self.tuning_necessary})
        # Replace string names in self.tasks with actual functions
        self.setup_tasks()
        self.processing_loop.tasks = self.tasks
        self.processing_loop.start()
        self._t = threading.Thread(target=self._mapping_thread)
        self._t.start()
    
    def abort(self):
        self._abort_event.set()
    
    def pause(self, time=None):
        self._pause_event.clear()
    
    def unpause(self):
        self._pause_event.set()
    
    def _mapping_thread(self):
        stagex, stagey, stagex_corrected, stagey_corrected, stagez, focus, counter, info_dict = self.mapping_loop.start()
        self.write_log('{:.0f}/{:.0f} (No. {:.0f}): x: {:f}, y: {:f}, z: {:f}, focus: {:f}'.format(counter,
                                                                                                   len(self.map_coords),
                                                                                                   info_dict['number'],
                                                                                                   stagex_corrected,
                                                                                                   stagey_corrected,
                                                                                                   stagez,
                                                                                                   focus))
        basename = '{:.4d}_{:.3f}_{:.3f}'.format(info_dict['number'], stagex_corrected, stagey_corrected)
        
        self.processing_loop.start()
        while not self._abort_event.is_set():
            self.test_map.append((stagex, stagey, stagex_corrected, stagey_corrected, stagez, focus))
            if self.number_of_images < 2:
                image_info = [{'name': basename}]
            else:
                num_len = str(len(str(self.number_of_images)))
                image_info = [{'name': basename + ('_{:0' + num_len + 'd}').format(i)}
                              for i in range(self.number_of_images)]
            def get_info_dict():
                return image_info.pop(0)
            self.acquisition_loop = AcquisitionLoop(buffer=self.buffer, get_info_dict=get_info_dict,
                                                    superscan=self.superscan,
                                                    nion_frame_parameters=self.nion_frame_parameters)
            self.acquisition_loop.start()
            self.wait_for_message_or_finished()
            self._pause_event.wait()
            try:
                stagex, stagey, stagex_corrected, stagey_corrected, stagez, focus, counter, info_dict = self.mapping_loop.next()
            except StopIteration:
                break
            self.write_log('{:.0f}/{:.0f} (No. {:.0f}): x: {:f}, y: {:f}, z: {:f}, focus: {:f}'.format(counter,
                                                                                                   len(self.map_coords),
                                                                                                   info_dict['number'],
                                                                                                   stagex_corrected,
                                                                                                   stagey_corrected,
                                                                                                   stagez,
                                                                                                   focus))
            basename = '{:.4d}_{:.3f}_{:.3f}'.format(info_dict['number'], stagex_corrected, stagey_corrected)
        
        self.write_log('\nDONE')
        self.close()
            
        
    def write_log(self, message):
        try:
            self.Tuner.logwrite(message)
        except Exception as e:
            print('Could not print log message! Reason: ' + str(e))
        try:
            self.logfile.write(message + '\n')
        except Exception as e:
            print('Could not write log message to logfile! Reason: ' + str(e))
    
    def wait_for_message_or_finished(self):
        if self.switches.get('wait_for_processing', True):
            self._processing_finished_event.wait()
            self._processing_finished_event.clear()
        else:
            self.acquisition_loop.wait_for_acquisition()
    
    def processing_event_occured(self, taskname, obj):
        """
        Handles events from the processing loop.
        taskname : (str) name of the task that fired that event
        obj : (object) object that was returned by the task
        """
        if taskname == 'processing_finished':
            self._processing_finished_event.set()
        if taskname == 'tuning_necessary':
            if obj[0]:
                self.write_log('Starting retuning, reason: ' + obj[1])
                self.handle_retuning()
        if taskname == 'dirt_detector':
            if self.switches.get('abort_series_on_dirt') and np.sum(obj) > np.prod(obj.shape)*self.dirt_area:
                self.acquisition_loop.abort()
                self.write_log('Aborted series because of too high dirt coverage.')
    
    def handle_retuning(self):
        self.acquisition_loop.pause()
        message = ''
        return_message, focused = self.wait_for_focused(message)
        message = return_message
        if focused is not None:
            new_z, newEHTFocus = focused
            self.tuning_successful(True, self.mapping_loop.current_position[:2] + (new_z, newEHTFocus))
        else:
            self.tuning_successful(False, None)
        self.acquisition_loop.unpause()
    
    def save_image(self, image, *args, **kwargs):
        tifffile.imsave(image.get('name') + '.tif', image.get('data').data)
    
    def setup_tasks(self):
        tasks = []
        for task in self.tasks:
            if callable(task.get('function')):
               tasks.append(task)
            else:
                try:
                    function = getattr(self, task.get('function'))
                except AttributeError:
                    self.write_log('Could not find a function matching name {:s}. Skipping this task.'.format(task))
                else:
                    task['function'] = function
                    tasks.append(task)
        self.tasks = tasks
        
    def create_nion_frame_parameters(self):
        self.nion_frame_parameters = {}
        self.nion_frame_parameters['size'] = tuple(self.frame_parameters['size_pixels'])
        self.nion_frame_parameters['pixel_time_us'] = self.frame_parameters['pixeltime']
        self.nion_frame_parameters['fov_nm'] = self.frame_parameters['fov']
        self.nion_frame_parameters['rotation_rad'] = self.frame_parameters['rotation']/180*np.pi
        
    def close(self):
        self.logfile.close()
        self.acquisition_loop.close()
        self.processing_loop.close()
        
        
#def find_offset_and_rotation(as2, superscan):
#    """
#    This function finds the current rotation of the scan with respect to the stage coordinate system and the offset that has to be set between two neighboured images when no overlap should occur.
#    It takes no input arguments, so the current frame parameters are used for image acquisition.
#
#    It returns a tuple of the form (rotation(degrees), offset(fraction of images)).
#
#    """
#
#    frame_parameters = superscan.get_frame_parameters()
#
#    imsize = frame_parameters['fov_nm']
#
#    image_grabber_parameters = {'size_pixels': frame_parameters['size'], 'rotation': 0,
#                                'pixeltime': frame_parameters['pixel_time_us'], 'fov': frame_parameters['fov_nm']}
#
#    leftX = vt.as2_get_control(as2, 'StageOutX')
#    vt.as2_set_control(as2, 'StageOutX', leftX + 6.0*imsize)
#    time.sleep(5)
#
#    image1 = autotune.image_grabber(frame_parameters=image_grabber_parameters, detectors={'MAADF': True, 'HAADF': False})
#    #Go to the right by one half image size
#    vt.as2_set_control('StageOutX', leftX + 6.5*imsize)
#    time.sleep(3)
#    image2 = autotune.image_grabber(frame_parameters=image_grabber_parameters, detectors={'MAADF': True, 'HAADF': False})
#    #find offset between the two images
#    try:
#        frame_rotation, frame_distance = autoalign.rot_dist_fft(image1, image2)
#    except:
#        raise
#
#    return (frame_rotation, frame_distance)

def find_nearest_neighbors(number, target, points):
    """
    Finds the nearest neighbor(s) of a point (target) in a number of points (points).

    Parameters
    -----------
    number : Integer
        Number of nearest neighbors that should be returned.
    target : Tuple
        Point of which the nearest neighbors will be returned. Length of
        the tuple is arbitrary, the first two entries are assumed to be (x,y)
    points : List of tuples
        Points in which to search for the nearest neighbors.
        Again, the first two entries are assumed to be (x,y)

    Returns
    --------
    nearest_neighbors : List of tuples
        The point with the smallest distance to target is at the first position.
        The tuples are the same as in the imput with an additional entry at their first position
        which is the distance to target.
    """
    nearest = []
    for i in range(number):
        nearest.append((np.inf,))

    for point in points:
        distance = np.sqrt((target[0]-point[0])**2 + (target[1]-point[1])**2)
        if distance < nearest[0][0]:
            nearest[0] = (distance,) + point
            nearest.sort(reverse=True)

    nearest.sort()

    return nearest