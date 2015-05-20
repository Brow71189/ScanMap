# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:03:46 2015

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

from nion.swift import Application
from nion.swift.model import Image
from nion.swift.model import Operation
from nion.swift.model import Region
from nion.swift.model import HardwareSource

try:
    import nionccd1010
except:
    pass
    #warnings.warn('Could not import nionccd1010. If You\'re not on an offline version of Swift the ronchigram camera might not work!')
    #logging.warn('Could not import nionccd1010. If You\'re not on an offline version of Swift the ronchigram camera might not work!')
    
try:    
    from superscan import SuperScanPy as ss    
except:
    pass
    #logging.warn('Could not import SuperScanPy. Maybe you are running in offline mode.')
    
import autotune
from autotune import DirtError
import autoalign
    
def interpolation(target, points):
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
    
#Bilinear interpolation within 4 points that are not lying on a regular grid.
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

def sort_quadrangle(points):
    """
    Brings 4 points in the correct order to form a quadrangle.
    
    Parameters
    ----------
    points : List of tuples
        4 points that are supposed to form a quadrangle.

    Returns
    ------
    sorted_points : Dictionary
        Keys are called 'top-left', 'top-right', 'bottom-right' and 'bottom-left'. They contain the respective input tuple.
        Axis of the result are in standard directions, e.g. x points to the right and y to the top.
    """
    result = {}
    points.sort()
    if points[0][1] > points[1][1]:
        result['top-left'] = points[0]
        result['bottom-left'] = points[1]
    elif points[0][1] < points[1][1]:
        result['top-left'] = points[1]
        result['bottom-left'] = points[0]
    
    if points[2][1] > points[3][1]:
        result['top-right'] = points[2]
        result['bottom-right'] = points[3]
    elif points[2][1] < points[3][1]:
        result['top-right'] = points[3]
        result['bottom-right'] = points[2]
    
    return result
        

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
    
    return nearest.sort()
    
    
def SuperScan_mapping(coord_dict, filepath='Z:\\ScanMap\\', do_autofocus=False, offset = 0.0, rotation = 0.0, imsize=200, impix=512, \
                      pixeltime=4, detectors=('MAADF'), use_z_drive=False, auto_offset=False, auto_rotation=False, autofocus_pattern='edges', \
                      number_of_images=1, acquire_overview=False, document_controller=None, event=None):
    """
        This function will take a series of STEM images (subframes) to map a large rectangular sample area.
        coord_dict is a dictionary that has to consist of at least 4 tuples that hold stage coordinates in x,y,z - direction
        and the fine focus value, e.g. the tuples have to have the form: (x,y,z,focus). The keys for the coordinates of the 4 corners
        have to be called 'top-left', 'top-right', 'bottom-right' and 'bottom-left'.
        The user has to set the correct focus in all 4 corners. The function will then adjust the focus continuosly during mapping.
        Optionally, automatic focus adjustment is applied (default = off).
        Files will be saved to disk in a folder specified by the user. For each map a subfolder with current date and time is created.
        
    """
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
                document_controller.queue_main_thread_task(lambda: logging.info(str(msg)))
            elif level.lower() == 'warn':
                document_controller.queue_main_thread_task(lambda: logging.warn(str(msg)))
            elif level.lower() == 'error':
                document_controller.queue_main_thread_task(lambda: logging.error(str(msg)))
            else:
                document_controller.queue_main_thread_task(lambda: logging.debug(str(msg)))
                
    imsize = float(imsize)*1e-9
    rotation = float(rotation)*np.pi/180.0
    
    if autofocus_pattern not in ['edges', 'testing']:
        logwrite('Unknown option for autofocus_pattern. Defaulting to \'edges\'.', level='warn')
        autofocus_pattern = 'edges'
    
    if np.size(pixeltime) > 1 and np.size(pixeltime) != number_of_images:
        raise ValueError('The number of given pixeltimes do not match the given number of frames that should be recorded per location. You can either input one number or a list with a matching length.')
    
    if np.size(pixeltime) > 1 and do_autofocus == True:
        logwrite('Acquiring an image series and using autofocus is currently not possible. Autofocus will be disabled.', level='warn')
        do_autofocus = False
    #If not running on microscope computer this will be set to true later.
    #No functions that require real Microscope Hardware will be executed, so no errors appear (test mode)    
    offline = False
    
    HAADF = MAADF = False
    if 'MAADF' in detectors:
        MAADF = True
    if 'HAADF' in detectors:
        HAADF = True
    
    #start with getting the corners of the area that will be mapped
    corners = ('top-left', 'top-right', 'bottom-right', 'bottom-left')
    #list of tuples that contain the coordinates of the corners (x,y,z) in the same order as they are listed obove
    coords = []
    for corner in corners:
        coords.append(coord_dict[corner])
    
    #Sort points in case the axis are not pointing in positive x- and y-direction        
    coord_dict_sorted = sort_quadrangle(coords)
    
    #make list of tuples from sorted dictionary
    coords = []
    for corner in corners:
        coords.append(coord_dict_sorted[corner])
        
    #Find bounding rectangle of the four points given by the user
    leftX = np.min((coords[0][0],coords[3][0]))
    rightX = np.max((coords[1][0],coords[2][0]))
    topY = np.max((coords[0][1],coords[1][1]))
    botY = np.min((coords[2][1],coords[3][1]))
    
    #Find scan rotation and offset between the images if desired by the user    
    if auto_offset or auto_rotation:
        #set scan parameters for finding rotation and offset
        ss.SS_Functions_SS_SetFrameParams(impix, impix, 0, 0, 2, imsize*1e9, 0, False, True, False, False)
        #Go first some distance into the opposite direction to reduce the influence of backlash in the mechanics on the calibration
        vt.as2_set_control('StageOutX', leftX-5.0*imsize)
        vt.as2_set_control('StageOutY', topY)
        time.sleep(4)
        #Goto point for first image and aquire it
        vt.as2_set_control('StageOutX', leftX)
        vt.as2_set_control('StageOutY', topY)
        if use_z_drive:
            vt.as2_set_control('StageOutZ', interpolation((leftX,  topY), coords)[0])
        vt.as2_set_control('EHTFocus', interpolation((leftX,  topY), coords)[1])
        time.sleep(4)
        frame_nr = ss.SS_Functions_SS_StartFrame(0)
        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
        im1 = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
        #Go to the right by one half image size
        vt.as2_set_control('StageOutX', leftX+imsize/2.0)
        if use_z_drive:
            vt.as2_set_control('StageOutZ', interpolation((leftX+imsize/2.0,  topY), coords)[0])
        vt.as2_set_control('EHTFocus', interpolation((leftX+imsize/2.0,  topY), coords)[1])
        time.sleep(2)
        frame_nr = ss.SS_Functions_SS_StartFrame(0)
        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
        im2 = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
        #find offset between the two images        
        #frame_rotation, frame_distance = autoalign.rot_dist(im1, im2)
        try:
            frame_rotation, frame_distance = autoalign.rot_dist_fft(im1, im2)
        except RuntimeError:
            logwrite('Could not find offset and/or rotation automatically. Please disable these two options and set values manually.', level='error')
            raise
        
        logwrite('Found rotation between x-axis of stage and scan to be: '+str(frame_rotation))
        logwrite('Found that the stage moves %.2f times the image size when setting the moving distance to the image size.' % (frame_distance*2.0/impix))
        if auto_offset:
            offset = impix/(frame_distance*2.0) - 1.0
        if auto_rotation:
            rotation = -frame_rotation/180*np.pi
        
    #Scan configuration
    try:
        if np.size(pixeltime) > 1:
            ss.SS_Functions_SS_SetFrameParams(impix, impix, 0, 0, pixeltime[0], imsize*1e9, rotation, HAADF, MAADF, False, False)
        else:
            ss.SS_Functions_SS_SetFrameParams(impix, impix, 0, 0, pixeltime, imsize*1e9, rotation, HAADF, MAADF, False, False)
        logwrite('Using frame rotation of: ' + str(rotation*180/np.pi) + ' deg')
    except BaseException as detail:
        offline = True        
        logwrite('Not able to set frame parameters. Going back to offline mode. Reason: '+str(detail), level='warn')
        
    
    #calculate the number of subframes in (x,y). A small distance is kept between the subframes
    #to ensure they do not overlap
    distance = offset*imsize
    num_subframes = ( int(np.abs(rightX-leftX)/(imsize+distance))+1, int(np.abs(topY-botY)/(imsize+distance))+1 )
    #make a list of coordinates where images will be aquired.
    #Starting point is the upper-left corner and mapping will proceed to the right. The next line will start
    #at the right and scan towards the left. The next line will again start at the left, and so on. E.g. a "snake shaped"
    #path is chosen for the mapping.
    map_coords = []
    frame_number = []
    new_focus_point = []
    bad_frames = {}
    
    for j in range(num_subframes[1]):
        for i in range(num_subframes[0]):
            if j%2 == 0: #Odd lines (have even indices because numbering starts with 0), e.g. map from left to right
                map_coords.append( tuple( ( leftX+i*(imsize+distance),  topY-j*(imsize+distance) ) ) + tuple( interpolation((leftX+i*(imsize+distance),  topY-j*(imsize+distance)), coords) ) )
                frame_number.append(j*num_subframes[0]+i)
                if do_autofocus and autofocus_pattern == 'edges':
                    if i==0:
                        new_focus_point.append('top-left')
                    elif i==num_subframes[0]-1:
                        new_focus_point.append('top-right')
                    else:
                        new_focus_point.append(None)
            else: #Even lines, e.g. scan from right to left
                map_coords.append( tuple( ( leftX+(num_subframes[0]-(i+1))*(imsize+distance),  topY-j*(imsize+distance) ) ) + tuple( interpolation( (leftX+(num_subframes[0]-(i+1))*(imsize+distance),  topY-j*(imsize+distance)), coords) ) )
                frame_number.append(j*num_subframes[0]+(num_subframes[0]-(i+1)))
                if do_autofocus and autofocus_pattern=='edges':
                    if i==0:
                        new_focus_point.append('top-right')
                    elif i==num_subframes[0]-1:
                        new_focus_point.append('top-left')
                    else:
                        new_focus_point.append(None)
    
    #Now go to each position in "map_coords" and take a snapshot
    
    #create output folder:
    if os.name is 'posix':
        store = filepath+'/map_'+time.strftime('%Y_%m_%d_%H_%M')+'/'
    else:
        store = filepath+'\\map_'+time.strftime('%Y_%m_%d_%H_%M')+'\\'
    
    if not os.path.exists(store):
        os.makedirs(store)

    def translator(switch_state):
        if switch_state:
            return 'ON'
        else:
            return 'OFF'
    
    config_file = open(store+'map_configurations.txt', 'w')
    config_file.write('#This file contains all parameters used for the mapping.\n\n')
    config_file.write('#Map parameters:\n')
    map_paras = {'Autofocus': translator(do_autofocus), 'Autofocus_pattern': autofocus_pattern, 'Auto Rotation': translator(auto_rotation), 'Auto Offset': translator(auto_offset), 'Z Drive': translator(use_z_drive), \
                'top-left': str(coord_dict_sorted['top-left']), 'top-right': str(coord_dict_sorted['top-right']), 'bottom-left': str(coord_dict_sorted['bottom-left']), 'bottom-right': str(coord_dict_sorted['bottom-right']), \
                'Number of frames': str(num_subframes[0])+'x'+str(num_subframes[1]), 'Acquire_Overview': translator(acquire_overview)}
    for key, value in map_paras.items():
        if key is not 'Autofocus_pattern' or do_autofocus:
            config_file.write('{0:18}{1:}\n'.format(key+':', value))
    config_file.write('\n#Scan parameters:\n')
    scan_paras = {'SuperScan FOV value': str(imsize*1e9)+' nm', 'Image size': str(impix)+' px', 'Pixel time': str(pixeltime)+' us', 'Offset between images': str(offset)+' x image size', 'Scan rotation': str('%.2f' % (rotation*180.0/np.pi)) +' deg', 'Detectors': str(detectors)}
    for key, value in scan_paras.items():
        config_file.write('{0:25}{1:}\n'.format(key+':', value))
    
    config_file.close()
    
    test_map = []    
    counter = 0
    missing_peaks = 0
    
    for frame_coord in map_coords:
        if event is not None and event.is_set():
            break
        counter += 1        
        stagex, stagey, stagez, fine_focus = frame_coord
        logwrite(str(counter)+': (No. '+str(frame_number[counter-1])+') x: '+str((stagex))+', y: '+str((stagey))+', z: '+str((stagez))+', focus: '+str((fine_focus)))
        #print(str(counter)+': x: '+str((stagex))+', y: '+str((stagey))+', z: '+str((stagez))+', focus: '+str((fine_focus)))

        #only do hardware operations when online
        if not offline:
            vt.as2_set_control('StageOutX', stagex)
            vt.as2_set_control('StageOutY', stagey)
            if use_z_drive:
                vt.as2_set_control('StageOutZ', stagez)
            vt.as2_set_control('EHTFocus', fine_focus)
            
            #Wait until movement of stage is done (wait longer time before first frame)
            if counter == 1:            
                time.sleep(10) #time in seconds
            else:
                time.sleep(3)
            
            if do_autofocus:
                if autofocus_pattern == 'edges':
                    #find focus at the edges of the scan area. The new_focus_point list is None everywhere inside the scan area
                    #and contains 'top-left' or 'top-right' at the left and right side of the scan area, respectively.
                    if new_focus_point[counter-1] != None:
                        #find amount of focus adjusting
                        focus_adjusted = autotune.autofocus(start_stepsize=2, end_stepsize=0.5)
                        logwrite('Focus at x: ' + str(frame_coord[0]) + ' y: ' + str(frame_coord[1]) + 'adjusted by ' + str(focus_adjusted) + ' nm. (Originally: '+ str(frame_coord[3]*1e9) + ' nm)')
                        #update the respective coordinate with the new focus
                        new_point = np.array(coord_dict_sorted[new_focus_point[counter-1]])
                        new_point[3] += focus_adjusted*1e-9
                        coord_dict_sorted[new_focus_point[counter-1]] = tuple(new_point)
                        #take frame at this point with the new focus                    
                        vt.as2_set_control('EHTFocus', new_point[3])
                        time.sleep(0.1)
                        frame_nr = ss.SS_Functions_SS_StartFrame(0)
                        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
                        data = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
                        tifffile.imsave(store+str('%.4d_%.2f_%.2f.tif' % (frame_number[counter-1],stagex*1e6,stagey*1e6)), data)
                        test_map.append(tuple(new_point))
                   #make list of tuples from sorted dictionary with new value inside
                    else:
                        #make list of new coordinates in coor_dict_sorted
                        coords = []
                        for corner in corners:
                            coords.append(coord_dict_sorted[corner])
                        #set focus to new interpolated value
                        new_focus = interpolation(frame_coord[0:2], coords)[1]
                        vt.as2_set_control('EHTFocus',  new_focus)
                        time.sleep(0.1)
                        #take frame
                        frame_nr = ss.SS_Functions_SS_StartFrame(0)
                        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
                        data = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
                        tifffile.imsave(store+str('%.4d_%.3f_%.3f.tif' % (frame_number[counter-1],stagex*1e6,stagey*1e6)), data)
                        test_map.append(frame_coord[0:3] + (new_focus,))
                        
                elif autofocus_pattern == 'testing':
                    #tests in each frame after aquisition if all 6 reflections in the fft are still there (only for frames where less than 50% of the area are
                    #covered with dirt). If not all reflections are visible, autofocus is applied and the result is added as offset to the interpolated focus values.
                    #The dirt coverage is calculated by considering all pixels intensities that are higher than 0.02 as dirt
                    frame_nr = ss.SS_Functions_SS_StartFrame(0)
                    ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
                    data = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
                    name = str('%.4d_%.3f_%.3f.tif' % (frame_number[counter-1],stagex*1e6,stagey*1e6))
                    dirt_mask = autotune.dirt_detector(data, threshold=0.01, median_blur_diam=39, gaussian_blur_radius=3)
                    #calculate the fraction of 'bad' pixels and save frame if fraction is >0.5, but add note to "bad_frames" file
                    if np.sum(dirt_mask)/(np.shape(data)[0]*np.shape(data)[1]) > 0.5:
                        tifffile.imsave(store+name, data)
                        test_map.append(frame_coord)
                        bad_frames[name] = 'Over 50% dirt coverage.'
                        logwrite('Over 50% dirt coverage in ' + name)
                    else:
                        try:
                            first_order, second_order = autotune.find_peaks(data, imsize*1e9, position_tolerance=9, second_order=True)
                            number_peaks = len(first_order) + len(second_order)
                        except:
                            first_order = second_order = 0
                            number_peaks = 0
                            
                        if number_peaks == 12:
                            missing_peaks = 0                            
                        elif number_peaks < 10:
                            bad_frames[name] = 'Missing '+str(12 - number_peaks)+' peaks.'
                            logwrite('No. '+str(frame_number[counter-1]) + ': Missing '+str(12 - number_peaks)+' peaks.')
                            missing_peaks += 12 - number_peaks
                        
                        if missing_peaks > 12:
                            logwrite('No. '+str(frame_number[counter-1]) + ': Retune because '+str(missing_peaks)+' peaks miss in total.')
                            bad_frames[name] = 'Retune because '+str(missing_peaks)+' peaks miss in total.'
                            try:
                                tuning_result = autotune.kill_aberrations(event=event, document_controller=document_controller)
                                if event is not None and event.is_set():
                                    break
                            except DirtError:
                                logwrite('No. '+str(frame_number[counter-1]) + ': Tuning was aborted because of dirt coming in.')
                                bad_frames[name] = 'Tuning was aborted because of dirt coming in.'
                                tifffile.imsave(store+name, data)
                                test_map.append(frame_coord)
                            else:
                                logwrite('No. '+str(frame_number[counter-1]) + ': New tuning: '+str(tuning_result))
                                bad_frames[name] = 'New tuning: '+str(tuning_result)
                            
                                frame_nr = ss.SS_Functions_SS_StartFrame(0)
                                ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
                                
                                data_new = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
                                try:
                                    first_order_new, second_order_new = autotune.find_peaks(data, imsize*1e9, position_tolerance=9, second_order=True)
                                    number_peaks_new = len(first_order_new)+len(second_order_new)
                                except RuntimeError:
                                    bad_frames[name] = 'Dismissed result because it did not improve tuning: '+str(tuning_result)
                                    logwrite('No. '+str(frame_number[counter-1]) + ': Dismissed result because it did not improve tuning: '+str(tuning_result))
                                    tifffile.imsave(store+name, data)
                                    test_map.append(frame_coord)
                                    #reset aberrations to values before tuning
                                    kwargs = {'relative_aberrations': True}
                                    for key, value in tuning_result.items():
                                        kwargs[key] = -value
                                    autotune.image_grabber(acquire_image=False, **kwargs)
                                    
                                else:
                                    if number_peaks_new > number_peaks:
                                        tifffile.imsave(store+name, data_new)
                                        #add new focus as offset to all coordinates
                                        for i in range(len(map_coords)):
                                            map_coords[i] = np.array(map_coords[i])
                                            map_coords[i][3] += tuning_result['EHTFocus']*1e-9
                                            map_coords[i] = tuple(map_coords[i])
                                        test_map.append(map_coords[counter-1])
                                        missing_peaks = 0
                                    else:
                                        bad_frames[name] = 'Dismissed result because it did not improve tuning: '+str(tuning_result)
                                        logwrite('No. '+str(frame_number[counter-1]) + ': Dismissed result because it did not improve tuning: '+str(tuning_result))
                                        tifffile.imsave(store+name, data)
                                        test_map.append(frame_coord)
                                        #reset aberrations to values before tuning
                                        kwargs = {'relative_aberrations': True}
                                        for key, value in tuning_result.items():
                                            kwargs[key] = -value
                                        autotune.image_grabber(acquire_image=False, **kwargs)
                                        missing_peaks=0

                        else:
                            tifffile.imsave(store+name, data)
                            test_map.append(frame_coord)                                
                    
            else:
                #Take frame and save it to disk
                if number_of_images < 2:
                    frame_nr = ss.SS_Functions_SS_StartFrame(0)
                    ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
                    data = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
                    tifffile.imsave(store+str('%.4d_%.3f_%.3f.tif' % (frame_number[counter-1],stagex*1e6,stagey*1e6)), data)
                    test_map.append(frame_coord)
                else:
                    if np.size(pixeltime) > 1:
                        original_frame_params = ss.SS_Functions_SS_GetFrameParams()
                        frame_params = list(original_frame_params)
                    for i in range(number_of_images):
                        if np.size(pixeltime) > 1:                        
                            frame_params[4] = pixeltime[i]
                            ss.SS_Functions_SS_SetFrameParams(*frame_params)
                        frame_nr = ss.SS_Functions_SS_StartFrame(0)
                        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
                        data = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
                        tifffile.imsave(store+str('%.4d_%.3f_%.3f_%.2d.tif' % (frame_number[counter-1],stagex*1e6,stagey*1e6, i)), data)
                    ss.SS_Functions_SS_SetFrameParams(*original_frame_params)
                    test_map.append(frame_coord)
        
        else:
            test_map.append(frame_coord)
    
    if do_autofocus and autofocus_pattern == 'testing':        
        bad_frames_file = open(store+'bad_frames.txt', 'w')
        bad_frames_file.write('#This file contains the filenames of \"bad\" frames and the cause for the listing.\n\n')
        for key, value in bad_frames.items():
            bad_frames_file.write('{0:30}{1:}\n'.format(key+':', value))        
        config_file.close()

    #acquire overview image if desired
    if acquire_overview:
        #Use longest edge as image size
        if abs(rightX-leftX) < abs(topY-botY):
            over_size = abs(topY-botY)*1.25e3
        else:
            over_size = abs(rightX-leftX)*1.25e3
        
        #Find center of mapped area:
        map_center = ((leftX+rightX)/2, (topY+botY)/2)
        #Goto center
        vt.as2_set_control('StageOutX', map_center[0])
        vt.as2_set_control('StageOutY', map_center[1])
        time.sleep(10)
        #acquire image and save it
        ss.SS_Functions_SS_SetFrameParams(4096, 4096, 0, 0, 2, over_size, rotation, False, True, False, False)
        image = autotune.image_grabber()
        tifffile.imsave(store+'Overview_'+str(over_size)+'_nm.tif', image)
        
    x_map = np.zeros((num_subframes[1],num_subframes[0]))
    y_map = np.zeros((num_subframes[1],num_subframes[0]))
    z_map = np.zeros((num_subframes[1],num_subframes[0]))
    focus_map = np.zeros((num_subframes[1],num_subframes[0]))
        
    for j in range(num_subframes[1]):
        for i in range(num_subframes[0]):
            if j%2 == 0: #Odd lines, e.g. map from left to right
                x_map[j,i] = test_map[i+j*num_subframes[0]][0]
                y_map[j,i] = test_map[i+j*num_subframes[0]][1]
                z_map[j,i] = test_map[i+j*num_subframes[0]][2]
                focus_map[j,i] = test_map[i+j*num_subframes[0]][3]
            else: #Even lines, e.g. scan from right to left
                x_map[j,(num_subframes[0]-(i+1))] = test_map[i+j*num_subframes[0]][0]
                y_map[j,(num_subframes[0]-(i+1))] = test_map[i+j*num_subframes[0]][1]
                z_map[j,(num_subframes[0]-(i+1))] = test_map[i+j*num_subframes[0]][2]
                focus_map[j,(num_subframes[0]-(i+1))] = test_map[i+j*num_subframes[0]][3]

    
    tifffile.imsave(store+str('x_map.tif'),np.asarray(x_map, dtype='float32'))
    tifffile.imsave(store+str('y_map.tif'),np.asarray(y_map, dtype='float32'))
    tifffile.imsave(store+str('z_map.tif'),np.asarray(z_map, dtype='float32'))
    tifffile.imsave(store+str('focus_map.tif'),np.asarray(focus_map, dtype='float32'))
    
    logwrite('\nDone')

