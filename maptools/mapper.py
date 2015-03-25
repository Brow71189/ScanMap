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
    #warnings.warn('Could not import nionccd1010. If You\'re not on an offline version of Swift the ronchigram camera might not work!')
    logging.warn('Could not import nionccd1010. If You\'re not on an offline version of Swift the ronchigram camera might not work!')
    
try:    
    from superscan import SuperScanPy as ss    
except:
    logging.warn('Could not import SuperScanPy. Maybe you are running in offline mode.')
    
import autotune
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
                      pixeltime=4, detectors=('MAADF'), use_z_drive=False, auto_offset=False, auto_rotation=False, autofocus_pattern='edges'):
    """
        This function will take a series of STEM images (subframes) to map a large rectangular sample area.
        coord_dict is a dictionary that has to consist of at least 4 tuples that hold stage coordinates in x,y,z - direction
        and the fine focus value, e.g. the tuples have to have the form: (x,y,z,focus). The keys for the coordinates of the 4 corners
        have to be called 'top-left', 'top-right', 'bottom-right' and 'bottom-left'.
        The user has to set the correct focus in all 4 corners. The function will then adjust the focus continuosly during mapping.
        Optionally, automatic focus adjustment is applied (default = off).
        Files will be saved to disk in a folder specified by the user. For each map a subfolder with current date and time is created.
        
    """
    
    imsize = float(imsize)*1e-9
    rotation = float(rotation)*np.pi/180.0
    
    if autofocus_pattern not in ['edges', 'testing']:
        logging.warn('Unknown option for autofocus_pattern. Defaulting to \'edges\'.')
        autofocus_pattern = 'edges'
    
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
        
    #Find corners of a rectangle that lies inside the four points given by the user
    leftX = np.max((coords[0][0],coords[3][0]))
    rightX = np.min((coords[1][0],coords[2][0]))
    topY = np.min((coords[0][1],coords[1][1]))
    botY = np.max((coords[2][1],coords[3][1]))
    
    #Find scan rotation and offset between the images if desired by the user    
    if auto_offset or auto_rotation:
        #set scan parameters for finding rotation and offset
        ss.SS_Functions_SS_SetFrameParams(impix, impix, 0, 0, 1, imsize*1e9, 0, False, True, False, False)
        #Go first some distance into the opposite direction to reduce the influence of backlash in the mechanics on the calibration
        vt.as2_set_control('StageOutX', leftX-5.0*imsize)
        vt.as2_set_control('StageOutY', topY)
        time.sleep(3)
        #Goto point for first image and aquire it
        vt.as2_set_control('StageOutX', leftX)
        vt.as2_set_control('StageOutY', topY)
        if use_z_drive:
            vt.as2_set_control('StageOutZ', interpolation((leftX,  topY), coords)[0])
        vt.as2_set_control('EHTFocus', interpolation((leftX,  topY), coords)[1])
        time.sleep(3)
        frame_nr = ss.SS_Functions_SS_StartFrame(0)
        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
        im1 = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
        #Go to the right by one half image size
        vt.as2_set_control('StageOutX', leftX+imsize/2.0)
        if use_z_drive:
            vt.as2_set_control('StageOutZ', interpolation((leftX+imsize/2.0,  topY), coords)[0])
        vt.as2_set_control('EHTFocus', interpolation((leftX+imsize/2.0,  topY), coords)[1])
        time.sleep(1)
        frame_nr = ss.SS_Functions_SS_StartFrame(0)
        ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
        im2 = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
        #find offset between the two images        
        frame_rotation, frame_distance = autoalign.align(im1, im2)
        #check if the correlation worked correctly and raise an error if not
        if frame_rotation or frame_distance is None:
            logging.error('Could not find offset and/or rotation automatically. Please disable these two options and set values manually.')
            raise RuntimeError('Could not find offset and/or rotation automatically. Please disable these two options and set values manually.')
        
        logging.info('Found rotation between x-axis of stage and scan to be: '+str(frame_rotation*180/np.pi))
        logging.info('Found that the stage moves %.2f times the image size when setting the moving distance to the image size.' % (frame_distance*2.0/impix))
        if auto_offset:
            offset = impix/frame_distance*2.0 - 1.0
        if auto_rotation:
            rotation = -frame_rotation
        
    #Scan configuration
    try:
        ss.SS_Functions_SS_SetFrameParams(impix, impix, 0, 0, pixeltime, imsize*1e9, rotation, HAADF, MAADF, False, False)
        logging.info('Using frame rotation of: ' + str(rotation*180/np.pi) + ' deg')
    except:
        offline = True        
        logging.warn('Not able to set frame parameters. Going back to offline mode.')
        
    
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
                if do_autofocus and i == 0 and autofocus_pattern == 'edges':
                    new_focus_point.append('top-left')
                elif do_autofocus and autofocus_pattern == 'edges':
                    new_focus_point.append(None)
            else: #Even lines, e.g. scan from right to left
                map_coords.append( tuple( ( leftX+(num_subframes[0]-(i+1))*(imsize+distance),  topY-j*(imsize+distance) ) ) + tuple( interpolation( (leftX+(num_subframes[0]-(i+1))*(imsize+distance),  topY-j*(imsize+distance)), coords) ) )
                frame_number.append(j*num_subframes[0]+(num_subframes[0]-(i+1)))
                if do_autofocus and i == 0 and autofocus_pattern=='edges':
                    new_focus_point.append('top-right')
                elif do_autofocus and autofocus_pattern == 'edges':
                    new_focus_point.append(None)
    
    #Now go to each position in "map_coords" and take a snapshot
    
    #create output folder:
    if os.name is 'posix':
        store = filepath+'/map_'+time.strftime('%d_%m_%Y_%H_%M')+'/'
    else:
        store = filepath+'\\map_'+time.strftime('%d_%m_%Y_%H_%M')+'\\'
    
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
    map_paras = {'Autofocus': translator(do_autofocus), 'Autofocus_pattern': autofocus_pattern, 'Auto Rotation': translator(auto_rotation), 'Auto Offset': translator(auto_offset), 'Z Drive': translator(use_z_drive)}
    for key, value in map_paras.items():
        config_file.write('{0:18}{1:}\n'.format(key+':', value))
    config_file.write('\n#Scan parameters:\n')
    scan_paras = {'SuperScan FOV value': str(imsize*1e9)+' nm', 'Image size': str(impix)+' px', 'Pixel time': str(pixeltime)+' us', 'Offset between images': str(offset)+' x image size', 'Scan rotation': str('%.2f' % (rotation*180.0/np.pi)) +' deg', 'Detectors': str(detectors)}
    for key, value in scan_paras.items():
        config_file.write('{0:25}{1:}\n'.format(key+':', value))
    
    config_file.close()
    
    test_map = []    
    counter = 0
    
    for frame_coord in map_coords:
        counter += 1        
        stagex, stagey, stagez, fine_focus = frame_coord
        logging.info(str(counter)+': x: '+str((stagex))+', y: '+str((stagey))+', z: '+str((stagez))+', focus: '+str((fine_focus)))
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
                time.sleep(4) #time in seconds
            else:
                time.sleep(2)
            
            if do_autofocus:
                if autofocus_pattern == 'edges':
                    #find focus at the edges of the scan area. The new_focus_point list is None everywhere inside the scan area
                    #and contains 'top-left' or 'top-right' at the left and right side of the scan area, respectively.
                    if new_focus_point[counter-1] != None:
                        #find amount of focus adjusting
                        focus_adjusted = autotune.autofocus(start_stepsize=2, end_stepsize=0.5)
                        logging.info('Focus at x: ' + str(frame_coord[0]) + ' y: ' + str(frame_coord[1]) + 'adjusted by ' + str(focus_adjusted) + ' nm. (Originally: '+ str(frame_coord[3]*1e9) + ' nm)')
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
                    #apply Gaussian blur and set all pixels smaller than the treshold to 1, all bigger to 0
                    dirt_detection = cv2.GaussianBlur(data, [0,0], 2.0)
                    dirt_detection[dirt_detection>0.2] = 0.0
                    dirt_detection[dirt_detection<= 0.2] = 1.0
                    #calculate the fraction of 'bad' pixels and save frame if fraction is >0.5, but add note to "bad_frames" file
                    if np.sum(dirt_detection)/(np.shape(data[0])*np.shape(data[1])) > 0.5:
                        tifffile.imsave(store+name, data)
                        test_map.append(frame_coord)
                        bad_frames['name'] = 'Over 50% dirt coverage.'
                        logging.info('Over 50% dirt coverage in ' + name)
                    else:
                        result = autotune.check_tuning(imsize=imsize*1e9, im=data, check_astig=False, process_image=False)
                        if result != 0:
                            intensities, coordinates, absolute_astig_angle, relative_astig_angle = result
                            #if not all 6 reflections are visible apply autofocus
                            if len(intensities) < 6:
                                focus_adjusted = autotune.autofocus(start_stepsize=2, end_stepsize=0.5)
                                logging.info('Focus at x: ' + str(frame_coord[0]) + ' y: ' + str(frame_coord[1]) + 'adjusted by ' + str(focus_adjusted) + ' nm. (Originally: '+ str(frame_coord[3]*1e9) + ' nm)')
                                time.sleep(0.1)
                                frame_nr = ss.SS_Functions_SS_StartFrame(0)
                                ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
                                data = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
                                tifffile.imsave(store+name, data)
                                bad_frames['name'] = str('Bad focus. Adjusted by %.2f nm. Originally: %.2f nm.' %(frame_coord[3]*1e9, focus_adjusted))
                                
                    
            else:
                #Take frame and save it to disk
                frame_nr = ss.SS_Functions_SS_StartFrame(0)
                ss.SS_Functions_SS_WaitForEndOfFrame(frame_nr)
                data = np.asarray(ss.SS_Functions_SS_GetImageForFrame(frame_nr, 0))
                tifffile.imsave(store+str('%.4d_%.3f_%.3f.tif' % (frame_number[counter-1],stagex*1e6,stagey*1e6)), data)
                test_map.append(frame_coord)
        
        else:
            test_map.append(frame_coord)
    
    if do_autofocus and autofocus_pattern == 'testing':        
        bad_frames_file = open(store+'bad_frames.txt', 'w')
        bad_frames_file.write('#This file contains the filenames of \"bad\" frames and the cause for the listing.\n\n')
        for key, value in bad_frames.items():
            bad_frames_file.write('{0:30}{1:}\n'.format(key+':', value))        
        config_file.close()
    
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

