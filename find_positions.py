# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:33:46 2015

@author: mittelberger
"""

import numpy as np
import cv2
import os
import tifffile
#import matplotlib.pyplot as plt
#from multiprocessing import Pool, Lock, Array
#from ctypes import c_float
    
#def find_position(name, added, over, shape_over, size_frames, size_overview, color):
#    #global added, l, over, shape_over, size_frame, size_overview, color
#    
#    im = cv2.imread(dirpath+name, -1)
#    shape_im = np.shape(im)
#    scale = (size_frames/float(shape_im[0])/(size_overview/float(shape_over[0])))
#    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
#    result = cv2.matchTemplate(over, im, method=cv2.TM_SQDIFF_NORMED)
#    maxi = (np.argmin(result), result.shape)
#    l.acquire()
#    try:
#        added[maxi:] += im.ravel()
#        cv2.rectangle(added, (maxi[1],maxi[0]), (maxi[1]+im.shape[1], maxi[0]+im.shape[0]), color, thickness=2)
#        cv2.putText(added, name[0:4], (maxi[1]-4,maxi[0]-2), cv2.FONT_HERSHEY_PLAIN, 3, color, thickness=2)
#    except Exception as detail:
#        print('Error in '+name+': '+str(detail))
#    finally:
#        l.release()
#
#def init(l):
#    global lock
#    lock=l

if __name__=='__main__':
    
    dirpath = '/3tb/maps_data/map_2015_04_23_19_12_00/'
    
    overview = '/3tb/maps_data/map_2015_04_23_19_12/Overview.tif'
    
    size_overview = 1024 #nm
    size_frames = 12 #nm
    #number of frames in x- and y-direction
    number_frames = (15,13)
    #borders of the area where the first frame is expected (in % of the overview image)
    #has to be a tuple of the form (lower-y, higher-y, lower-x, higher-x)
    area_first_frame = (0,100,0,100) #%
    #enter search area for the following images. Search is always performed around the position of the last image.
    #tuple hast to have the form (negtive-y, positive-y, negative-x, positive-x)
    tolerance_within_rows = (0.25,1.2,0.05,4)
    tolerance_within_columns = (0.25,4,0.25,4)
    
    
    area_first_frame = np.array(area_first_frame)/100.0
    frames = os.listdir(dirpath)
    frames.sort()
    matched_frames = []
    for name in frames:
        num = None
        try:
            int(name[0:4])
        except:
            pass
        else:
#            try:
#                num = int(name[-6:-4])
#            except:
#                pass
#            else:
#                if num is 0:
            matched_frames.append(name)
            
    
    matched_frames.sort()
    over = np.array(cv2.imread(overview, -1))#[1023:3072, 0:2048]*3
    shape_over = np.shape(over)
    over = cv2.GaussianBlur(over, None, 1)
#    l = Lock()
#    added = Array(c_float, over.ravel(), lock=l)
    added = np.zeros((3,)+np.shape(over))
    #added = over.copy()
    added[2] += over
    color = float(np.mean(over))
    position_list = []
    correlation_list = []
    
    #find position of first frame
    name = matched_frames[0]
    im = np.array(cv2.imread(dirpath+name, -1))
    shape_im = np.shape(im)
    scale = (size_frames/float(shape_im[0])/(size_overview/float(shape_over[0])))
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(over[int(area_first_frame[0]*shape_over[0]):int(area_first_frame[1]*shape_over[0]),
                                    int(area_first_frame[2]*shape_over[1]):int(area_first_frame[3]*shape_over[1])],
                                    im, method=cv2.TM_CCOEFF_NORMED)

    maxi = tuple(np.array(np.unravel_index(np.argmax(result), result.shape), dtype='int') + 
                 np.array((area_first_frame[0]*shape_over[0], area_first_frame[2]*shape_over[1]), dtype='int'))
    correlation_list.append(np.amax(result))
    position_list.append((int(name[0:4]), ) + maxi)
    added[1,maxi[0]:maxi[0]+im.shape[0], maxi[1]:maxi[1]+im.shape[1]] += im
    cv2.putText(added[0], str(int(name[0:4])), (maxi[1]-4,maxi[0]-2), cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
    
    counter = 0
    for name in matched_frames[1:]:
        im = np.array(cv2.imread(dirpath+name, -1))#*3
        shape_im = np.shape(im)
        scale = (size_frames/float(shape_im[0])/(size_overview/float(shape_over[0])))
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if int(name[0:4])%number_frames[0] == 0 and int(name[0:4]) != 0:
            subarray = np.array(( position_list[counter-number_frames[0]+1][1]-int(tolerance_within_columns[0]*scale*shape_im[0]),
                                  position_list[counter-number_frames[0]+1][1]+int(tolerance_within_columns[1]*scale*shape_im[0]),
                                  position_list[counter-number_frames[0]+1][2]-int(tolerance_within_columns[2]*scale*shape_im[0]),
                                  position_list[counter-number_frames[0]+1][2]+int(tolerance_within_columns[3]*scale*shape_im[0]) ))
        else:
            subarray = np.array(( position_list[counter][1]-int(tolerance_within_rows[0]*scale*shape_im[0]),
                                  position_list[counter][1]+int(tolerance_within_rows[1]*scale*shape_im[0]),
                                  position_list[counter][2]-int(tolerance_within_rows[2]*scale*shape_im[0]),
                                  position_list[counter][2]+int(tolerance_within_rows[3]*scale*shape_im[0]) ))
        subarray[subarray>= shape_over[0]] = shape_over[0]-1
        subarray[subarray< 0] = 0
        result = cv2.matchTemplate(over[subarray[0]:subarray[1],subarray[2]:subarray[3]], im, method=cv2.TM_CCOEFF_NORMED)
        maxi = tuple(np.array(np.unravel_index(np.argmax(result), result.shape), dtype='int') + np.array((subarray[0], subarray[2]), dtype='int'))
        correlation_list.append(np.amax(result))
        position_list.append((int(name[0:4]), ) + maxi)
        added[1,maxi[0]:maxi[0]+im.shape[0], maxi[1]:maxi[1]+im.shape[1]] += im
        #cv2.rectangle(added, (maxi[1],maxi[0]), (maxi[1]+im.shape[1], maxi[0]+im.shape[0]), color, thickness=2)
        cv2.putText(added[0], str(int(name[0:4])), (maxi[1]-4,maxi[0]-2), cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
        counter += 1
    
    added2 = added.copy()
    added2 = np.swapaxes(added2, 0, 2)
    added2 = np.swapaxes(added2, 0, 1)
    percentile95 = np.percentile(added, 95)
    added2[added2>percentile95] = percentile95
    added2 *= 255/percentile95
    added2 = added2.astype('uint8')
    correlation_list=np.array(correlation_list, dtype='float32').reshape((number_frames[1], number_frames[0]))
    
    np.savez( dirpath+'positions.npz',
              position_list=np.array(position_list, dtype='uint16'),
              overview_with_frames_raw=added.astype('float32'),
              overview_with_frames_rgb=added2, 
              correlation_list=correlation_list )
    
    tifffile.imsave(dirpath+'positions.tif', added2)
    #cv2.imwrite(dirpath+'positions.tif', added2)
#    p = Pool(initializer=init, initargs=(l,))
#    
#    res = [p.apply_async(find_position, (name, added,over, shape_over, size_frames, size_overview, color)) for name in matched_frames[0:10]]
#    res_list = [r.get() for r in res]
#    
#    p.close()
#    p.join()
#    p.terminate()
#    
#    added = np.array(added).reshape(shape_over)