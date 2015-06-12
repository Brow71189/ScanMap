# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:33:46 2015

@author: mittelberger
"""

import numpy as np
import cv2
import os
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
    
    dirpath = '/3tb/maps_data/map_2015_05_29_12_01/'
    
    overview = '/3tb/maps_data/map_2015_05_29_12_01/Overview_898.156649193_nm.tif'
    
    size_overview = 898.156649193 #nm
    size_frames = 64 #nm
    
    
    
    frames = os.listdir(dirpath)
    matched_frames = []
    for name in frames:
        try:
            int(name[0:4])
        except:
            pass
        else:
            matched_frames.append(name)
            
    
    matched_frames.sort()
    over = np.array(cv2.imread(overview, -1))#[1023:3072, 0:2048]*3
    shape_over = np.shape(over)
    over = cv2.GaussianBlur(over, None, 1)
#    l = Lock()
#    added = Array(c_float, over.ravel(), lock=l)
    added = over.copy()
    color = 1
    
    for name in matched_frames:
        im = np.array(cv2.imread(dirpath+name, -1))#*3
        shape_im = np.shape(im)
        scale = (size_frames/float(shape_im[0])/(size_overview/float(shape_over[0])))
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(over, im, method=cv2.TM_CCOEFF_NORMED)
        maxi = np.unravel_index(np.argmax(result), result.shape)
        added[maxi[0]:maxi[0]+im.shape[0], maxi[1]:maxi[1]+im.shape[1]] += im
        #cv2.rectangle(added, (maxi[1],maxi[0]), (maxi[1]+im.shape[1], maxi[0]+im.shape[0]), color, thickness=2)
        cv2.putText(added, name[0:4], (maxi[1]-4,maxi[0]-2), cv2.FONT_HERSHEY_PLAIN, 3, color, thickness=2)
    
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