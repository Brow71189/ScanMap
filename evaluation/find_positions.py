# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:33:46 2015

@author: mittelberger
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


dirpath = '/3tb/maps_data/map_2015_04_15_13_13/'

overview = '/3tb/maps_data/map_2015_04_15_13_13/overview.tif'

size_overview = 512 #nm
size_frames = 12 #nm



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
over = cv2.imread(overview, -1)
shape_over = np.shape(over)
over = cv2.GaussianBlur(over, None, 1)
added = over.copy()

for name in matched_frames[0:10]:
    im = cv2.imread(dirpath+name, -1)
    shape_im = np.shape(im)
    scale = (size_frames/float(shape_im[0])/(size_overview/float(shape_over[0])))
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    result = cv2.matchTemplate(over, im, method=cv2.TM_SQDIFF_NORMED)
    maxi = np.unravel_index(np.argmin(result), result.shape)
    added[maxi[0]:maxi[0]+im.shape[0], maxi[1]:maxi[1]+im.shape[1]] += im
    cv2.rectangle(added, (maxi[1],maxi[0]), (maxi[1]+im.shape[1], maxi[0]+im.shape[0]), 1, thickness=3)
    cv2.putText(added, name, (maxi[1],maxi[0]), cv2.FONT_HERSHEY_PLAIN, 3, 1, thickness=2)