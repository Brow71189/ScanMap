# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:01:33 2015

@author: mittelberger
"""

import numpy as np
import tifffile
import autotune as at
import os

foci = [0,1,2,3]
sizes = [19,20,21]
rotations = [5,6,7]
backgrounds = [1,2,3,4]

pixels = 2048

number_frames = 10

savepath = '/home/mittelberger/Documents/simulated_test_series/'

aberrations = {'EHTFocus': 0.0, 'C12_a': -0.5, 'C12_b': -1.5, 'C21_a': 380.0, 'C21_b': 30.0, 'C23_a': 200.0, 'C23_b': 50.0}



if not os.path.exists(savepath):
    os.makedirs(savepath)

config_file = open(savepath+'frame_infos.txt', mode='w+')

config_file.write('#Frame parameters for simulated image series:\n')
config_file.write('#number\tsize\trotation\tbackground\taberrations\n\n\n\n\n')

for i in range(number_frames):
    focus = foci[int(np.rint(np.random.random_sample()*(len(foci)-1)))]
    size = sizes[int(np.rint(np.random.random_sample()*(len(sizes)-1)))]
    rotation = rotations[int(np.rint(np.random.random_sample()*(len(rotations)-1)))]
    background = backgrounds[int(np.rint(np.random.random_sample()*(len(backgrounds)-1)))]
    
    image = at.graphene_generator(size, 2048, rotation)*(50+background)+background
    
    aberrations['EHTFocus'] = focus
    
    config_file.write('%.4d\t%.2f\t%.2f\t%.2f\t%s\n' % (i,size,rotation,background,str(aberrations)))
    aberrated = at.image_grabber(image=image, imsize=size, relative_aberrations=False, **aberrations)
    
    tifffile.imsave( savepath+str('%.4d_test_series.tif' % (i,)), aberrated.astype('uint16') )
    
config_file.close()
    
    