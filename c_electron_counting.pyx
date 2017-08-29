#!python
#cython: boundscheck=False
#cython: cdivision=True
"""
Created on Fri Jun 16 12:04:48 2017

@author: mittelberger2
"""

import numpy
from libc.math cimport round
from libc.stdlib cimport rand, RAND_MAX

def electron_counting(image, baseline=0.002, countlevel=0.01, peaklength=5):
    cdef float[:, :] c_image = image
    result = numpy.zeros(image.shape, dtype=numpy.uint16)
    cdef unsigned short[:, :] c_result = result
    cdef float c_baseline = baseline
    cdef float c_countlevel = countlevel
    cdef int c_peaklength = peaklength
    cdef int c_height = image.shape[0]
    cdef int c_width = image.shape[1]
    c_electron_counting(c_image, c_result, c_height, c_width, c_baseline, c_countlevel, c_peaklength)
    return result

cdef void c_electron_counting(float[:, :] image, unsigned short[:, :] result, int height, int width, float baseline,
                              float countlevel, short peaklength):
    cdef unsigned int k, i
    cdef float integral
    cdef int index
    cdef int randnum

    for k in range(height):
        integral = 0
        index = -1
        i = 0
        while i < width:
            if index != -1 and (image[k, i] < countlevel/2 or i - index >= peaklength):
                integral /= i - index - 1
                result[k, index] = int(round(integral/countlevel))
                integral = 0
                if image[k, i] >= countlevel/2:
                    randnum = int(float(rand())/float(RAND_MAX)*peaklength)
                    i = index + 1 + randnum
                    index = i
                else:
                    index = -1
            if image[k, i] >= countlevel/2:
                if index == -1:
                    if i > 0:
                        index = i - 1
                    else:
                        index = i
                    integral = 0
                integral += image[k, i]
            i += 1