#!python
#cython: boundscheck=False
#cython: cdivision=True
"""
Created on Fri Jun 16 12:04:48 2017

@author: mittelberger2
"""

import numpy
from libc.math cimport round, atan, log, exp, pow, sqrt
from libc.stdlib cimport rand, RAND_MAX

DEF factorials_table_length = 20
DEF integration_range_gamma = 2.0
DEF pi = 3.14159

cdef long factorials_table[factorials_table_length]

def electron_counting(image, baseline=0.002, countlevel=0.01, peakwidth=1, only_integrate=False):
    cdef float[:, :] c_image = image
    result = numpy.zeros(image.shape, dtype=numpy.float32)
    cdef float[:, :] c_result = result
    cdef float c_baseline = baseline
    cdef float c_countlevel = countlevel
    cdef float c_peakwidth = peakwidth
    cdef int c_height = image.shape[0]
    cdef int c_width = image.shape[1]
    cdef bint c_only_integrate = only_integrate
    c_electron_counting(c_image, c_result, c_height, c_width, c_baseline, c_countlevel, c_peakwidth, c_only_integrate)
    return result

def find_most_likely_counts(image, baseline=0.0, countlevel=0.1):
    cdef float[:, :] c_image = image
    result = numpy.zeros(image.shape, dtype=numpy.float32)
    cdef float[:, :] c_result = result
    cdef float c_baseline = baseline
    cdef float c_countlevel = countlevel
    cdef int c_height = image.shape[0]
    cdef int c_width = image.shape[1]
    c_find_most_likely_counts_image(image, c_result, c_height, c_width, c_baseline, c_countlevel)
    return result

cdef void c_electron_counting(float[:, :] image, float[:, :] result, int height, int width, float baseline,
                              float countlevel, float peakwidth, bint only_integrate):
    global factorials_table
    cdef int k, i
    cdef float integral
    cdef int index
    cdef int randnum
    cdef float threshold
    cdef float counts_divisor
    cdef float peak_max

    create_factorials_table(factorials_table_length)
    # All following calculations are done with the height of the Lorentzians normalized to 1
    threshold = 1/(1.0+pow(integration_range_gamma, 2)) # height of our peak at +- integration_range_gamma
    # Integral of our peak from -integration_range_gamma to +integration_range_gamma
    counts_divisor = countlevel*peakwidth*(atan(integration_range_gamma)-atan(-1.0*integration_range_gamma))
    # Integration range going from -integration_range_gamma to +integration_range_gamma
    peaklength = <int>(round(2.0*integration_range_gamma*peakwidth))
    # correct integral for baseline
    counts_divisor -= baseline*(<float>peaklength)

    for k in range(height):
        integral = 0
        index = -1
        i = 0
        peak_max = countlevel
        while i < width:
            if index != -1 and (image[k, i] - baseline <= peak_max*threshold or i - index >= peaklength or i == width-1):
                if only_integrate:
                    result[k, index] = integral
                else:
                    result[k, index] = c_find_most_likely_counts(integral/counts_divisor)
                integral = 0
                if image[k, i] - baseline > peak_max*threshold:
                    randnum = <int>(<float>rand() / <float>RAND_MAX * <float>peaklength)
                    i = index + 1 + randnum
                    index = i
                    integral = image[k, i]
                else:
                    index = -1
                peak_max = countlevel

            elif index != -1:
                integral += image[k, i] - baseline
                if image[k, i] > peak_max:
                    peak_max = image[k, i]

            if index == -1 and image[k, i] - baseline > peak_max*threshold:
                index = i
                integral = image[k, i]

            i += 1

cdef void c_find_most_likely_counts_image(float[:, :] image, float[:, :] result, int height, int width,
                                          float baseline, float countlevel):
    cdef int k, i
    create_factorials_table(factorials_table_length)
    for k in range(height):
        for i in range(width):
            result[k, i] = c_find_most_likely_counts((image[k ,i]-baseline)/countlevel)

cdef float c_count_probability(int counts, float x):
    return ((1/sqrt(2.0*pi*(<float>counts)**2)*exp(-1.0*pow(x - <float>counts, 2.0)/(2.0*(<float>counts)**2))) *
            (pow((<float>counts), <float>counts)*exp(-1.0*(<float>counts))/(<float>factorials_table[counts])))

cdef float c_find_most_likely_counts(float x):
    if round(x+0.25) == 0.0:
        return 0.0
    cdef int counter
    cdef float max_probability, current_probability
    max_probability = 0
    counter = 0
    while counter < factorials_table_length:
        current_probability = c_count_probability(counter, x)
        if current_probability < max_probability:
            return counter - 1
        max_probability = current_probability
        counter += 1
    return factorials_table_length

cdef void create_factorials_table(int length):
    cdef int counter
    factorials_table[0] = 1
    counter = 1
    while counter < length:
        factorials_table[counter] = factorials_table[counter-1] * <long>counter
        counter += 1