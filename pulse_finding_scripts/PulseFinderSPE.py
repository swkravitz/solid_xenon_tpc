import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from scipy.signal import find_peaks

tscale = (8.0/4096.0)


# Calculate moving avg of waveform
def wfm_convolve(data, window, avg=False):
    # Assumes data is already baseline-subtracted
    weights = np.repeat(1.0, window)
    if avg: weights /= window  # do avg instead of sum
    return np.convolve(data, weights, 'same')


# Takes a sum of waveform, with conv option, in a certain window for finding pulse area
def simpleSumArea(data, window, convolve=False):

    if convolve:
        conv_width = 40
        fixed_data = conv_data = wfm_convolve(data,conv_width, avg=True)
    else:
        fixed_data = data

    p_sarea = np.sum(fixed_data[window[0]:window[1]])
    return p_sarea


# Finds pulse area using find_peaks
def findLEDSPEs(data):

    conv_width = 40 # samples
    min_height = 0.06 # mV 
    min_width = 5 # samples
    rel_height = 0.10 # percent of pulse height to look at width, from the top
    pulse_start_frac = 0.5  # pulse starts at this fraction of peak height above baseline
    pulse_end_frac = 0  # pulse starts at this fraction of peak height above baseline       

    data_conv = wfm_convolve(data, conv_width, avg=True)

    peaks, properties = find_peaks(data_conv, height=min_height, width=min_width, rel_height=rel_height)
    
    # Restrict to one event
    if len(peaks) < 1:
        return 0,0,0,data_conv
    else:
        peaks = peaks[0]

    # Find start/end times
    # For loop is slow, maybe do something fancier
    start_times = 0
    end_times = 0
    start_flag = True
    end_flag = True
    for i in range(len(data)-peaks):

        if data_conv[peaks-i] < pulse_start_frac*data_conv[peaks] and start_flag: 
            start_times = peaks-i
            start_flag = False

        if data_conv[peaks+i] < pulse_end_frac*data_conv[peaks] and end_flag: 
            end_times = peaks+i
            end_flag = False
    
    
    return start_times, end_times, peaks, data_conv


