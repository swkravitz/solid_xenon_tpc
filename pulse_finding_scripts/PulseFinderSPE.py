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
    pulse_start_frac = 0.4  # pulse starts at this fraction of peak height above baseline
#    pulse_end_frac = 0.1  # pulse starts at this fraction of peak height above baseline       
    pulse_end_frac = 0.2  # pulse starts at this fraction of peak height above baseline       
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
    if start_times>end_times:
        return 0,0,0,data_conv
    
    return start_times, end_times, peaks, data_conv



def findBase1(data):
    size = len(data)

    # Calculate baseline in three places: beginning, middle, end. Choose the lowest value
    base_win = int(0.2/tscale) # 0.2 us
    base_1 = np.zeros(3)
    base_1[0] = np.mean(data[0:base_win])
    base_1[1] = np.mean(data[int(size/2):int(size/2)+base_win])
    base_1[2] = np.mean(data[-1-base_win:-1])
    base_1_min = np.mean(base_1)

    return base_1_min


def findDarkSPEs(data):
    base_win = int(0.2/tscale) # 0.2 us

    # Do the initial pulse finding

    conv_width = 60 # samples
    min_height = 0.09 # mV 
    min_width = 10 # samples
    rel_height = 0.10 # percent of pulse height to look at width, from the top
    pulse_start_frac = 0.5  # pulse starts at this fraction of peak height above baseline
    pulse_end_frac = 0.1  # pulse starts at this fraction of peak height above baseline       

    data_conv = wfm_convolve(data, conv_width, avg=True)

    old_peaks, properties = find_peaks(data_conv, height=min_height, width=min_width, rel_height=rel_height)
    old_peaks = sorted(old_peaks)
    # On to second sweep of pulse finding
    peaks = np.zeros(len(old_peaks), dtype=np.int)
    if len(old_peaks) == 0:
        baselines = np.array([np.mean(data[0:base_win])])
        baselines_start = np.array([0])
        baselines_end = np.array([base_win])
    else:
        baselines = np.zeros(len(old_peaks))
        baselines_start = np.zeros(len(old_peaks))
        baselines_end = np.zeros(len(old_peaks))
    start_times = np.zeros(len(old_peaks), dtype=np.int)
    end_times = np.zeros(len(old_peaks), dtype=np.int)
    peak_data = 0

    l_sample = 40
    l_p_bound = l_sample + base_win # Stops 40 samples + baseline before
    r_p_bound = int(0.3/tscale) # Stops .3 us after
    data_conv2 = np.zeros(len(data_conv))
    for i in range(len(old_peaks)):
        # Check to see if peak is too close to start/end or to previous peak
        if old_peaks[i] - l_p_bound < 1 or old_peaks[i] + r_p_bound > len(data):
#            print ("peaks out of bounds!")
            continue
        elif i> 0 and abs(old_peaks[i] - old_peaks[i-1]) < r_p_bound:
            continue
#            print ("peaks too close: %d ns"%(abs(old_peaks[i] - old_peaks[i-1])*tscale*1000))
        # Divides waveform between peaks found and finds pulses
        else:
            peak_data = data[old_peaks[i] - l_p_bound:old_peaks[i] + r_p_bound] 
            baselines[i] = np.mean(data[old_peaks[i]-l_p_bound:old_peaks[i]-l_sample])
            baselines_start[i] = int(old_peaks[i]-l_p_bound)
            baselines_end[i] = int(old_peaks[i]-l_sample)
            peak_data = peak_data-baselines[i]
                
            start_times[i], end_times[i], peaks[i], data_conv_led = findLEDSPEs(peak_data)
            data_conv2[old_peaks[i] - l_p_bound:old_peaks[i] + r_p_bound] = data_conv_led 
            # Shift times to proper position
            start_times[i] += old_peaks[i] - l_p_bound
            end_times[i] += old_peaks[i] - l_p_bound
            peaks[i] += old_peaks[i] - l_p_bound
    return start_times, end_times, peaks, baselines, data_conv2, baselines_start, baselines_end

