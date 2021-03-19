import numpy as np
import time
from scipy.signal import find_peaks

tscale = (8.0/4096.0)

# Calculate moving avg of waveform
def wfm_convolve(data, window, avg=False):
    # Assumes data is already baseline-subtracted
    weights = np.repeat(1.0, window)
    if avg: weights /= window  # do avg instead of sum
    return np.convolve(data, weights, 'same')

# Keep only the elements corresponding to indices in peak_ind_keep
# Note that it annoyingly can't edit the values in peaks, etc. directly since numpy arrays are fixed-length
# and just doing the variable reassignment doesn't change the passed array
def cull_peaks(peaks, peak_conv_heights, properties, peak_ind_keep):
    peaks = peaks[peak_ind_keep]
    peak_conv_heights = peak_conv_heights[peak_ind_keep]
    for key in properties.keys(): # also remove from properties dictionary
        properties[key] = properties[key][peak_ind_keep]
    return peaks, peak_conv_heights, properties

def pulse_bounds_area(data, t_min, window, start_frac, end_frac):
    # Assumes data is already baseline-subtracted
    start_pos = -1
    end_pos = -1
    min_search = np.maximum(0, t_min)
    max_search = np.minimum(len(data) - 1, t_min + window)
    cdf = np.cumsum(data[min_search:max_search])/np.sum(data[min_search:max_search])

    # find first index where cdf > start_frac, last index where cdf < end_frac
    start_pos = np.where(cdf>start_frac)[0][0] + min_search
    end_pos = np.where(cdf<(1-end_frac))[0][-1] + min_search

    return (start_pos, end_pos)

def pulse_bounds(data, t_min, window, start_frac, end_frac):
    # Assumes data is already baseline-subtracted
    chA_spe_size = 29.02 # TODO: decide where these params come from; separate stored file?
    start_pos = -1
    end_pos = -1
    min_search = np.maximum(0, t_min)
    max_search = np.minimum(len(data) - 1, t_min + window)
    peak_val = np.max(data[min_search:max_search])
    peak_pos=np.argmax(data[min_search:max_search])+min_search
    # TODO: instead of a for loop, compare full array against max(...), take first, last values from np.where (add min_search!)
    start_thresh = max(peak_val * start_frac, 2.0 / chA_spe_size)
    below_start_thresh = data[min_search:peak_pos] <= start_thresh
    # find last instance before peak below threshold
    if np.any(below_start_thresh):
        start_pos = np.where(below_start_thresh)[0][-1] + min_search
    else:
        start_pos = min_search # if no points are below threshold, use window start

    # for i_start in range(min_search, max_search):
    #     if data[i_start] > max(peak_val * start_frac, 8.0 / chA_spe_size):
    #         start_pos = i_start
    #         break

    end_thresh = max(peak_val * end_frac, 2.0 / chA_spe_size)
    below_end_thresh = data[peak_pos:max_search] <= end_thresh
    # find first instance after peak below threshold
    if np.any(below_end_thresh):
        end_pos = np.where(below_end_thresh)[0][0] + peak_pos
    else:
        end_pos = max_search # if no points are below threshold, use window end

    # for i_start in range(max_search, min_search, -1):
    #     if data[i_start] > max(peak_val * end_frac, 8.0 / chA_spe_size):
    #         end_pos = i_start
    #         break

    return (start_pos, end_pos)


# function to find start and end of all pulses in a given event
# inputs:
#   waveforms_bls: baseline subtracted waveform numpy array for this event only
#   max_pulses: max pulses to find; output lengths will never exceed this number
# ouputs:
#   start_times, end_times: variable-length arrays of pulse start and end times
#   peaks: locations of peaks in waveform, in samples
#   data_conv: convolved waveform; used for debugging
#   properties: list of properties from scipy's peak finder; used for debugging/plotting
def findPulses(waveform_bls, max_pulses, SPEMode=False):

    if SPEMode:
        pulse_window = int(12.0 / tscale) # was 7 us; any reason this can't just always go to next pulse or end of wfm?
        conv_width = 150 #int(0.3 / tscale) # in samples
        min_height = 0.15 # mV
        min_dist = int(0.5 / tscale) # in samples
        bounds_conv_width = 5 # in samples
        pulse_start_frac = 0.01  # pulse starts at this fraction of peak height above baseline
        pulse_end_frac = 0.01  # pulse starts at this fraction of peak height above baseline       

    # pulse finder parameters for tuning
    else:
        pulse_window = int(12.0 / tscale) # was 7 us; any reason this can't just always go to next pulse or end of wfm?
        conv_width = 100 #int(0.3 / tscale) # in samples
        min_height = 0.10 # phd/sample
        min_dist = int(0.5 / tscale) # in samples
        bounds_conv_width = 5 # in samples
        pulse_start_frac = 0.01  # pulse starts at this fraction of peak height above baseline
        pulse_end_frac = 0.01  # pulse starts at this fraction of peak height above baseline

    # Do a moving average (sum) of the waveform
    # Look for local maxima using scipy's find_peaks
    #
    t0 = time.time()
    data_conv = wfm_convolve(waveform_bls, conv_width, avg=True)
    peaks, properties = find_peaks(data_conv, distance=min_dist, height=min_height,
                                   width=int(0.01 / tscale), prominence=0.1)  # could restrict search if desired
    if len(peaks)<1 and not SPEMode: return [],[],[],[],[]
    peak_conv_heights = data_conv[peaks]

    # only keep the largest max_pulses peaks
    peak_height_order = np.argsort(peak_conv_heights)
    lg_peaks = np.sort(peak_height_order[:max_pulses])
    # peak_ind_cut = [ii for ii in range(len(peaks)) if not ii in lg_peaks]
    peaks, peak_conv_heights, properties = cull_peaks(peaks, peak_conv_heights, properties, lg_peaks)

    # Mark peaks that should be removed (merged w/ previous ones):
    # If a peak has small prominence relative to previous peak height w/in some window, remove it
    # Aimed at removing S2 falling tails
    merge_frac = 0.05  # Initial testing suggests this is about right "by eye" (0.07 is too high, 0.02 is too low)
    time_diffs = (peaks[1:] - peaks[:-1])
    small_peak = properties['prominences'][1:] < peak_conv_heights[:-1] * merge_frac
    prev_peak_near = time_diffs < int(pulse_window / 2)  # was [:-1]...
    peak_ind_cut = np.where(small_peak * prev_peak_near)[0] + 1  # add 1 since we're always looking back one pulse

    # Remove peaks that were marked to be cut
    if len(peak_ind_cut) > 0 and True:
        # print("remove peaks: ", peak_ind_cut)
        # print("Prominences: ", properties["prominences"])
        # print("prominence ratios: ", properties['prominences'][1:][peak_ind_cut-1]/(peak_conv_heights[:-1][peak_ind_cut-1]))
        peak_ind_keep = [ii for ii in range(len(peaks)) if not ii in peak_ind_cut]
        peaks, peak_conv_heights, properties = cull_peaks(peaks, peak_conv_heights, properties, peak_ind_keep)

    # Define windows to search for peak boundaries (very wide, by definition; this mainly deals w/ overlapping peaks)
    window_starts = []
    window_ends = []
    prev_end = -1
    for peak_ind in range(len(peaks)):
        # Peak start calc
        if prev_end != -1:  # Use previous peak's end as this peak's start, if within window
            window_starts.append(prev_end)
        else:
            # look back half the max window; could also shift according to int(conv_width/2), probably not important
            window_starts.append(peaks[peak_ind] - int(pulse_window / 2))

        # Peak end calc
        if peak_ind == len(peaks) - 1:
            window_ends.append(peaks[peak_ind] + int(pulse_window / 2))  # look forward half the
        else:
            # If the next pulse is overlapping, set the pulse boundary as the minimum between the two
            if time_diffs[peak_ind] < pulse_window:
                valley_time = np.argmin(waveform_bls[peaks[peak_ind]:peaks[peak_ind + 1]]) + peaks[peak_ind]
                window_ends.append(valley_time)
                prev_end = valley_time
            else:
                window_ends.append(peaks[peak_ind] + int(pulse_window / 2))
                prev_end = -1

    #t1 = time.time()
    #print(t1-t0)

    # Find peak start, end times
    pulse_start_times = []
    pulse_end_times = []
    data_bounds_conv = wfm_convolve(waveform_bls, bounds_conv_width, avg=True) # use somewhat-smoothed wfm to find bounds
    for peak_ind in range(len(peaks)):
        #start_time, end_time = pulse_bounds_area(waveform_bls, window_starts[peak_ind], window_ends[peak_ind]-window_starts[peak_ind], pulse_start_frac,
        #                                        pulse_end_frac)
        start_time, end_time = pulse_bounds(data_bounds_conv, window_starts[peak_ind], window_ends[peak_ind]-window_starts[peak_ind], pulse_start_frac,
                                                pulse_end_frac)
        pulse_start_times.append(start_time)
        pulse_end_times.append(end_time)


    return pulse_start_times, pulse_end_times, peaks, data_conv, properties
