import numpy as np
# import pylab as pl
import matplotlib.pyplot as pl
import matplotlib as mpl
import scipy
from scipy.signal import find_peaks
from scipy.signal import argrelmin
import time


def smooth(data, t_min, window):
    avg_array = np.zeros(len(data))
    # print(avg_array)
    m = 20
    for i in range(t_min, t_min + window):
        min_ind = i - m
        max_ind = i + m
        if min_ind < 0: min_ind = 0
        if max_ind > len(data): max_ind = len(data)
        avg_array[i] = (np.mean(data[min_ind:max_ind]))
    return (avg_array)


# def maxima(data):
# y = data
# peaks = argrelextrema(y, np.greater)
# return(peaks)
# def minima(data):
# z = data
# valleys = argrelextrema(z, np.less)
# return(valleys)

def pulse_finder_area_series(data, t_min_search, t_max_search, window):
    # Assumes data is already baseline-subtracted
    max_area = -1
    max_ind = -1
    for i_start in range(t_min_search, t_max_search):
        area = np.sum(data[i_start:i_start + window])
        if area > max_area:
            max_area = area
            max_ind = i_start
    return (max_ind, max_area)


def wfm_convolve(data, window, avg=False):
    # Assumes data is already baseline-subtracted
    weights = np.repeat(1.0, window)
    if avg: weights /= window  # do avg instead of sum
    return np.convolve(data, weights, 'same')


def pulse_finder_area(data, t_min_search, t_max_search, window):
    # Assumes data is already baseline-subtracted
    if t_max_search < t_min_search + 1:
        return (-1, -1)

    data_conv = wfm_convolve(data, window)
    # Search only w/in search range, offset so that max_ind is the start of the window rather than the center
    max_ind = np.argmax(data_conv[int(t_min_search + window / 2):int(t_max_search + window / 2)]) + int(t_min_search)
    return (max_ind, data_conv[max_ind + int(window / 2)])

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
    start_pos = -1
    end_pos = -1
    min_search = np.maximum(0, t_min)
    max_search = np.minimum(len(data) - 1, t_min + window)
    peak_val = np.max(data[min_search:max_search])
    peak_pos = np.argmax(data[min_search:max_search])
    # print("peak_val: ",peak_val,"peak_pos: ",(peak_pos+min_search)*tscale)
    # print("min_search: ",min_search*tscale,"max_search: ",max_search*tscale)
    # start_frac: pulse starts at this fraction of peak height above baseline
    # TODO: instead of a for loop, compare full array against max(...), take first, last values from np.where (add min_search!)
    for i_start in range(min_search, max_search):
        if data[i_start] > max(peak_val * start_frac, 8.0 / chA_spe_size):
            start_pos = i_start
            break
    # end_frac: pulse ends at this fraction of peak height above baseline
    for i_start in range(max_search, min_search, -1):
        if data[i_start] > max(peak_val * end_frac, 8.0 / chA_spe_size):
            end_pos = i_start
            break

    return (start_pos, end_pos)


def merged_bounds(data, t_min, window, start_frac, end_frac):
    start_pos = -1
    end_pos = -1
    a = (np.diff(np.sign(np.diff(output))) > 0).nonzero()[0] + 1
    b = argrelmin(data)
    peak_v = np.max(data[t_min:t_min + window])
    print("b:", b)
    second = []
    print("a[0]:", a[0])
    for i in scipy.nditer(b):
        # if data[i]>0.2 and max(data[t_min:t_min+window]) < i :
        # for i_start in range(t_min,t_min+window):
        # if data[i_start]>max(data[a[0]]*start_frac,4.5/chA_spe_size):
        # start_pos=i_start
        # break
        if data[i] > 1:
            start_pos = i
            print("i:", i)
            break

    for j in a:

        if data[j] > 0.2:
            second.append(j)
    for k in scipy.nditer(b):
        if max(a) > k:

            for z in second[1:]:

                # end_frac: pulse ends at this fraction of peak height above baseline
                for i_start in range(t_min + window, t_min, -1):
                    if data[i_start] > max(data[z] * end_frac, 4.5 / chA_spe_size):
                        end_pos = i_start
                        break
        else:
            end_pos = b[0]

    return (start_pos, end_pos)

# Keep only the elements corresponding to indices in peak_ind_keep
# Note that it annoyingly can't edit the values in peaks, etc. directly since numpy arrays are fixed-length
# and just doing the variable reassignment doesn't change the passed array
def cull_peaks(peaks, peak_conv_heights, properties, peak_ind_keep):
    peaks = peaks[peak_ind_keep]
    peak_conv_heights = peak_conv_heights[peak_ind_keep]
    for key in properties.keys(): # also remove from properties dictionary
        properties[key] = properties[key][peak_ind_keep]
    return peaks, peak_conv_heights, properties

# set plotting style
# mpl.rcParams['font.size']=28
# mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout'] = True
# mpl.rcParams['figure.figsize']=[16.0,12.0]


data_dir = "C:/Users/swkra/Desktop/Jupyter temp/data-202009/091720/bkg_3.5g_3.9c_27mV_7_postrecover2_5min/"  # "../../092420/bkg_3.5g_3.9c_1.0bara_25mV_fan_in_4_allchanOR_post30sfill_5min/"
event_window = 25.  # in us
wsize = int(500 * event_window)  # samples per waveform # 12500 for 25 us
max_evts = 1000  # 25000 # -1 means read in all entries; 25000 is roughly the max allowed in memory on the DAQ computer
max_pts = -1  # do not change
if max_evts > 0:
    max_pts = wsize * max_evts
channel_0 = np.fromfile(data_dir + "wave0.dat", dtype="int16", count=max_pts)
channel_1 = np.fromfile(data_dir + "wave1.dat", dtype="int16", count=max_pts)
channel_2 = np.fromfile(data_dir + "wave2.dat", dtype="int16", count=max_pts)
channel_3 = np.fromfile(data_dir + "wave3.dat", dtype="int16", count=max_pts)
channel_4 = np.fromfile(data_dir + "wave4.dat", dtype="int16", count=max_pts)
channel_5 = np.fromfile(data_dir + "wave5.dat", dtype="int16", count=max_pts)
channel_6 = np.fromfile(data_dir + "wave6.dat", dtype="int16", count=max_pts)
channel_7 = np.fromfile(data_dir + "wave7.dat", dtype="int16", count=max_pts)

# channel_0=np.fromfile("../../Desktop/crystallize_data/t3-0805/A-thorium-4kv-t3.dat", dtype="int16")
# channel_1=np.fromfile("../../Desktop/crystallize_data/t3-0805/B-thorium-4kv-t3.dat", dtype="int16")
# channel_2=np.fromfile("../../Desktop/crystallize_data/t3-0805/C-thorium-4kv-t3.dat", dtype="int16")
# channel_3=np.fromfile("../../Desktop/crystallize_data/t3-0805/D-thorium-4kv-t3.dat", dtype="int16")

# channel_0=np.fromfile("A-thorium-3kv.dat", dtype="int16")
# channel_1=np.fromfile("B-thorium-3kv.dat", dtype="int16")
# channel_2=np.fromfile("C-thorium-3kv.dat", dtype="int16")
# channel_3=np.fromfile("D-thorium-3kv.dat", dtype="int16")
timer_start = time.time()

vscale = (2000.0 / 16384.0)
chA_spe_size = 29.02
V = vscale * channel_0 / chA_spe_size  # ch A, calib size 644
# Ensure we have an integer number of events
V = V[:int(len(V) / wsize) * wsize]
chB_spe_size = 30.61
V_1 = vscale * channel_1 / chB_spe_size
V_1 = V_1[:int(len(V) / wsize) * wsize]
chC_spe_size = 28.87
V_2 = vscale * channel_2 / chC_spe_size
V_2 = V_2[:int(len(V) / wsize) * wsize]
chD_spe_size = 28.86
V_3 = vscale * channel_3 / chD_spe_size
V_3 = V_3[:int(len(V) / wsize) * wsize]
chE_spe_size = 30.4
V_4 = vscale * channel_4 / chE_spe_size
V_4 = V_4[:int(len(V) / wsize) * wsize]
chF_spe_size = 30.44
V_5 = vscale * channel_5 / chF_spe_size
V_5 = V_5[:int(len(V) / wsize) * wsize]
chG_spe_size = 30.84
V_6 = vscale * channel_6 / chG_spe_size
V_6 = V_6[:int(len(V) / wsize) * wsize]
chH_spe_size = 30.3
V_7 = vscale * channel_7 / chH_spe_size
V_7 = V_7[:int(len(V) / wsize) * wsize]
n_channels = 9  # including sum
v_matrix = V.reshape(int(V.size / wsize), wsize)
v1_matrix = V_1.reshape(int(V.size / wsize), wsize)
v2_matrix = V_2.reshape(int(V.size / wsize), wsize)
v3_matrix = V_3.reshape(int(V.size / wsize), wsize)
v4_matrix = V_4.reshape(int(V.size / wsize), wsize)
v5_matrix = V_5.reshape(int(V.size / wsize), wsize)
v6_matrix = V_6.reshape(int(V.size / wsize), wsize)
v7_matrix = V_7.reshape(int(V.size / wsize), wsize)
vsum_matrix = v_matrix + v1_matrix + v2_matrix + v3_matrix + v4_matrix + v5_matrix + v6_matrix + v7_matrix
v_matrix_all_ch = [v_matrix, v1_matrix, v2_matrix, v3_matrix, v4_matrix, v5_matrix, v6_matrix, v7_matrix, vsum_matrix]
x = np.arange(0, wsize, 1)
tscale = (8.0 / 4096.0)
t = tscale * x
t_matrix = np.repeat(t[np.newaxis, :], V.size / wsize, 0)
# One entry per channel
max_ind_array = np.zeros((v_matrix.shape[0], n_channels))
max_val_array = np.zeros((v_matrix.shape[0], n_channels))
integral_array = np.zeros((v_matrix.shape[0], n_channels))
s2_integral_array = np.zeros((v_matrix.shape[0], n_channels))
s1_ch_array = np.zeros((v_matrix.shape[0], n_channels - 1))
s2_ch_array = np.zeros((v_matrix.shape[0], n_channels - 1))
# One entry per event
s2_area_array = np.zeros(v_matrix.shape[0])
s1_area_array = np.zeros(v_matrix.shape[0])
s2_width_array = np.zeros(v_matrix.shape[0])
s2_height_array = np.zeros(v_matrix.shape[0])
s1_height_array = np.zeros(v_matrix.shape[0])
t_drift_array = np.zeros(v_matrix.shape[0])
s2_found_array = np.zeros(v_matrix.shape[0], dtype='bool')
s1_found_array = np.zeros(v_matrix.shape[0], dtype='bool')

# s2_area_array=[]
# s1_area_array=[]
# s2_width_array=[]
# t_drift_array=[]
# s2_found_array=[]
# s1_found_array=[]

inn = ""
center = np.zeros(v_matrix.shape[0], dtype='bool')
print("Total events: ", v_matrix.shape[0])
for i in range(0, int(v_matrix.shape[0])):
    if i % 100 == 0: print("Event #", i)
    t0 = time.time()
    # for each channel
    # for j in range(0, n_channels):
    # i = input("Window number between 1 and " + str((V.size/wsize)) + ": ")

    # baseline=np.mean(v_matrix_all_ch[j][i,:500]) #avg ~1 us
    # print("baseline: ",baseline)

    # Look for events with S1 and S2 from summed channel
    n_top = int((n_channels - 1) / 2)
    top_channels = np.array(range(n_top), int)
    bottom_channels = np.array(range(n_top, 2 * n_top), int)
    post_trigger = 0.5  # Was 0.2 for data before 11/22/19
    trigger_time_us = event_window * (1 - post_trigger)
    trigger_time = int(trigger_time_us / tscale)
    t_offset = int(0.2 / tscale)
    s1_window = int(0.5 / tscale)
    s2_window = int(7.0 / tscale)
    pulse_window = int(7.0 / tscale)
    max_pulses = 4
    t_min_search = trigger_time - int(event_window / 2.1 / tscale)  # was int(10./tscale)
    t_max_search = trigger_time + int(event_window / 2.1 / tscale)  # was int(22./tscale)
    s1_thresh = 100 / chA_spe_size
    s1_range_thresh = 10 / chA_spe_size
    s2_thresh = 150 / chA_spe_size
    s2_start_frac = 0.01  # s2 pulse starts at this fraction of peak height above baseline
    s2_end_frac = 0.01  # s2 pulse starts at this fraction of peak height above baseline
    s1_start_frac = 0.1
    s1_end_frac = 0.1
    s1_max = s1_thresh
    s1_max_ind = -1
    s1_area = -1
    s1_height_range = -1
    s1_start_pos = -1
    s1_end_pos = -1
    s2_max = s2_thresh
    s2_max_ind = -1
    s2_area = -1
    # s2_start_pos=[]
    # s2_end_pos=[]
    s2_width = -1
    s2_height = -1
    s1_height = -1
    t_drift = -1
    s1_found = False
    s2_found = False
    s1_ch_area = [-1] * (n_channels - 1)
    s2_ch_area = [-1] * (n_channels - 1)
    fiducial = False

    baseline_start = int(0. / tscale)
    baseline_end = int(2. / tscale)
    t0 = time.time()
    sum_baseline = np.mean(v_matrix_all_ch[-1][i, baseline_start:baseline_end])  # avg ~us, avoiding trigger
    baselines = [np.mean(ch_j[i, baseline_start:baseline_end]) for ch_j in v_matrix_all_ch]
    # print("baseline:", baselines)
    # print("sum baseline:", sum_baseline)
    sum_data = v_matrix_all_ch[-1][i, :] - sum_baseline
    ch_data = [ch_j[i, :] - baseline_j for (ch_j, baseline_j) in zip(v_matrix_all_ch, baselines)]
    t0a = time.time()
    # print("ch baseline calc time: ",t0a-t0)

    # Do a moving average (sum) of the waveform with different time windows for s1, s2
    # Look for where this value is maximized
    # Look for the s2 using a moving average (sum) of the waveform over a wide window
    conv_width = int(0.3 / tscale)
    data_conv = wfm_convolve(sum_data, conv_width, avg=True)
    peaks, properties = find_peaks(data_conv, distance=int(0.5 / tscale), height=0.8,
                          width=int(0.01 / tscale), prominence=0.1)  # could restrict search if desired
    peak_conv_heights = data_conv[peaks]
    #print("peaks: ", peaks*tscale)
    #print("(total found: {0})".format(np.size(peaks)))

    # only keep the larges max_pulses peaks
    peak_height_order = np.argsort(peak_conv_heights)
    lg_peaks = np.sort(peak_height_order[:max_pulses])
    #peak_ind_cut = [ii for ii in range(len(peaks)) if not ii in lg_peaks]
    peaks, peak_conv_heights, properties = cull_peaks(peaks, peak_conv_heights, properties, lg_peaks)

    # Mark peaks that should be removed (merged w/ previous ones):
    # If a peak has small prominence relative to previous peak height w/in some window, remove it
    # Aimed at removing S2 falling tails
    merge_frac = 0.05 # Initial testing suggests this is about right "by eye" (0.07 is too high, 0.02 is too low)
    time_diffs = (peaks[1:]-peaks[:-1])
    small_peak = properties['prominences'][1:] < peak_conv_heights[:-1] * merge_frac
    prev_peak_near = time_diffs < int(pulse_window / 2) # was [:-1]...
    peak_ind_cut = np.where(small_peak*prev_peak_near)[0]+1 # add 1 since we're always looking back one pulse

    # Remove peaks that were marked to be cut
    if len(peak_ind_cut) > 0:
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
        if prev_end != -1: # Use previous peak's end as this peak's start, if within window
            window_starts.append(prev_end)
        else:
            # look back half the max window; could also shift according to int(conv_width/2), probably not important
            window_starts.append(peaks[peak_ind]-int(pulse_window/2))

        # Peak end calc
        if peak_ind == len(peaks)-1:
            window_ends.append(peaks[peak_ind]+int(pulse_window/2)) # look forward half the
        else:
            # If the next pulse is overlapping, set the pulse boundary as the minimum between the two
            if time_diffs[peak_ind]<int(pulse_window/2):
                valley_time = np.argmin(sum_data[peaks[peak_ind]:peaks[peak_ind+1]])+peaks[peak_ind]
                window_ends.append(valley_time)
                prev_end = valley_time
            else:
                window_ends.append(peaks[peak_ind]+int(pulse_window/2))
                prev_end = -1

    # Find peak start, end times
    pulse_start_times = []
    pulse_end_times = []
    for peak_ind in range(len(peaks)):
        #start_time, end_time = pulse_bounds_area(sum_data, window_starts[peak_ind], window_ends[peak_ind]-window_starts[peak_ind], s2_start_frac,
        #                                        s2_end_frac)
        start_time, end_time = pulse_bounds(sum_data, window_starts[peak_ind], window_ends[peak_ind]-window_starts[peak_ind], s2_start_frac,
                                                s2_end_frac)
        pulse_start_times.append(start_time)
        pulse_end_times.append(end_time)


    # s2_found=s2_max>s2_thresh
    t1 = time.time()

    t2 = time.time()

    # s2_bottom = np.sum(np.array(s2_ch_area)[bottom_channels])
    # s2_top = np.sum(np.array(s2_ch_area)[top_channels])
    # s2_tba = (s2_top-s2_bottom)/(s2_top+s2_bottom)

    # Code to allow skipping to another event index for plotting
    plot_event_ind = i
    try:
        plot_event_ind = int(inn)
        if plot_event_ind < i:
            inn = ''
            plot_event_ind = i
            print("Can't go backwards! Continuing to next event.")
    except ValueError:
        plot_event_ind = i

    if False and not inn == 'q' and plot_event_ind == i:
        # if s1_found and s2_found:
        fig = pl.figure(1, figsize=(10, 7))
        # pl.rc('xtick', labelsize=25)
        # pl.rc('ytick', labelsize=25)

        ax = pl.subplot2grid((2, 2), (0, 0))
        pl.title("Top array, event " + str(i))
        pl.grid(b=True, which='major', color='lightgray', linestyle='--')
        ch_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        ch_colors = [pl.cm.tab10(ii) for ii in range(n_channels)]
        # ch_colors=[pl.cm.Dark2(ii) for ii in np.linspace(0.2,0.9,n_channels)]
        # ch_colors=['y','cyan','magenta','b','y','cyan','magenta','b']
        if s2_found:
            ax.axvspan(s2_start_pos * tscale, s2_end_pos * tscale, alpha=0.3, color='blue')
        if s1_found:
            ax.axvspan(s1_start_pos * tscale, s1_end_pos * tscale, alpha=0.3, color='green')
        for i_chan in range(n_channels - 1):
            if i_chan == (n_channels - 1) / 2:
                ax = pl.subplot2grid((2, 2), (0, 1))
                pl.title("Bottom array, event " + str(i))
                pl.grid(b=True, which='major', color='lightgray', linestyle='--')

                if s2_found:
                    ax.axvspan(s2_start_pos * tscale, s2_end_pos * tscale, alpha=0.3, color='blue')
                if s1_found:
                    ax.axvspan(s1_start_pos * tscale, s1_end_pos * tscale, alpha=0.3, color='green')

            pl.plot(t_matrix[i, :], v_matrix_all_ch[i_chan][i, :], color=ch_colors[i_chan], label=ch_labels[i_chan])
            pl.xlim([trigger_time_us - 10, trigger_time_us + 10])
            pl.ylim([0, 1000 / chA_spe_size])
            pl.xlabel('Time (us)')
            pl.ylabel('Phd/sample')
            pl.legend()
            # triggertime_us = (t[-1]*0.2)
            # pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')

        ax = pl.subplot2grid((2, 2), (1, 0), colspan=2)
        pl.plot(t_matrix[i, :], vsum_matrix[i, :], 'blue')
        pl.plot(t_matrix[i, :], data_conv + sum_baseline, 'red')
        pl.plot(t_matrix[i, :], np.tile(sum_baseline, np.size(data_conv)), 'gray')
        pl.vlines(x=peaks*tscale, ymin=data_conv[peaks]+sum_baseline - properties["prominences"],
                   ymax=data_conv[peaks]+sum_baseline, color="C1")
        pl.hlines(y=properties["width_heights"]+sum_baseline, xmin=properties["left_ips"]*tscale,
                   xmax=properties["right_ips"]*tscale, color="C1")
        #print("Width heights: ", properties["width_heights"]+sum_baseline)
        #print("Left ips (us): ", properties["left_ips"]*tscale)
        #print("Right ips (us): ", properties["right_ips"]*tscale)
        #print("Prominences: ", properties["prominences"])
        # pl.plot(t_matrix[i,:],sum_data,'blue')
        pl.xlim([0, event_window])
        pl.ylim([0, 4000 / chA_spe_size])
        pl.xlabel('Time (us)')
        pl.ylabel('Phd/sample')
        pl.title("Sum, event " + str(i))
        pl.grid(b=True, which='major', color='lightgray', linestyle='--')
        triggertime_us = (t[-1] * 0.2)
        pl.plot(t_matrix[i, peaks], vsum_matrix[i, peaks], 'x')
        # pl.plot(t_matrix[i,peaks], sum_data[peaks], 'x')
        # pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        for start_time, end_time in zip(pulse_start_times, pulse_end_times):
            ax.axvspan(start_time * tscale, end_time * tscale, alpha=0.3, color='blue')
        if s2_found:
            ax.axvspan(s2_start_pos * tscale, s2_end_pos * tscale, alpha=0.3, color='blue')
        if s1_found:
            ax.axvspan(s1_start_pos * tscale, s1_end_pos * tscale, alpha=0.3, color='green')

        pl.draw()
        pl.show(block=0)
        inn = input("Press enter to continue, q to skip plotting, event number to skip to that event (forward only)")
        fig.clf()

timer_end = time.time()
print("Processing time: ",timer_end-timer_start)