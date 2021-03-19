import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import time
import sys

import PulseFinderScipy as pf
import PulseQuantities as pq
import PulseClassification as pc

data_dir = "C:/Users/ryanm/Documents/Research/Data/SPE_LN/"

SPEMode = True
LED = True
plotyn = True
saveplot = True

# set plotting style
mpl.rcParams['font.size']=10
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[8.0,6.0]

# ==================================================================
# define DAQ and other parameters
#wsize = 12500             # size of event window in samples. 1 sample = 2 ns.
event_window = 4.  # in us
wsize = int(500 * event_window)  # samples per waveform # 12500 for 25 us
vscale = (2000.0/16384.0) # = 0.122 mV/ADCC, vertical scale
tscale = (8.0/4096.0)     # = 0.002 µs/sample, time scale

# Set range to look for pulses
if LED:
    left_bound = int(2/tscale)
    right_bound = int(3/tscale)
else:
    left_bound = 0
    right_bound = wsize

n_sipms = 16


# ==================================================================

#load in raw data

t_start = time.time()

block_size = 5000
n_block = 100
max_evts = n_block*block_size#5000  # 25000 # -1 means read in all entries; 25000 is roughly the max allowed in memory on the DAQ computer
max_pts = -1  # do not change
if max_evts > 0:
    max_pts = wsize * max_evts
load_dtype = "int16"


# RQ's for SPE analysis
max_pulses = 4
p_found = np.zeros((n_sipms, max_evts, max_pulses), dtype=np.int)
p_start = np.zeros((n_sipms, max_evts, max_pulses), dtype=np.int)
p_end   = np.zeros((n_sipms, max_evts, max_pulses), dtype=np.int)

p_area = np.zeros((n_sipms, max_evts, max_pulses))
p_max_height = np.zeros((n_sipms, max_evts, max_pulses))
p_width = np.zeros((n_sipms, max_evts, max_pulses))

n_pulses = np.zeros((n_sipms, max_evts), dtype=np.int)

inn=""

# Loop over blocks
for j in range(n_block):
    ch_data = []
    for ch_ind in range(n_sipms):
        ch_data.append(np.fromfile(data_dir + "wave"+str(ch_ind)+".dat", dtype=load_dtype, offset = block_size*wsize*j, count=wsize*block_size))

    t_end_load = time.time()
    print("Time to load files: ", t_end_load-t_start)

    # scale waveforms to get units of mV/sample
    # then for each channel ensure we 
    # have an integer number of events
    array_dtype = "float32" # using float32 reduces memory for big files, otherwise implicitly converts to float64

    # matrix of all channels including the sum waveform
    v_matrix_all_ch = []
    for ch_ind in range(n_sipms):
        V = vscale * ch_data[ch_ind].astype(array_dtype)
        V = V[:int(len(V) / wsize) * wsize]
        V = V.reshape(int(V.size / wsize), wsize) # reshape to make each channel's matrix of events
        v_matrix_all_ch.append(V)
        if ch_ind==0: v_sum = np.copy(V)
        else: v_sum += V
    v_matrix_all_ch.append(v_sum)
    
    # create a time axis in units of µs:
    x = np.arange(0, wsize, 1)
    t = tscale*x
    t_matrix = np.repeat(t[np.newaxis,:], V.size/wsize, 0)

    # Note: if max_evts != -1, we won't load in all events in the dataset
    n_events = int(v_matrix_all_ch[0].shape[0])
    if n_events == 0: break
        

    # Baseline subtraction
    # For LED, looks 0.5 us before expected range of pulse
    if LED:
        baseline_start = left_bound - int(0.5/tscale)
        baseline_end = left_bound
    # Otherwise, looks at first 0.5 us
    else:
        baseline_start = 0
        baseline_end = int(0.5/tscale)

    # baseline subtracted (bls) waveforms saved in this matrix:
    v_bls_matrix_all_ch = np.zeros( np.shape(v_matrix_all_ch), dtype=array_dtype) # dims are (chan #, evt #, sample #)

    t_end_wfm_fill = time.time()
    print("Time to fill all waveform arrays: ", t_end_wfm_fill - t_end_load)

    print("Events to process: ",n_events)
    for b in range(0, n_events):

        baselines = [ np.mean( ch_j[b,baseline_start:baseline_end] ) for ch_j in v_matrix_all_ch ]
        
        ch_data = [ch_j[b,:]-baseline_j for (ch_j,baseline_j) in zip(v_matrix_all_ch,baselines)]
        
        v_bls_matrix_all_ch[:,b,:] = ch_data


    # ==================================================================
    # ==================================================================
    # now setup for pulse finding on the baseline-subtracted sum waveform

    print("Running pulse finder on {:d} events...".format(n_events))

    # Loop index key:
    # j = blocks 
    # b = baseline sub 
    # i = events 
    # k = sipms
    # m = start time sorting
    # n = pulses

    # Loop over events
    for i in range(j*block_size, j*block_size+n_events):
        if i%500==0: print("Event #",i)
        
        # Loop over channels, slowest part of the code
        # Have to do a loop, findPulses does not like v_bls_matrix_all_ch[:,i,:] 
        for k in range(n_sipms):
            
            # Find pulse locations; other quantities for pf tuning/debugging
            start_times, end_times, peaks, data_conv, properties = pf.findPulses( v_bls_matrix_all_ch[k,i-j*block_size,left_bound:right_bound], max_pulses, SPEMode=SPEMode)
          
            start_times = [t+left_bound for t in start_times] 
            end_times  = [t+left_bound for t in end_times] 

            # Sort pulses by start times, not areas
            startinds = np.argsort(start_times)
            for m in startinds:
                p_start[k,i,m] = start_times[m]
                p_end[k,i,m] = end_times[m]

            # Loop over found pulses, calculate interesting quantities
            n_pulses[k,i] = len(start_times)
            for n in range(n_pulses[k,i]):
                p_area[k,i,n] = pq.GetPulseArea(p_start[k,i,n], p_end[k,i,n], v_bls_matrix_all_ch[k,i-j*block_size,:] )
                p_max_height[k,i,n] = pq.GetPulseMaxHeight(p_start[k,i,n], p_end[k,i,n], v_bls_matrix_all_ch[k,i-j*block_size,:] )
                p_width[k,i,n] = p_end[k,i,n] - p_start[k,i,n]

            # Plotter
            if not inn == 'q' and plotyn:
                # Plot something
                fig = pl.figure(1,figsize=(10, 7))
                pl.grid(b=True,which='major',color='lightgray',linestyle='--')
                pl.plot(t_matrix[i,:], v_bls_matrix_all_ch[k,i-j*block_size,:], color='blue')
                pl.plot(t_matrix[i,left_bound:right_bound], data_conv, color='red')
                for pulse in range(len(start_times)):
                    pl.axvspan(start_times[pulse] * tscale, end_times[pulse] * tscale, alpha=0.25, color='green')
                
                pl.ylim([-2,5])

                pl.title("Channel "+str(k)+", Event "+str(i) )
                pl.xlabel('Time (us)')
                pl.ylabel('mV')
                pl.draw()
                pl.show(block=0)
                inn = input("Press enter to continue, q to stop plotting, evt # to skip to # (forward only)")
                fig.clf()

            
# End of pulse finding and plotting event loop

n_events = i
print("total number of events processed:", n_events)


# Clean up and Plotting


def SPE_Area_Hist(data,sipm_n):
    pl.figure()
    pl.hist(data, 100)
    pl.xlabel("Pulse Area (mV*us)")
    if saveplot: pl.savefig(data_dir+"SPE_area_sipm_"+str(sipm_n)+".png")
    return 

list_rq = {}

for p in range(n_sipms):
    cleanCut = p_area[p,:,:] > 0
    cleanArea = p_area[p,cleanCut].flatten()
    SPE_Area_Hist(cleanArea*tscale, p)
    list_rq['p_area_'+str(p)] = cleanArea

rq = open(data_dir + "spe_rq.npz",'wb')
np.savez(rq, **list_rq)
rq.close()
