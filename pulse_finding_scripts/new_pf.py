
import numpy as np
#import pylab as pl
import matplotlib.pyplot as pl
import matplotlib as mpl
import scipy
from scipy.signal import find_peaks
from scipy.signal import argrelmin
import time

# set plotting style
mpl.rcParams['font.size']=10
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[8.0,6.0]

# ==================================================================
# define DAQ and other parameters
wsize = 12500             # size of event window in samples. 1 sample = 2 ns.
vscale = (2000.0/16384.0) # = 0.122 mV/ADCC, vertical scale
tscale = (8.0/4096.0)     # = 0.002 µs/sample, time scale

n_sipms = 8
n_channels = n_sipms+1 # includes sum

# sphe sizes in mV*sample
chA_spe_size = 29.02
chB_spe_size = 30.61
chC_spe_size = 28.87
chD_spe_size = 28.86
chE_spe_size = 30.4
chF_spe_size = 30.44
chG_spe_size = 30.84
chH_spe_size = 30.3
# ==================================================================

#load in raw data
data_dir="../data/fewevts/"
channel_0 = np.fromfile(data_dir+"wave0.dat", dtype="int16")
channel_1 = np.fromfile(data_dir+"wave1.dat", dtype="int16")
channel_2 = np.fromfile(data_dir+"wave2.dat", dtype="int16")
channel_3 = np.fromfile(data_dir+"wave3.dat", dtype="int16")
channel_4 = np.fromfile(data_dir+"wave4.dat", dtype="int16")
channel_5 = np.fromfile(data_dir+"wave5.dat", dtype="int16")
channel_6 = np.fromfile(data_dir+"wave6.dat", dtype="int16")
channel_7 = np.fromfile(data_dir+"wave7.dat", dtype="int16")

# scale waveforms to get units of mV/sample
# then for each channel ensure we 
# have an integer number of events
V = vscale*channel_0/chA_spe_size
V = V[:int(len(V)/wsize)*wsize]

V_1 = vscale*channel_1/chB_spe_size
V_1 = V_1[:int(len(V)/wsize)*wsize]

V_2 = vscale*channel_2/chC_spe_size
V_2 = V_2[:int(len(V)/wsize)*wsize]

V_3 = vscale*channel_3/chD_spe_size
V_3 = V_3[:int(len(V)/wsize)*wsize]

V_4 = vscale*channel_4/chE_spe_size
V_4 = V_4[:int(len(V)/wsize)*wsize]

V_5 = vscale*channel_5/chF_spe_size
V_5 = V_5[:int(len(V)/wsize)*wsize]

V_6 = vscale*channel_6/chG_spe_size
V_6 = V_6[:int(len(V)/wsize)*wsize]

V_7 = vscale*channel_7/chH_spe_size
V_7 = V_7[:int(len(V)/wsize)*wsize]

# reshape to make each channel's matrix of events
v_matrix = V.reshape(int(V.size/wsize),wsize)
v1_matrix = V_1.reshape(int(V.size/wsize),wsize)
v2_matrix = V_2.reshape(int(V.size/wsize),wsize)
v3_matrix = V_3.reshape(int(V.size/wsize),wsize)
v4_matrix = V_4.reshape(int(V.size/wsize),wsize)
v5_matrix = V_5.reshape(int(V.size/wsize),wsize)
v6_matrix = V_6.reshape(int(V.size/wsize),wsize)
v7_matrix = V_7.reshape(int(V.size/wsize),wsize)

# sum waveform:
vsum_matrix = v_matrix+v1_matrix+v2_matrix+v3_matrix+v4_matrix+v5_matrix+v6_matrix+v7_matrix

# matrix of all channels including the sum waveform:
v_matrix_all_ch = [v_matrix,v1_matrix,v2_matrix,v3_matrix,v4_matrix,v5_matrix,v6_matrix,v7_matrix,vsum_matrix]

# create a time axis in units of µs:
x = np.arange(0, wsize, 1)
t = tscale*x
t_matrix = np.repeat(t[np.newaxis,:], V.size/wsize, 0)

tot_events = int(v_matrix.shape[0])

# perform baseline subtraction:
# for now, using first 2 µs of event
baseline_start = int(0./tscale)
baseline_end = int(2./tscale)

# baseline subtracted (bls) waveforms saved in this matrix:
v_bls_matrix_all_ch = np.zeros( np.shape(v_matrix_all_ch) ) # dims are (chan #, evt #, sample #)

print("Total events: ",tot_events)
for i in range(0, tot_events):
    #if i%100==0: print("Event #",i)
    
    sum_baseline = np.mean( v_matrix_all_ch[-1][i,baseline_start:baseline_end] ) #avg ~us, avoiding trigger
    baselines = [ np.mean( ch_j[i,baseline_start:baseline_end] ) for ch_j in v_matrix_all_ch ]
    
    sum_data = v_matrix_all_ch[-1][i,:] - sum_baseline
    ch_data = [ch_j[i,:]-baseline_j for (ch_j,baseline_j) in zip(v_matrix_all_ch,baselines)]
    
    v_bls_matrix_all_ch[:,i,:] = ch_data


# ==================================================================
# ==================================================================
# now setup for pulse finding on the baseline-subtracted sum waveform

# parameters
post_trigger = 0.5 # Was 0.2 for data before 11/22/19
event_window = wsize*tscale
trigger_time_us = event_window*(1-post_trigger)
trigger_time = int(trigger_time_us/tscale)
t_offset = int(0.2/tscale)
s1_window = int(0.5/tscale)
s2_window = int(3.5/tscale)
t_min_search = trigger_time-int(10./tscale)# was int(10./tscale)
t_max_search = trigger_time+int(10./tscale)# was int(22./tscale)
s1_thresh = 100/chA_spe_size
s1_range_thresh = 10/chA_spe_size
s2_thresh = 150/chA_spe_size
s2_start_frac=0.06 # s2 pulse starts at this fraction of peak height above baseline
s2_end_frac=0.05 # s2 pulse starts at this fraction of peak height above baseline
s1_start_frac=0.1
s1_end_frac=0.1


inn=""

for i in range(0, tot_events):
    
    
    
    
    
    
    
    
    # draw the waveform and the pulse bounds found
    if True and not inn=='q':
        
        fig = pl.figure(1,figsize=(10, 7))
        pl.rc('xtick', labelsize=10)
        pl.rc('ytick', labelsize=10)
        
        ax = pl.subplot2grid((2,2),(0,0))
        pl.title("Top array, event "+str(i))
        pl.grid(b=True,which='major',color='lightgray',linestyle='--')
        ch_labels = ['A','B','C','D','E','F','G','H']
        ch_colors = [pl.cm.tab10(ii) for ii in range(n_channels)]
        for i_chan in range(n_channels-1):
            if i_chan == (n_channels-1)/2:
                ax = pl.subplot2grid((2,2),(0,1))
                pl.title("Bottom array, event "+str(i))
                pl.grid(b=True,which='major',color='lightgray',linestyle='--')
            
            pl.plot(t_matrix[i,:],v_bls_matrix_all_ch[i_chan,i,:],color=ch_colors[i_chan],label=ch_labels[i_chan])
            pl.xlim([trigger_time_us-8,trigger_time_us+8])
            pl.ylim([-10, 1000/chA_spe_size])
            pl.xlabel('Time (us)')
            pl.ylabel('phd/sample')
            pl.legend()
        
        ax = pl.subplot2grid((2,2),(1,0),colspan=2)
        pl.plot(t_matrix[i,:],v_bls_matrix_all_ch[-1,i,:],'blue')
        pl.xlim([0,25])
        pl.ylim([-10, 4000/chA_spe_size])
        pl.xlabel('Time (us)')
        pl.ylabel('phd/sample')
        pl.title("Sum, event "+ str(i))
        pl.grid(b=True,which='major',color='lightgray',linestyle='--')
        triggertime_us = (t[-1]*0.2)
        
        pl.draw()
        pl.show(block=0)
        inn = input("Press enter to continue, q to skip plotting")
        fig.clf()
        
# end of pulse finding and plotting event loop
