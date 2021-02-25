
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import time
import sys

import PulseFinderScipy as pf
import PulseQuantities as pq
import PulseClassification as pc

# set plotting style
mpl.rcParams['font.size']=10
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[8.0,6.0]

# ==================================================================
# define DAQ and other parameters
#wsize = 12500             # size of event window in samples. 1 sample = 2 ns.
event_window = 25.  # in us
wsize = int(500 * event_window)  # samples per waveform # 12500 for 25 us
vscale = (2000.0/16384.0) # = 0.122 mV/ADCC, vertical scale
tscale = (8.0/4096.0)     # = 0.002 µs/sample, time scale

post_trigger = 0.5 # Was 0.2 for data before 11/22/19
trigger_time_us = event_window*(1-post_trigger)
trigger_time = int(trigger_time_us/tscale)

n_sipms = 8
n_channels = n_sipms+1 # includes sum

# define top, bottom channels
n_top = int((n_channels-1)/2)
top_channels=np.array(range(n_top),int)
bottom_channels=np.array(range(n_top,2*n_top),int)

# sphe sizes in mV*sample
chA_spe_size = 29.02
chB_spe_size = 30.61
chC_spe_size = 28.87
chD_spe_size = 28.86*1.25 # scale factor (0.7-1.4) empirical as of Dec 9, 2020
chE_spe_size = 30.4
chF_spe_size = 30.44
chG_spe_size = 30.84
chH_spe_size = 30.3*1.8 # scale factor (1.6-2.2) empirical as of Dec 9, 2020
spe_sizes = [chA_spe_size, chB_spe_size, chC_spe_size, chD_spe_size, chE_spe_size, chF_spe_size, chG_spe_size, chH_spe_size]
# ==================================================================

#load in raw data
#data_dir="../data/bkg_3.5g_3.9c_27mV_7_postrecover2_5min/"
#data_dir="../data/bkg_3.5g_3.9c_27mV_1_5min/"
#data_dir="../data/fewevts/"
#data_dir="../data/po_5min/"
#data_dir = "C:/Users/ryanm/Documents/Research/Data/bkg_3.5g_3.9c_27mV_6_postrecover_5min/" # Old data
#data_dir = "C:/Users/swkra/Desktop/Jupyter temp/data-201909/091219/"
#data_dir = "/home/xaber/caen/wavedump-3.8.2/data/011521/Flow_Th_Co57_4.8g_5.0c_25mV_1.5bar_circ_5min/"
data_dir = "D:/.shortcut-targets-by-id/11qeqHWCbcKfFYFQgvytKem8rulQCTpj8/crystalize/data/data-202102/022421/Po_4.8g_7.0c_3mV_1.75bar_circ_20min/"
#data_dir  = "C:/Users/ryanm/Documents/Research/Data/bkg_2.8g_3.2c_25mV_1_1.6_circ_0.16bottle_5min/" # Weird but workable data
#data_dir = "C:/Users/ryanm/Documents/Research/Data/Flow_Th_with_Ba133_0g_0c_25mV_1.5bar_nocirc_5min/" # Weird double s1 data
#"C:/Users/swkra/Desktop/Jupyter temp/data-202009/091720/bkg_3.5g_3.9c_27mV_7_postrecover2_5min/"

t_start = time.time()

max_evts = 20000#5000  # 25000 # -1 means read in all entries; 25000 is roughly the max allowed in memory on the DAQ computer
max_pts = -1  # do not change
if max_evts > 0:
    max_pts = wsize * max_evts
load_dtype = "int16"
ch_data = []
for ch_ind in range(n_sipms):
    ch_data.append(np.fromfile(data_dir + "wave"+str(ch_ind)+".dat", dtype=load_dtype, count=max_pts))

t_end_load = time.time()
print("Time to load files: ", t_end_load-t_start)

# scale waveforms to get units of mV/sample
# then for each channel ensure we 
# have an integer number of events
array_dtype = "float32" # using float32 reduces memory for big files, otherwise implicitly converts to float64

# matrix of all channels including the sum waveform
v_matrix_all_ch = []
for ch_ind in range(n_sipms):
    V = vscale * ch_data[ch_ind].astype(array_dtype) / spe_sizes[ch_ind]
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
    
# perform baseline subtraction:
# for now, using first 2 µs of event
baseline_start = int(0./tscale)
baseline_end = int(2./tscale)

# baseline subtracted (bls) waveforms saved in this matrix:
v_bls_matrix_all_ch = np.zeros( np.shape(v_matrix_all_ch), dtype=array_dtype) # dims are (chan #, evt #, sample #)

t_end_wfm_fill = time.time()
print("Time to fill all waveform arrays: ", t_end_wfm_fill - t_end_load)

print("Events to process: ",n_events)
for i in range(0, n_events):
    
    sum_baseline = np.mean( v_matrix_all_ch[-1][i,baseline_start:baseline_end] ) #avg ~us, avoiding trigger
    baselines = [ np.mean( ch_j[i,baseline_start:baseline_end] ) for ch_j in v_matrix_all_ch ]
    
    sum_data = v_matrix_all_ch[-1][i,:] - sum_baseline
    ch_data = [ch_j[i,:]-baseline_j for (ch_j,baseline_j) in zip(v_matrix_all_ch,baselines)]
    
    v_bls_matrix_all_ch[:,i,:] = ch_data


# ==================================================================
# ==================================================================
# now setup for pulse finding on the baseline-subtracted sum waveform

# max number of pulses per event
max_pulses = 4

# pulse RQs to save

# RQs to add:
# Pulse level: channel areas (fracs; max fracs), TBA, rise time? (just difference of AFTs...)
# Event level: drift time; S1, S2 area
# Pulse class (S1, S2, other)

p_start = np.zeros(( n_events, max_pulses), dtype=np.int)
p_end   = np.zeros(( n_events, max_pulses), dtype=np.int)
p_found = np.zeros(( n_events, max_pulses), dtype=np.int)

p_area = np.zeros(( n_events, max_pulses))
p_max_height = np.zeros(( n_events, max_pulses))
p_min_height = np.zeros(( n_events, max_pulses))
p_width = np.zeros(( n_events, max_pulses))

p_afs_2l = np.zeros((n_events, max_pulses) )
p_afs_2r = np.zeros((n_events, max_pulses) )
p_afs_1 = np.zeros((n_events, max_pulses) )
p_afs_25 = np.zeros((n_events, max_pulses) )
p_afs_50 = np.zeros((n_events, max_pulses) )
p_afs_75 = np.zeros((n_events, max_pulses) )
p_afs_99 = np.zeros((n_events, max_pulses) )
            
p_hfs_10l = np.zeros((n_events, max_pulses) )
p_hfs_50l = np.zeros((n_events, max_pulses) )
p_hfs_10r = np.zeros((n_events, max_pulses) )
p_hfs_50r = np.zeros((n_events, max_pulses) )

p_mean_time = np.zeros((n_events, max_pulses) )
p_rms_time = np.zeros((n_events, max_pulses) )

# Channel level (per event, per pulse, per channel)
p_start_ch = np.zeros((n_events, max_pulses, n_channels-1), dtype=np.int)
p_end_ch = np.zeros((n_events, max_pulses, n_channels-1), dtype=np.int )
p_area_ch = np.zeros((n_events, max_pulses, n_channels-1) )
p_area_ch_frac = np.zeros((n_events, max_pulses, n_channels-1) )

p_area_top = np.zeros((n_events, max_pulses))
p_area_bottom = np.zeros((n_events, max_pulses))
p_tba = np.zeros((n_events, max_pulses))

p_class = np.zeros((n_events, max_pulses), dtype=np.int)

# Event-level variables
n_pulses = np.zeros(n_events, dtype=np.int)

n_s1 = np.zeros(n_events, dtype=np.int)
n_s2 = np.zeros(n_events, dtype=np.int)
sum_s1_area = np.zeros(n_events)
sum_s2_area = np.zeros(n_events)
drift_Time = np.zeros(n_events)
s1_before_s2 = np.zeros(n_events)

# Temporary, for testing low area, multiple-S1 events
dt = np.zeros(n_events)
small_weird_areas = np.zeros(n_events)
big_weird_areas = np.zeros(n_events)

inn=""

print("Running pulse finder on {:d} events...".format(n_events))

# use for coloring pulses
pulse_class_colors = np.array(['blue', 'green', 'red', 'magenta', 'darkorange'])
pulse_class_labels = np.array(['Other', 'S1-like LXe', 'S1-like gas', 'S2-like', 'Merged S1/S2'])
pc_legend_handles=[]
for class_ind in range(len(pulse_class_labels)):
    pc_legend_handles.append(mpl.patches.Patch(color=pulse_class_colors[class_ind], label=str(class_ind)+": "+pulse_class_labels[class_ind]))

n_golden = int(0)

for i in range(0, n_events):
    if i%500==0: print("Event #",i)
    
    # Find pulse locations; other quantities for pf tuning/debugging
    start_times, end_times, peaks, data_conv, properties = pf.findPulses( v_bls_matrix_all_ch[-1,i,:], max_pulses )


    # Sort pulses by start times, not areas
    startinds = np.argsort(start_times)
    n_pulses[i] = len(start_times)
    #if (n_pulses[i] < 1):
        #print("No pulses found for event {0}; skipping".format(i))
        #continue
    for m in startinds:
        if m >= max_pulses:
            continue
        p_start[i,m] = start_times[m]
        p_end[i,m] = end_times[m]



    # Individual channel pulse locations, in case you want this info
    # Can't just ":" the the first index in data, findPulses doesn't like it, so have to loop 
    #for j in range(n_channels-1):
    #    start_times_ch, end_times_ch, peaks_ch, data_conv_ch, properties_ch = pf.findPulses( v_bls_matrix_all_ch[j,i,:], max_pulses )
        # Sorting by start times from the sum of channels, not each individual channel
    #    for k in startinds:
    #        if k >= len(start_times_ch):
    #            continue
    #        p_start_ch[i,k,j] = start_times_ch[k]
    #        p_end_ch[i,k,j] = end_times_ch[k]
        

    # Calculate interesting quantities, only for pulses that were found
    for pp in range(n_pulses[i]):

        # Area, max & min heights, width, pulse mean & rms
        p_area[i,pp] = pq.GetPulseArea(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[-1,i,:] )
        p_max_height[i,pp] = pq.GetPulseMaxHeight(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[-1,i,:] )
        p_min_height[i,pp] = pq.GetPulseMinHeight(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[-1,i,:] )
        p_width[i,pp] = p_end[i,pp] - p_start[i,pp]
        #(p_mean_time[i,pp], p_rms_time[i,pp]) = pq.GetPulseMeanAndRMS(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[-1,i,:])

        # Area and height fractions      
        (p_afs_2l[i,pp], p_afs_1[i,pp], p_afs_25[i,pp], p_afs_50[i,pp], p_afs_75[i,pp], p_afs_99[i,pp]) = pq.GetAreaFraction(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[-1,i,:] )
        (p_hfs_10l[i,pp], p_hfs_50l[i,pp], p_hfs_10r[i,pp], p_hfs_50r[i,pp]) = pq.GetHeightFractionSamples(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[-1,i,:] )
    
        # Areas for individual channels and top bottom
        p_area_ch[i,pp,:] = pq.GetPulseAreaChannel(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[:,i,:] )
        p_area_ch_frac[i,pp,:] = p_area_ch[i,pp,:]/p_area[i,pp]
        p_area_top[i,pp] = sum(p_area_ch[i,pp,top_channels])
        p_area_bottom[i,pp] = sum(p_area_ch[i,pp,bottom_channels])
        p_tba[i, pp] = (p_area_top[i, pp] - p_area_bottom[i, pp]) / (p_area_top[i, pp] + p_area_bottom[i, pp])

        
    # Pulse classifier, work in progress
    p_class[i,:] = pc.ClassifyPulses(p_tba[i, :], (p_afs_50[i, :]-p_afs_2l[i, :])*tscale, n_pulses[i])

    # Event level analysis. Look at events with both S1 and S2.
    index_s1 = (p_class[i,:] == 1) + (p_class[i,:] == 2) # S1's
    index_s2 = (p_class[i,:] == 3) + (p_class[i,:] == 4) # S2's
    n_s1[i] = np.sum(index_s1)
    n_s2[i] = np.sum(index_s2)
    
    if n_s1[i] > 0:
        sum_s1_area[i] = sum(p_area[i, index_s1])
    if n_s2[i] > 0:
        sum_s2_area[i] = sum(p_area[i, index_s2])
    if n_s1[i] == 1:
        if n_s2[i] == 1:
            drift_Time[i] = tscale*(p_start[i, np.argmax(index_s2)] - p_start[i, np.argmax(index_s1)])
        if n_s2[i] > 1:
            s1_before_s2[i] = np.argmax(index_s1) < np.argmax(index_s2) 
    
    if drift_Time[i]>0:
        n_golden += 1


    # =============================================================
    # draw the waveform and the pulse bounds found

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

    # Condition to plot now includes this rise time calc, not necessary
    riseTimeCondition = ((p_afs_50[i,:n_pulses[i]]-p_afs_2l[i,:n_pulses[i]] )*tscale < 0.6)*((p_afs_50[i,:n_pulses[i]]-p_afs_2l[i,:n_pulses[i]] )*tscale > 0.2)
    
    po_test = np.any((p_area[i,:]>5.0e4)*((p_afs_50[i,:]-p_afs_2l[i,:] )*tscale<1.0))
    
    # Condition to skip the individual plotting
    #plotyn = False#np.any((p_tba[i,:]>-0.75)*(p_tba[i,:]<-0.25)*(p_area[i,:]<3200)*(p_area[i,:]>1500))#np.any((p_tba[i,:]>-0.91)*(p_tba[i,:]<-0.82)*(p_area[i,:]<2800)*(p_area[i,:]>1000))# True#np.any(p_class[i,:]==4)#False#np.any(p_area[i,:]>1000) and 
    #plotyn = sum_s1_area[i]>1000 and sum_s1_area[i]<3000 and sum_s2_area[i]<1*10**4
    plotyn = False
    # Pulse area condition
    areaRange = np.sum((p_area[i,:] < 50)*(p_area[i,:] > 5))
    if areaRange > 0:
        dt[i] = abs(p_start[i,1] - p_start[i,0]) # For weird double s1 data
        weird_areas =[p_area[i,0], p_area[i,1] ]
        small_weird_areas[i] = min(weird_areas)
        big_weird_areas[i] = max(weird_areas)


    # Both S1 and S2 condition
    s1s2 = (n_s1[i] == 1)*(n_s2[i] == 1)

    if inn == 'stop': sys.exit()
    
    if not inn == 'q' and plot_event_ind == i and plotyn:

        fig = pl.figure(1,figsize=(10, 7))
        pl.rc('xtick', labelsize=10)
        pl.rc('ytick', labelsize=10)
        
        ax = pl.subplot2grid((2,2),(0,0))
        pl.title("Top array, event "+str(i))
        pl.grid(b=True,which='major',color='lightgray',linestyle='--')
        ch_labels = ['A','B','C','D','E','F','G','H']
        ch_colors = [pl.cm.tab10(ii) for ii in range(n_channels)]
        for pulse in range(len(start_times)): # fill found pulse regions for top
            ax.axvspan(start_times[pulse] * tscale, end_times[pulse] * tscale, alpha=0.25,
                       color=pulse_class_colors[p_class[i, pulse]])
        for i_chan in range(n_channels-1):
            if i_chan == (n_channels-1)/2:
                ax = pl.subplot2grid((2,2),(0,1))
                pl.title("Bottom array, event "+str(i))
                pl.grid(b=True,which='major',color='lightgray',linestyle='--')
                for pulse in range(len(start_times)):  # fill found pulse regions for bottom
                    ax.axvspan(start_times[pulse] * tscale, end_times[pulse] * tscale, alpha=0.25,
                               color=pulse_class_colors[p_class[i, pulse]])
            
            pl.plot(t_matrix[i,:],v_bls_matrix_all_ch[i_chan,i,:],color=ch_colors[i_chan],label=ch_labels[i_chan])
            #pl.plot( x, v_bls_matrix_all_ch[i_chan,i,:],color=ch_colors[i_chan],label=ch_labels[i_chan] )
            pl.xlim([trigger_time_us-8,trigger_time_us+8])
            #pl.xlim([wsize/2-4000,wsize/2+4000])
            pl.ylim([-5, 3000/chA_spe_size])
            pl.xlabel('Time (us)')
            #pl.xlabel('Samples')
            pl.ylabel('phd/sample')
            pl.legend()
        
        ax = pl.subplot2grid((2,2),(1,0),colspan=2)
        #pl.plot(t_matrix[i,:],v_bls_matrix_all_ch[-1,i,:],'blue')
        pl.plot( x*tscale, v_bls_matrix_all_ch[-1,i,:],'blue' )
        #pl.xlim([0,wsize])
        pl.xlim([0,event_window])
        pl.ylim( [-1, 1.01*np.max(v_bls_matrix_all_ch[-1,i,:])])
        pl.xlabel('Time (us)')
        #pl.xlabel('Samples')
        pl.ylabel('phd/sample')
        pl.title("Sum, event "+ str(i))
        pl.grid(b=True,which='major',color='lightgray',linestyle='--')
        pl.legend(handles=pc_legend_handles)

        for pulse in range(len(start_times)):
            ax.axvspan(start_times[pulse] * tscale, end_times[pulse] * tscale, alpha=0.25, color=pulse_class_colors[p_class[i, pulse]])
            ax.text((end_times[pulse]) * tscale, 0.9 * ax.get_ylim()[1], '{:.1f} phd'.format(p_area[i, pulse]),
                    fontsize=9, color=pulse_class_colors[p_class[i, pulse]])
        
        #ax.axhline( 0.276, 0, wsize, linestyle='--', lw=1, color='orange')

        # Debugging of pulse finder
        debug_pf = True
        if debug_pf and n_pulses[i]>0:
            pl.plot(t_matrix[i, :], data_conv, 'red')
            pl.plot(t_matrix[i, :], np.tile(0., np.size(data_conv)), 'gray')
            pl.vlines(x=peaks*tscale, ymin=data_conv[peaks] - properties["prominences"],
                       ymax=data_conv[peaks], color="C1")
            pl.hlines(y=properties["width_heights"], xmin=properties["left_ips"]*tscale,
                       xmax=properties["right_ips"]*tscale, color="C1")

        pl.draw()
        pl.show(block=0)
        inn = input("Press enter to continue, q to stop plotting, evt # to skip to # (forward only)")
        fig.clf()
        
# end of pulse finding and plotting event loop

rq = open(data_dir + "rq.npz",'wb')
np.savez(rq, n_s1=n_s1, n_s2= n_s2, s1_before_s2=s1_before_s2, n_pulses=n_pulses, n_events = n_events, p_area = p_area, p_class=p_class, drift_Time=drift_Time, p_max_height=p_max_height, p_min_height=p_min_height, p_width=p_width, p_afs_2l=p_afs_2l, p_afs_50=p_afs_50, p_area_ch=p_area_ch, p_area_ch_frac=p_area_ch_frac, p_area_top=p_area_top, p_area_bottom=p_area_bottom, p_tba=p_tba, sum_s1_area=sum_s1_area, sum_s2_area=sum_s2_area   )
rq.close()