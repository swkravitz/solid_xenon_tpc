
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import time

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
# ==================================================================

#load in raw data
#data_dir="../data/bkg_3.5g_3.9c_27mV_7_postrecover2_5min/"
#data_dir="../data/bkg_3.5g_3.9c_27mV_1_5min/"
#data_dir = "C:/Users/ryanm/Documents/Research/Data/bkg_3.5g_3.9c_27mV_6_postrecover_5min/" # Old data
#data_dir = "C:/Users/swkra/Desktop/Jupyter temp/data-202012/120920/Flow_Th_with_Ba133_0g_0c_25mV_1.5bar_nocirc_5min/"
#data_dir = "../data/Flow_Th_with_Ba133_0g_0c_25mV_1.5bar_nocirc_5min/"
data_dir = "../data/Flow_Th_with_Ba133_4.8g_5.0c_25mV_1.5bar_nocirc_5min/"

t_start = time.time()

max_evts = 1000 #5000  # 25000 # -1 means read in all entries; 25000 is roughly the max allowed in memory on the DAQ computer
max_pts = -1  # do not change
if max_evts > 0:
    max_pts = wsize * max_evts
load_dtype = "int16"
channel_0 = np.fromfile(data_dir + "wave0.dat", dtype=load_dtype, count=max_pts)
channel_1 = np.fromfile(data_dir + "wave1.dat", dtype=load_dtype, count=max_pts)
channel_2 = np.fromfile(data_dir + "wave2.dat", dtype=load_dtype, count=max_pts)
channel_3 = np.fromfile(data_dir + "wave3.dat", dtype=load_dtype, count=max_pts)
channel_4 = np.fromfile(data_dir + "wave4.dat", dtype=load_dtype, count=max_pts)
channel_5 = np.fromfile(data_dir + "wave5.dat", dtype=load_dtype, count=max_pts)
channel_6 = np.fromfile(data_dir + "wave6.dat", dtype=load_dtype, count=max_pts)
channel_7 = np.fromfile(data_dir + "wave7.dat", dtype=load_dtype, count=max_pts)

t_end_load = time.time()
print("Time to load files: ", t_end_load-t_start)

# scale waveforms to get units of mV/sample
# then for each channel ensure we 
# have an integer number of events
array_dtype = "float32" # using float32 reduces memory for big files, otherwise implicitly converts to float64
V = vscale*channel_0.astype(array_dtype)/chA_spe_size
V = V[:int(len(V)/wsize)*wsize]

V_1 = vscale*channel_1.astype(array_dtype)/chB_spe_size
V_1 = V_1[:int(len(V)/wsize)*wsize]

V_2 = vscale*channel_2.astype(array_dtype)/chC_spe_size
V_2 = V_2[:int(len(V)/wsize)*wsize]

V_3 = vscale*channel_3.astype(array_dtype)/chD_spe_size
V_3 = V_3[:int(len(V)/wsize)*wsize]

V_4 = vscale*channel_4.astype(array_dtype)/chE_spe_size
V_4 = V_4[:int(len(V)/wsize)*wsize]

V_5 = vscale*channel_5.astype(array_dtype)/chF_spe_size
V_5 = V_5[:int(len(V)/wsize)*wsize]

V_6 = vscale*channel_6.astype(array_dtype)/chG_spe_size
V_6 = V_6[:int(len(V)/wsize)*wsize]

V_7 = vscale*channel_7.astype(array_dtype)/chH_spe_size
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

# Note: if max_evts != -1, we won't load in all events in the dataset
n_events = int(v_matrix.shape[0])
    
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
p_area_tba = np.zeros((n_events, max_pulses))

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
pulse_class_colors = np.array(['blue', 'green', 'red', 'purple', 'black', 'magenta', 'darkorange'])
pulse_class_labels = np.array(['S1-like 1', 'S1-like 2', 'leading S1-like', 'meh?', 'also meh?', 'S2-like', 'S2-like'])

for i in range(0, n_events):
    if i%500==0: print("Event #",i)
    
    # Find pulse locations; other quantities for pf tuning/debugging
    start_times, end_times, peaks, data_conv, properties = pf.findPulses( v_bls_matrix_all_ch[-1,i,:], max_pulses )


    # Sort pulses by start times, not areas
    startinds = np.argsort(start_times)
    n_pulses[i] = len(start_times)
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
        p_area_tba[i,pp] = (p_area_top[i,pp] - p_area_bottom[i,pp])/(p_area_top[i,pp] + p_area_bottom[i,pp])

        
    # Pulse classifier, work in progress
    p_class[i,:] = pc.ClassifyPulses(p_area_tba[i,:], p_afs_2l[i,:], p_afs_50[i,:], tscale)

    # Event level analysis. Look at events with both S1 and S2.
    index_s1 = (p_class[i,:] == 1) + (p_class[i,:] == 2) # S1's
    index_s2 = (p_class[i,:] == 5) + (p_class[i,:] == 6) # S2's
    n_s1[i] = np.sum(index_s1)
    n_s2[i] = np.sum(index_s2)
    
    if n_s1[i] > 0:
        sum_s1_area[i] = sum(p_area[i, index_s1])
    if n_s2[i] > 0:
        sum_s2_area[i] = sum(p_area[i, index_s2])
    if n_s1[i] == 1:
        if n_s2[i] == 1:
            drift_Time[i] = p_start[i, np.argmax(index_s2)] - p_start[i, np.argmax(index_s1)]
        if n_s2[i] > 1:
            s1_before_s2[i] = np.argmax(index_s1) < np.argmax(index_s2) 
        


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
    
    # Condition to skip the individual plotting
    plotyn = True

    # Pulse area condition
    areaRange = np.sum((p_area[i,:] < 50)*(p_area[i,:] > 5))
    if areaRange > 0:
        dt[i] = abs(p_start[i,1] - p_start[i,0]) # For weird double s1 data
        weird_areas =[p_area[i,0], p_area[i,1] ]
        small_weird_areas[i] = min(weird_areas)
        big_weird_areas[i] = max(weird_areas)


    # Both S1 and S2 condition
    s1s2 = (n_s1[i] == 1)*(n_s2[i] == 1)

    if not inn == 'q' and plot_event_ind == i and plotyn:

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
        triggertime_us = (t[-1]*0.2)

        for pulse in range(len(start_times)):
            ax.axvspan(start_times[pulse] * tscale, end_times[pulse] * tscale, alpha=0.25, color=pulse_class_colors[p_class[i, pulse]])
            ax.text((end_times[pulse])*tscale,0.9*ax.get_ylim()[1],'{:.1f} phd'.format(p_area[i, pulse]),fontsize=9,color=pulse_class_colors[p_class[i, pulse]])
        #ax.axhline( 0.276, 0, wsize, linestyle='--', lw=1, color='orange')

        # Debugging of pulse finder
        debug_pf = True
        if debug_pf:
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


# Define some standard cuts for plotting
cut_dict = {}
cut_dict['ValidPulse'] = p_area > 0
cut_dict['PulseClass0'] = p_class == 0
cut_dict['S1'] = (p_class == 1) + (p_class == 2)
cut_dict['S2'] = (p_class == 5) + (p_class == 6)

# Pick which cut from cut_dict to apply here and whether to save plots
save_pulse_plots=True # One entry per pulse
save_S1S2_plots=True # One entry per S1 (S2) pulse
save_event_plots=True # One entry per event
pulse_cut_name = 'ValidPulse'
pulse_cut = cut_dict[pulse_cut_name]

cleanArea = p_area[pulse_cut].flatten()
cleanMax = p_max_height[pulse_cut].flatten()
cleanMin = p_min_height[pulse_cut].flatten()
cleanWidth = p_width[pulse_cut].flatten()
cleanPulseClass = p_class[pulse_cut].flatten()

cleanAFS2l = p_afs_2l[pulse_cut].flatten()
cleanAFS50 = p_afs_50[pulse_cut].flatten()
cleanRiseTime = tscale*(cleanAFS50-cleanAFS2l)

cleanAreaCh = p_area_ch[pulse_cut] # pulse_cut gets broadcast to the right shape
cleanAreaChFrac = p_area_ch_frac[pulse_cut]
cleanAreaTop = p_area_top[pulse_cut].flatten()
cleanAreaBottom = p_area_bottom[pulse_cut].flatten()
cleanAreaTBA = p_area_tba[pulse_cut].flatten()

s1_cut = pulse_cut*cut_dict['S1']
cleanS1Area = p_area[s1_cut].flatten()
cleanS1RiseTime = tscale*(p_afs_50[s1_cut]-p_afs_2l[s1_cut] ).flatten()
cleanS1AreaChFrac = p_area_ch_frac[s1_cut]
#cleanS1AreaChFrac = p_area_ch_frac[:, s1_cut].reshape(n_channels-1, -1)
cleanS1TBA = p_area_tba[s1_cut].flatten()

s2_cut = pulse_cut*cut_dict['S2']
cleanS2Area = p_area[s2_cut].flatten()
cleanS2RiseTime = tscale*(p_afs_50[s2_cut]-p_afs_2l[s2_cut] ).flatten()
cleanS2AreaChFrac = p_area_ch_frac[s2_cut]
#cleanS2AreaChFrac = p_area_ch_frac[:, s2_cut].reshape(n_channels-1, -1)
cleanS2TBA = p_area_tba[s2_cut].flatten()

# Quantities for plotting only events with n number of pulses, not just all of them
# May still contain empty pulses
howMany = n_pulses < 1000 # How many pulses you do want
nArea = p_area[howMany,:]
nMax = p_max_height[howMany,:]
nmin = p_min_height[howMany,:]
nWidth = p_width[howMany,:]

na2l = p_afs_2l[howMany]
na50 = p_afs_50[howMany]


# Event level quantities 
event_cut_dict = {}
event_cut_dict["SS"] = drift_Time > 0 
event_cut_dict["MS"] = (n_s1 == 1)*(n_s2 > 1)*s1_before_s2

event_cut_name = "SS"
event_cut = event_cut_dict[event_cut_name] 
cleanSumS1 = sum_s1_area[event_cut]
cleanSumS2 = sum_s2_area[event_cut]
cleanDT = tscale*drift_Time[event_cut]


t_end_recon = time.time()
print('time to reconstruct: ', t_end_recon - t_end_wfm_fill)


# =============================================================
# =============================================================
# now make plots of interesting pulse quantities

# Plots of all pulses combined (after cuts)
pl.figure()
pl.hist(cleanAreaTBA, bins=100, range=(-1,1) )
pl.axvline(x=np.mean(cleanAreaTBA), ls='--', color='r')
pl.xlabel("TBA")
if save_pulse_plots: pl.savefig(data_dir+"TBA_"+pulse_cut_name+".png")
#pl.show() 

pl.figure()
pl.yscale("log")
pl.hist(cleanRiseTime, 100)
pl.axvline(x=np.mean(cleanRiseTime), ls='--', color='r')
pl.xlabel("Rise time, 50-2 (us)")
if save_pulse_plots: pl.savefig(data_dir+"RiseTime_"+pulse_cut_name+".png")
#pl.show()

pl.figure()
#pl.yscale("log")
pl.hist(np.log10(cleanArea), 100)
pl.axvline(x=np.mean(np.log10(cleanArea)), ls='--', color='r')
pl.xlabel("log10 Pulse area (phd)")
if save_pulse_plots: pl.savefig(data_dir+"log10PulseArea_"+pulse_cut_name+".png")

pl.figure()
pl.yscale("log")
pl.hist(cleanPulseClass )
pl.xlabel("Pulse Class")
if save_pulse_plots: pl.savefig(data_dir+"PulseClass_"+pulse_cut_name+".png")


pl.figure()
pl.scatter(cleanAreaTBA, cleanRiseTime, s = 1.2, c = pulse_class_colors[cleanPulseClass])
pl.xlim(-1,1)
pl.ylabel("Rise time, 50-2 (us)")
pl.xlabel("TBA")
if save_pulse_plots: pl.savefig(data_dir+"RiseTime_vs_TBA_"+pulse_cut_name+".png")

pl.figure()
pl.xscale("log")
pl.scatter(cleanArea, cleanRiseTime, s = 1.2, c = pulse_class_colors[cleanPulseClass])
pl.ylabel("Rise time, 50-2 (us)")
pl.xlabel("Pulse area (phd)")
#pl.xlim(0.7*min(p_area.flatten()), 1.5*max(p_area.flatten()))
if save_pulse_plots: pl.savefig(data_dir+"RiseTime_vs_PulseArea_"+pulse_cut_name+".png")


# Plots of all S1 or all S2 pulses
pl.figure()
for j in range(0, n_channels-1):
    pl.subplot(4,2,j+1)
    pl.hist(cleanS1AreaChFrac[:,j],bins=100,range=(0,1))
    pl.axvline(x=np.mean(cleanS1AreaChFrac[:,j]), ls='--', color='r')
    #print("S1 ch {0} area frac mean: {1}".format(j,np.mean(cleanS1AreaChFrac[:,j])))
    #pl.yscale('log')
    pl.xlabel("S1 area fraction")
    pl.title('Ch '+str(j))
if save_S1S2_plots: pl.savefig(data_dir+"S1_ch_area_frac_"+pulse_cut_name+".png")

pl.figure()
for j in range(0, n_channels-1):
    pl.subplot(4,2,j+1)
    pl.hist(cleanS2AreaChFrac[:,j],bins=100,range=(0,1))
    pl.axvline(x=np.mean(cleanS2AreaChFrac[:,j]), ls='--', color='r')
    #print("S2 ch {0} area frac mean: {1}".format(j,np.mean(cleanS2AreaChFrac[:,j])))
    #pl.yscale('log')
    pl.xlabel("S2 area fraction")
    pl.title('Ch '+str(j))
if save_S1S2_plots: pl.savefig(data_dir+"S2_ch_area_frac_"+pulse_cut_name+".png")

pl.figure()
pl.hist(cleanS1TBA, bins=100, range=(-1,1) )
pl.axvline(x=np.mean(cleanS1TBA), ls='--', color='r')
pl.xlabel("S1 TBA")
if save_S1S2_plots: pl.savefig(data_dir+"S1TBA_"+pulse_cut_name+".png")

pl.figure()
pl.hist(cleanS2TBA, bins=100, range=(-1,1) )
pl.axvline(x=np.mean(cleanS2TBA), ls='--', color='r')
pl.xlabel("S2 TBA")
if save_S1S2_plots: pl.savefig(data_dir+"S2TBA_"+pulse_cut_name+".png")

pl.figure()
pl.hist(np.log10(cleanS1Area), 100)
pl.axvline(x=np.mean(np.log10(cleanS1Area)), ls='--', color='r')
pl.xlabel("log10 S1 area")
if save_S1S2_plots: pl.savefig(data_dir+"log10_S1_"+pulse_cut_name +".png")

pl.figure()
pl.hist(np.log10(cleanS2Area), 100)
pl.axvline(x=np.mean(np.log10(cleanS2Area)), ls='--', color='r')
pl.xlabel("log10 S2 area")
if save_S1S2_plots: pl.savefig(data_dir+"log10_S2_"+pulse_cut_name +".png")

pl.figure()
pl.hist(cleanS1Area, 500)
pl.axvline(x=np.mean(cleanS1Area), ls='--', color='r')
pl.xlabel("S1 area (phd)")
if save_S1S2_plots: pl.savefig(data_dir+"S1_"+pulse_cut_name +".png")

pl.figure()
pl.hist(cleanS2Area, 500)
pl.axvline(x=np.mean(cleanS2Area), ls='--', color='r')
pl.xlabel("S2 area (phd)")
if save_S1S2_plots: pl.savefig(data_dir+"S2_"+pulse_cut_name +".png")

# Plots of event-level variables
pl.figure()
pl.scatter(cleanSumS1, np.log10(cleanSumS2), s = 1)
pl.xlabel("Sum S1 area (phd)")
pl.ylabel("log10 Sum S2 area")
if save_event_plots: pl.savefig(data_dir+"log10_SumS2_vs_SumS1_"+event_cut_name +".png")

pl.figure() # Only ever plot this for SS events?
pl.hist(cleanDT, 100)
pl.axvline(x=np.mean(cleanDT), ls='--', color='r')
pl.xlabel("Drift time (us)")
if save_event_plots: pl.savefig(data_dir+"DriftTime_"+event_cut_name +".png")

pl.figure() # Only ever plot this for SS events?
pl.scatter(cleanDT, cleanSumS2)
pl.xlabel("Drift time (us)")
pl.ylabel("Sum S2 area")
# Calculate mean vs drift bin
drift_bins=np.linspace(0,10,100)
drift_ind=np.digitize(cleanDT, bins=drift_bins)
s2_medians=np.zeros(np.shape(drift_bins))
s2_std_err=np.ones(np.shape(drift_bins))*0#10000
for i_bin in range(len(drift_bins)):
    found_i_bin = np.where(drift_ind==i_bin)
    s2_area_i_bin = cleanSumS2[found_i_bin]
    if len(s2_area_i_bin) < 1: continue
    s2_medians[i_bin]=np.median(s2_area_i_bin) # Median instead of mean, better at ignoring outliers
    s2_std_err[i_bin]=np.std(s2_area_i_bin)/np.sqrt(len(s2_area_i_bin))
pl.errorbar(drift_bins, s2_medians, yerr=s2_std_err, linewidth=3, elinewidth=3, capsize=5, capthick=4, color='red')
pl.ylim(bottom=0)
if save_event_plots: pl.savefig(data_dir+"SumS2_vs_DriftTime_"+event_cut_name +".png")



#cleandt = dt[dt > 0]
#pl.figure()
#pl.hist(tscale*cleandt.flatten(), 100)
#pl.xlabel("dt")

# pl.figure()
# pl.scatter(small_weird_areas, big_weird_areas, s=7)
# pl.xlabel("Small Pulse Area (phd)")
# pl.ylabel("Big Pulse Area (phd)")

pl.show()
