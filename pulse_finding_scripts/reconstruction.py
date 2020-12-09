
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
#data_dir="../data/bkg_3.5g_3.9c_27mV_7_postrecover2_5min/"
#data_dir="../data/bkg_3.5g_3.9c_27mV_1_5min/"
#data_dir="../data/fewevts/"
#data_dir="../data/po_5min/"
#data_dir = "C:/Users/ryanm/Documents/Research/Data/bkg_3.5g_3.9c_27mV_6_postrecover_5min/"
data_dir = "C:/Users/ryanm/Documents/Research/Data/bkg_2.8g_3.2c_25mV_1_1.6_circ_0.16bottle_5min/"
#"C:/Users/swkra/Desktop/Jupyter temp/data-202009/091720/bkg_3.5g_3.9c_27mV_7_postrecover2_5min/"

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

t0 = time.time()

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

# Note: if max_evts != -1, we won't load in all events in the dataset
n_events = int(v_matrix.shape[0])
    
# perform baseline subtraction:
# for now, using first 2 µs of event
baseline_start = int(0./tscale)
baseline_end = int(2./tscale)

# baseline subtracted (bls) waveforms saved in this matrix:
v_bls_matrix_all_ch = np.zeros( np.shape(v_matrix_all_ch) ) # dims are (chan #, evt #, sample #)

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

start = np.zeros( max_pulses, dtype=np.int)
end   = np.zeros( max_pulses, dtype=np.int)
found = np.zeros( max_pulses, dtype=np.int)

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

n_pulses = np.zeros(n_events, dtype=np.int)

p_start_ch = np.zeros((n_channels-1, n_events, max_pulses), dtype=np.int)
p_end_ch = np.zeros((n_channels-1, n_events, max_pulses), dtype=np.int )
p_area_ch = np.zeros((n_channels-1, n_events, max_pulses) )
p_area_ch_frac = np.zeros((n_channels-1, n_events, max_pulses) )
p_area_top = np.zeros((n_events, max_pulses))
p_area_bottom = np.zeros((n_events, max_pulses))
p_area_tba = np.zeros((n_events, max_pulses))

p_class = np.zeros((n_events, max_pulses), dtype=np.int)

inn=""

#make copy of waveforms:
v_bls_matrix_all_ch_cpy = v_bls_matrix_all_ch.copy() # Do we need this? Not zeroing out anymore...
print("Running pulse finder on {:d} events...".format(n_events))

for i in range(0, n_events):
    if i%100==0: print("Event #",i)
    
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
    #        p_start_ch[j,i,k] = start_times_ch[k]
    #        p_end_ch[j,i,k] = end_times_ch[k]
        

    # Calculate interesting quantities
    for pp in range(max_pulses):

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
        p_area_ch[:,i,pp] = pq.GetPulseAreaChannel(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[:,i,:] )
        p_area_ch_frac[:,i,pp] = p_area_ch[:,i,pp]/p_area[i,pp]
        p_area_top[i,pp] = sum(p_area_ch_frac[0:3,i,pp])
        p_area_bottom[i,pp] = sum(p_area_ch_frac[4:7,i,pp])
        p_area_tba[i,pp] = (p_area_top[i,pp] - p_area_bottom[i,pp])/(p_area_top[i,pp] + p_area_bottom[i,pp])

        
    # Pulse classifier, work in progress
    p_class[i,:] = pc.ClassifyPulses(p_area_tba[i,:], p_afs_2l[i,:], p_afs_50[i,:], tscale)


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
    plotyn = True #False

    if True and not inn == 'q' and plot_event_ind == i and plotyn:

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
        pl.ylim( [-5, 1.01*np.max(v_bls_matrix_all_ch[-1,i,:])])
        pl.xlabel('Time (us)')
        #pl.xlabel('Samples')
        pl.ylabel('phd/sample')
        pl.title("Sum, event "+ str(i))
        pl.grid(b=True,which='major',color='lightgray',linestyle='--')
        triggertime_us = (t[-1]*0.2)
        
        colors = np.array(['blue','green','red','purple','black','magenta','darkorange'])
        for pulse in range(len(start_times)):
            ax.axvspan( start_times[pulse]*tscale, end_times[pulse]*tscale, alpha=0.25, color=colors[p_class[i,pulse] ] )
        
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


# Removing empty pulses
cleanPulses = p_area > 0

cleanArea = p_area[cleanPulses]
cleanMax = p_max_height[cleanPulses]
cleanMin = p_min_height[cleanPulses]
cleanWidth = p_width[cleanPulses]

cleanAFS2l = p_afs_2l[cleanPulses]
cleanAFS50 = p_afs_50[cleanPulses]

cleanAreaCh = p_area_ch[:,cleanPulses]
cleanAreaChFrac = p_area_ch_frac[:,cleanPulses]
cleanAreaTop = p_area_top[cleanPulses]
cleanAreaBottom = p_area_bottom[cleanPulses]
cleanAreaTBA = p_area_tba[cleanPulses]


# Quantities for plotting only events with n number of pulses, not just all of them
# May still contain empty pulses
howMany = n_pulses < 1000 # How many pulses you do want
nArea = p_area[howMany,:]
nMax = p_max_height[howMany,:]
nmin = p_min_height[howMany,:]
nWidth = p_width[howMany,:]

na2l = p_afs_2l[howMany]
na50 = p_afs_50[howMany]


# Quantities for plotting a specific class of pulses
class1 = p_class == 1
tba1 = p_area_tba[class1]
afs2l1 = p_afs_2l[class1]
afs501 = p_afs_50[class1]


t1 = time.time()
print('time to complete: ',t1-t0)


# =============================================================
# =============================================================
# now make plots of interesting pulse quantities



pl.figure()
pl.hist(cleanAreaTBA.flatten(), 100 )
pl.xlabel("TBA")
#pl.show() 

pl.figure()
pl.hist(tscale*(cleanAFS50.flatten()-cleanAFS2l.flatten() ), 100)
pl.xlabel("50 - 2")
pl.yscale("log")
#pl.show()

pl.figure()
pl.scatter(p_area_tba.flatten(), tscale*(p_afs_50.flatten() - p_afs_2l.flatten() ),  s = 1, c = colors[p_class.flatten()])
pl.ylabel("50 - 2")
pl.xlabel("TBA")

pl.figure()
pl.scatter(p_area.flatten(), tscale*(p_afs_50.flatten() - p_afs_2l.flatten() ),  s = 1, c = colors[p_class.flatten()])
pl.ylabel("50 - 2")
pl.xlabel("Area")
pl.xscale("log")


pl.show()
