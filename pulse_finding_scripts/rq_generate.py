
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import time
import sys

import PulseFinderScipy as pf
import PulseQuantities as pq
import PulseClassification as pc

#data_dir = "G:/.shortcut-targets-by-id/11qeqHWCbcKfFYFQgvytKem8rulQCTpj8/crystalize/data/data-202103/031121/Po_2.8g_3.0c_0.78bar_circ_30min_1312/"
#data_dir = "/home/xaber/caen/wavedump-3.8.2/data/041921/Po_2.8g_3.0c_0.72bar_circ_20min_0928/"
#data_dir = "G:/My Drive/crystalize/data/data-202104/041421/Po_2.8g_3.0c_1.1bar_circ_60min_1747/"
with open("path.txt", 'r') as path:
    data_dir = path.read()

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

save_avg_wfm = False # get the average waveform passing some cut and save to file

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
ns_to_sample = 1024./2000. #convert spe size unit from mV*ns to mV*sample
spe = {}
with open(data_dir+"spe.txt", 'r') as file:
    for line in file:
        (key, val) = line.split()
        spe[key] = float(val)*ns_to_sample

chA_spe_size = spe["ch0"]#29.02
chB_spe_size = spe["ch1"]#30.61
chC_spe_size = spe["ch2"]#28.87
chD_spe_size = spe["ch3"]#28.86*1.25 # scale factor (0.7-1.4) empirical as of Dec 9, 2020
chE_spe_size = spe["ch4"]#30.4
chF_spe_size = spe["ch5"]#30.44
chG_spe_size = spe["ch6"]#30.84
chH_spe_size = spe["ch7"]#30.3*1.8 # scale factor (1.6-2.2) empirical as of Dec 9, 2020
spe_sizes = [chA_spe_size, chB_spe_size, chC_spe_size, chD_spe_size, chE_spe_size, chF_spe_size, chG_spe_size, chH_spe_size]

# ==================================================================

#load in raw data

t_start = time.time()

block_size = 3000
n_block = 100
max_evts = n_block*block_size#5000  # 25000 # -1 means read in all entries; 25000 is roughly the max allowed in memory on the DAQ computer
max_pts = -1  # do not change
if max_evts > 0:
    max_pts = wsize * max_evts
load_dtype = "int16"

# pulse RQs to save

# RQs to add:
# Pulse level: channel areas (fracs; max fracs), TBA, rise time? (just difference of AFTs...)
# Event level: drift time; S1, S2 area
# Pulse class (S1, S2, other)
# max number of pulses per event
max_pulses = 4
p_start = np.zeros(( max_evts, max_pulses), dtype=int)
p_end   = np.zeros(( max_evts, max_pulses), dtype=int)
p_found = np.zeros(( max_evts, max_pulses), dtype=int)

#center of mass
center_top_x = np.zeros(( max_evts, max_pulses))
center_top_y = np.zeros(( max_evts, max_pulses))
center_bot_x = np.zeros(( max_evts, max_pulses))
center_bot_y = np.zeros(( max_evts, max_pulses))

p_area = np.zeros(( max_evts, max_pulses))
p_max_height = np.zeros(( max_evts, max_pulses))
p_min_height = np.zeros(( max_evts, max_pulses))
p_width = np.zeros(( max_evts, max_pulses))

p_afs_2l = np.zeros((max_evts, max_pulses) )
p_afs_2r = np.zeros((max_evts, max_pulses) )
p_afs_1 = np.zeros((max_evts, max_pulses) )
p_afs_25 = np.zeros((max_evts, max_pulses) )
p_afs_50 = np.zeros((max_evts, max_pulses) )
p_afs_75 = np.zeros((max_evts, max_pulses) )
p_afs_99 = np.zeros((max_evts, max_pulses) )
            
p_hfs_10l = np.zeros((max_evts, max_pulses) )
p_hfs_50l = np.zeros((max_evts, max_pulses) )
p_hfs_10r = np.zeros((max_evts, max_pulses) )
p_hfs_50r = np.zeros((max_evts, max_pulses) )

p_mean_time = np.zeros((max_evts, max_pulses) )
p_rms_time = np.zeros((max_evts, max_pulses) )

# Channel level (per event, per pulse, per channel)
p_start_ch = np.zeros((max_evts, max_pulses, n_channels-1), dtype=int)
p_end_ch = np.zeros((max_evts, max_pulses, n_channels-1), dtype=int )
p_area_ch = np.zeros((max_evts, max_pulses, n_channels-1) )
p_area_ch_frac = np.zeros((max_evts, max_pulses, n_channels-1) )

p_area_top = np.zeros((max_evts, max_pulses))
p_area_bottom = np.zeros((max_evts, max_pulses))
p_tba = np.zeros((max_evts, max_pulses))

p_class = np.zeros((max_evts, max_pulses), dtype=int)

# Event-level variables
n_pulses = np.zeros(max_evts, dtype=int)

n_s1 = np.zeros(max_evts, dtype=int)
n_s2 = np.zeros(max_evts, dtype=int)
sum_s1_area = np.zeros(max_evts)
sum_s2_area = np.zeros(max_evts)
drift_Time = np.zeros(max_evts)
drift_Time_AS = np.zeros(max_evts) # for multi-scatter drift time, defined by the first S2. 
s1_before_s2 = np.zeros(max_evts, dtype=bool)

n_wfms_summed = 0
avg_wfm = np.zeros(wsize)

# Temporary, for testing low area, multiple-S1 events
dt = np.zeros(max_evts)
small_weird_areas = np.zeros(max_evts)
big_weird_areas = np.zeros(max_evts)

n_golden = 0
inn=""

inn="" # used to control hand scan

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
    if n_events == 0: break
        
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

    
#check mark

    print("Running pulse finder on {:d} events...".format(n_events))

    # use for coloring pulses
    pulse_class_colors = np.array(['blue', 'green', 'red', 'magenta', 'darkorange'])
    pulse_class_labels = np.array(['Other', 'S1-like LXe', 'S1-like gas', 'S2-like', 'Merged S1/S2'])
    pc_legend_handles=[]
    for class_ind in range(len(pulse_class_labels)):
        pc_legend_handles.append(mpl.patches.Patch(color=pulse_class_colors[class_ind], label=str(class_ind)+": "+pulse_class_labels[class_ind]))

    for i in range(j*block_size, j*block_size+n_events):
        if i%500==0: print("Event #",i)
        
        # Find pulse locations; other quantities for pf tuning/debugging
        start_times, end_times, peaks, data_conv, properties = pf.findPulses( v_bls_matrix_all_ch[-1,i-j*block_size,:], max_pulses )


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
            

        # More precisely estimate baselines immediately before each pulse
        baselines_precise = pq.GetBaselines(p_start[i,:n_pulses[i]], p_end[i,:n_pulses[i]], v_bls_matrix_all_ch[:,i-j*block_size,:])

        # Calculate interesting quantities, only for pulses that were found
        for pp in range(n_pulses[i]):
            # subtract out more precise estimate of baseline for better RQ estimates
            baselines_pulse = baselines_precise[pp] # array of baselines per channel, for this pulse
            v_pulse_bls = np.array([ch_j - baseline_j for (ch_j, baseline_j) in zip(v_bls_matrix_all_ch[:,i-j*block_size,:], baselines_pulse)])

            # Version w/o pulse-level baseline subtraction
            #v_pulse_bls = v_bls_matrix_all_ch[:,i-j*block_size,:]

            # copied from above, for reference
            #sum_data = v_matrix_all_ch[-1][i, :] - sum_baseline
            #ch_data = [ch_j[i, :] - baseline_j for (ch_j, baseline_j) in zip(v_matrix_all_ch, baselines)]

            # Area, max & min heights, width, pulse mean & rms
            p_area[i,pp] = pq.GetPulseArea(p_start[i,pp], p_end[i,pp], v_pulse_bls[-1] )
            p_max_height[i,pp] = pq.GetPulseMaxHeight(p_start[i,pp], p_end[i,pp], v_pulse_bls[-1] )
            p_min_height[i,pp] = pq.GetPulseMinHeight(p_start[i,pp], p_end[i,pp], v_pulse_bls[-1] )
            p_width[i,pp] = p_end[i,pp] - p_start[i,pp]
            #(p_mean_time[i,pp], p_rms_time[i,pp]) = pq.GetPulseMeanAndRMS(p_start[i,pp], p_end[i,pp], v_bls_matrix_all_ch[-1,i,:])

            # Area and height fractions      
            (p_afs_2l[i,pp], p_afs_1[i,pp], p_afs_25[i,pp], p_afs_50[i,pp], p_afs_75[i,pp], p_afs_99[i,pp]) = pq.GetAreaFraction(p_start[i,pp], p_end[i,pp], v_pulse_bls[-1] )
            (p_hfs_10l[i,pp], p_hfs_50l[i,pp], p_hfs_10r[i,pp], p_hfs_50r[i,pp]) = pq.GetHeightFractionSamples(p_start[i,pp], p_end[i,pp], v_pulse_bls[-1] )
        
            # Areas for individual channels and top bottom
            p_area_ch[i,pp,:] = pq.GetPulseAreaChannel(p_start[i,pp], p_end[i,pp], v_pulse_bls )
            p_area_ch_frac[i,pp,:] = p_area_ch[i,pp,:]/p_area[i,pp]
            p_area_top[i,pp] = sum(p_area_ch[i,pp,top_channels])
            p_area_bottom[i,pp] = sum(p_area_ch[i,pp,bottom_channels])
            p_tba[i, pp] = (p_area_top[i, pp] - p_area_bottom[i, pp]) / (p_area_top[i, pp] + p_area_bottom[i, pp])
            center_top_x[i,pp] = (p_area_ch[i,pp,1]+p_area_ch[i,pp,3]-p_area_ch[i,pp,0]-p_area_ch[i,pp,2])/p_area_top[i,pp]
            center_top_y[i,pp] = (p_area_ch[i,pp,0]+p_area_ch[i,pp,1]-p_area_ch[i,pp,2]-p_area_ch[i,pp,3])/p_area_top[i,pp]
            center_bot_x[i,pp] = (p_area_ch[i,pp,5]+p_area_ch[i,pp,7]-p_area_ch[i,pp,4]-p_area_ch[i,pp,6])/p_area_bottom[i,pp]
            center_bot_y[i,pp] = (p_area_ch[i,pp,4]+p_area_ch[i,pp,5]-p_area_ch[i,pp,6]-p_area_ch[i,pp,7])/p_area_bottom[i,pp]
            
        # Pulse classifier, work in progress
        p_class[i,:] = pc.ClassifyPulses(p_tba[i, :], (p_afs_50[i, :]-p_afs_2l[i, :])*tscale, n_pulses[i], p_area[i,:])

        # Event level analysis. Look at events with both S1 and S2.
        index_s1 = (p_class[i,:] == 1) + (p_class[i,:] == 2) # S1's
        index_s2 = (p_class[i,:] == 3) + (p_class[i,:] == 4) # S2's
        n_s1[i] = np.sum(index_s1)
        n_s2[i] = np.sum(index_s2)
        
        if n_s1[i] > 0:
            sum_s1_area[i] = np.sum(p_area[i, index_s1])
        if n_s2[i] > 0:
            sum_s2_area[i] = np.sum(p_area[i, index_s2])
        if n_s1[i] == 1:
            if n_s2[i] == 1:
                drift_Time[i] = tscale*(p_start[i, np.argmax(index_s2)] - p_start[i, np.argmax(index_s1)])
                drift_Time_AS[i] = tscale*(p_start[i, np.argmax(index_s2)] - p_start[i, np.argmax(index_s1)])
            if n_s2[i] > 1:
                s1_before_s2[i] = np.argmax(index_s1) < np.argmax(index_s2) 
                drift_Time_AS[i] = tscale*(p_start[i, np.argmax(index_s2)] - p_start[i, np.argmax(index_s1)]) #For multi-scatter events. 
        
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
        
        # Condition to skip the individual plotting, hand scan condition
        #plotyn = drift_Time[i]<2 and drift_Time[i]>0 and np.any((p_tba[i,:]>-0.75)*(p_tba[i,:]<-0.25)*(p_area[i,:]<3000)*(p_area[i,:]>1400))#np.any((p_tba[i,:]>-0.91)*(p_tba[i,:]<-0.82)*(p_area[i,:]<2800)*(p_area[i,:]>1000))# True#np.any(p_class[i,:]==4)#False#np.any(p_area[i,:]>1000) and 
        #plotyn = drift_Time[i]>2.5 and (center_bot_y[i,0]**2+center_bot_x[i,0]**2) <0.1
        plotyn = True #np.any((p_class[i,:] == 3) + (p_class[i,:] == 4))#np.any((p_tba[i,:]>-0.75)*(p_tba[i,:]<-0.25)*(p_area[i,:]<3000)*(p_area[i,:]>1000))
        #plotyn = np.any((np.log10(p_area[i,:])>3.2)*(np.log10(p_area[i,:])<3.4) )#False#np.any((p_tba[i,:]>-0.75)*(p_tba[i,:]<-0.25)*(p_area[i,:]<3000)*(p_area[i,:]>1000))
        # Pulse area condition
        areaRange = np.sum((p_area[i,:] < 50)*(p_area[i,:] > 5))
        if areaRange > 0:
            dt[i] = abs(p_start[i,1] - p_start[i,0]) # For weird double s1 data
            weird_areas =[p_area[i,0], p_area[i,1] ]
            small_weird_areas[i] = min(weird_areas)
            big_weird_areas[i] = max(weird_areas)

        # Condition to include a wfm in the average
        add_wfm = np.any((p_area[i,:]>5000)*(p_tba[i,:]<-0.75))*(n_s1[i]==1)*(n_s2[i]==0)
        if add_wfm and save_avg_wfm:
            plotyn = add_wfm # in avg wfm mode, plot the events which will go into the average
            avg_wfm += v_bls_matrix_all_ch[-1,i-j*block_size,:]
            n_wfms_summed += 1

        # Both S1 and S2 condition
        s1s2 = (n_s1[i] == 1)*(n_s2[i] == 1)

        if inn == 's': sys.exit()
        
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
                
                pl.plot(t,v_bls_matrix_all_ch[i_chan,i-j*block_size,:],color=ch_colors[i_chan],label=ch_labels[i_chan])
                #pl.plot( x, v_bls_matrix_all_ch[i_chan,i,:],color=ch_colors[i_chan],label=ch_labels[i_chan] )
                pl.xlim([trigger_time_us-8,trigger_time_us+8])
                #pl.xlim([wsize/2-4000,wsize/2+4000])
                pl.ylim([-5, 3000/chA_spe_size])
                pl.xlabel('Time (us)')
                #pl.xlabel('Samples')
                pl.ylabel('phd/sample')
                pl.legend()
            
            ax = pl.subplot2grid((2,2),(1,0),colspan=2)
            #pl.plot(t,v_bls_matrix_all_ch[-1,i,:],'blue')
            pl.plot( x*tscale, v_bls_matrix_all_ch[-1,i-j*block_size,:],'blue' )
            #pl.xlim([0,wsize])
            pl.xlim([0,event_window])
            pl.ylim( [-1, 1.01*np.max(v_bls_matrix_all_ch[-1,i-j*block_size,:])])
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
                pl.plot(t_matrix[i-j*block_size, :], data_conv, 'red')
                pl.plot(t_matrix[i-j*block_size, :], np.tile(0., np.size(data_conv)), 'gray')
                pl.vlines(x=peaks*tscale, ymin=data_conv[peaks] - properties["prominences"],
                           ymax=data_conv[peaks], color="C1")
                pl.hlines(y=properties["width_heights"], xmin=properties["left_ips"]*tscale,
                           xmax=properties["right_ips"]*tscale, color="C1")
                #print("pulse heights: ", data_conv[peaks] )
                #print("prominences:", properties["prominences"])

            pl.draw()
            pl.show(block=0)
            inn = input("Press enter to continue, q to stop plotting, evt # to skip to # (forward only)")
            fig.clf()
            
    # end of pulse finding and plotting event loop

if save_avg_wfm:
    avg_wfm /= n_wfms_summed
    np.savetxt(data_dir+'average_waveform.txt',avg_wfm)
    print("Average waveform saved")

n_events = i
print("total number of events processed:", n_events)

#create a dictionary with all RQs
list_rq = {}
list_rq['center_top_x'] = center_top_x
list_rq['center_top_y'] = center_top_y
list_rq['center_bot_x'] = center_bot_x
list_rq['center_bot_y'] = center_bot_y
list_rq['n_s1'] = n_s1
list_rq['n_s2'] = n_s2
list_rq['s1_before_s2'] = s1_before_s2
list_rq['n_pulses'] = n_pulses
list_rq['n_events'] = n_events
list_rq['p_area'] = p_area
list_rq['p_class'] = p_class
list_rq['drift_Time'] = drift_Time
list_rq['drift_Time_AS'] = drift_Time_AS
list_rq['p_max_height'] = p_max_height
list_rq['p_min_height'] = p_min_height
list_rq['p_width'] = p_width
list_rq['p_afs_2l'] = p_afs_2l
list_rq['p_afs_50'] = p_afs_50
list_rq['p_area_ch'] = p_area_ch
list_rq['p_area_ch_frac'] = p_area_ch_frac
list_rq['p_area_top'] = p_area_top
list_rq['p_area_bottom'] = p_area_bottom
list_rq['p_tba'] = p_tba
list_rq['p_start'] = p_start
list_rq['p_end'] = p_end
list_rq['sum_s1_area'] = sum_s1_area
list_rq['sum_s2_area'] = sum_s2_area
#list_rq[''] =    #add more rq

#remove zeros in the end of each RQ array. 
for rq in list_rq.keys():
    if rq != 'n_events':
        list_rq[rq] = list_rq[rq][:n_events]

rq = open(data_dir + "rq.npz",'wb')
np.savez(rq, **list_rq)
rq.close()
