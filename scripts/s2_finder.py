


import numpy as np
#import pylab as pl
import matplotlib.pyplot as pl
import matplotlib as mpl
import scipy
from scipy.signal import find_peaks
from scipy.signal import argrelmin
import time
def smooth(data,t_min,window):
    avg_array=np.zeros(len(data))
    #print(avg_array)
    m=20
    for i in range(t_min, t_min+window):
        min_ind=i-m
        max_ind=i+m
        if min_ind<0: min_ind=0
        if max_ind>len(data): max_ind = len(data)
        avg_array[i]=(np.mean(data[min_ind:max_ind]))
    return(avg_array)
#def maxima(data):
    #y = data
    #peaks = argrelextrema(y, np.greater)
   #return(peaks)
#def minima(data):
    #z = data
    #valleys = argrelextrema(z, np.less)
    #return(valleys)

def pulse_finder_area_series(data,t_min_search,t_max_search,window):
# Assumes data is already baseline-subtracted
    max_area=-1
    max_ind=-1
    for i_start in range(t_min_search,t_max_search):
        area=np.sum(data[i_start:i_start+window])
        if area>max_area:
            max_area=area
            max_ind=i_start
    return (max_ind, max_area)
	
def pulse_finder_area(data,t_min_search,t_max_search,window):
# Assumes data is already baseline-subtracted
    if t_max_search < t_min_search+1:
        return (-1, -1)
    weights = np.repeat(1.0, window)#/window #to do avg instead of sum
    data_conv = np.convolve(data, weights, 'same')
	# Search only w/in search range, offset so that max_ind is the start of the window rather than the center
    max_ind=np.argmax(data_conv[int(t_min_search+window/2):int(t_max_search+window/2)])+int(t_min_search)
    return (max_ind, data_conv[max_ind+int(window/2)])

def pulse_bounds(data,t_min,window,start_frac,end_frac):
    
# Assumes data is already baseline-subtracted
    start_pos=-1
    end_pos=-1
    min_search = np.maximum(0,t_min)
    max_search = np.minimum(len(data)-1,t_min+window)
    peak_val=np.max(data[min_search:max_search])
    peak_pos=np.argmax(data[min_search:max_search])
    #print("peak_val: ",peak_val,"peak_pos: ",(peak_pos+min_search)*tscale)
    #print("min_search: ",min_search*tscale,"max_search: ",max_search*tscale)
    #start_frac: pulse starts at this fraction of peak height above baseline
    for i_start in range(min_search,max_search):
        if data[i_start]>max(peak_val*start_frac,6.0/chA_spe_size):
            start_pos=i_start
            break
    #end_frac: pulse ends at this fraction of peak height above baseline
    for i_start in range(max_search,min_search,-1):
        if data[i_start]>max(peak_val*end_frac,6.0/chA_spe_size):
            end_pos=i_start
            break
    
    return (start_pos, end_pos)
def merged_bounds(data, t_min, window, start_frac, end_frac):
    start_pos=-1
    end_pos=-1
    a = (np.diff(np.sign(np.diff(output))) > 0).nonzero()[0] + 1
    b = argrelmin(data)
    peak_v=np.max(data[t_min:t_min+window])
    print("b:", b)
    second = []
    print("a[0]:", a[0])
    for i in scipy.nditer(b):
        #if data[i]>0.2 and max(data[t_min:t_min+window]) < i : 
            #for i_start in range(t_min,t_min+window):
                #if data[i_start]>max(data[a[0]]*start_frac,4.5/chA_spe_size):
                    #start_pos=i_start
                    #break
        if data[i]>1:
            start_pos=i
            print("i:",i)
            break
 
    for j in a:
            
        if data[j]>0.2:
            second.append(j)
    for k in scipy.nditer(b):
        if max(a) > k:
            
            for z in second[1:]:       
       
    #end_frac: pulse ends at this fraction of peak height above baseline
                for i_start in range(t_min+window,t_min,-1):
                    if data[i_start]>max(data[z]*end_frac,4.5/chA_spe_size):
                        end_pos=i_start
                        break
        else:
            end_pos = b[0]
           
    return(start_pos, end_pos) 
# set plotting style
mpl.rcParams['font.size']=28
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[16.0,12.0]


data_dir="../../101520/bkg_2.8g_3.0c_25mV_5_afterfill_1.2bar_5min/"
event_window=25. # in us
wsize=int(500*event_window) # samples per waveform # 12500 for 25 us
max_evts=-1 #25000 # -1 means read in all entries; 25000 is roughly the max allowed in memory on the DAQ computer
max_pts=-1 # do not change
if max_evts>0:
    max_pts = wsize*max_evts
channel_0=np.fromfile(data_dir+"wave0.dat", dtype="int16", count=max_pts)
channel_1=np.fromfile(data_dir+"wave1.dat", dtype="int16", count=max_pts)
channel_2=np.fromfile(data_dir+"wave2.dat", dtype="int16", count=max_pts)
channel_3=np.fromfile(data_dir+"wave3.dat", dtype="int16", count=max_pts)
channel_4=np.fromfile(data_dir+"wave4.dat", dtype="int16", count=max_pts)
channel_5=np.fromfile(data_dir+"wave5.dat", dtype="int16", count=max_pts)
channel_6=np.fromfile(data_dir+"wave6.dat", dtype="int16", count=max_pts)
channel_7=np.fromfile(data_dir+"wave7.dat", dtype="int16", count=max_pts)

#channel_0=np.fromfile("../../Desktop/crystallize_data/t3-0805/A-thorium-4kv-t3.dat", dtype="int16")
#channel_1=np.fromfile("../../Desktop/crystallize_data/t3-0805/B-thorium-4kv-t3.dat", dtype="int16")
#channel_2=np.fromfile("../../Desktop/crystallize_data/t3-0805/C-thorium-4kv-t3.dat", dtype="int16")
#channel_3=np.fromfile("../../Desktop/crystallize_data/t3-0805/D-thorium-4kv-t3.dat", dtype="int16")

#channel_0=np.fromfile("A-thorium-3kv.dat", dtype="int16")
#channel_1=np.fromfile("B-thorium-3kv.dat", dtype="int16")
#channel_2=np.fromfile("C-thorium-3kv.dat", dtype="int16")
#channel_3=np.fromfile("D-thorium-3kv.dat", dtype="int16")


vscale=(2000.0/16384.0)
chA_spe_size = 29.02 # mV*samples/phd
V=vscale*channel_0/chA_spe_size # ch A, calib size 644 
# Ensure we have an integer number of events
V=V[:int(len(V)/wsize)*wsize]
chB_spe_size = 30.61
V_1=vscale*channel_1/chB_spe_size # convert from ADC counts to mV, then to phd/sample
V_1=V_1[:int(len(V)/wsize)*wsize]
chC_spe_size = 28.87
V_2=vscale*channel_2/chC_spe_size
V_2=V_2[:int(len(V)/wsize)*wsize]
chD_spe_size = 28.86
V_3=vscale*channel_3/chD_spe_size
V_3=V_3[:int(len(V)/wsize)*wsize]
chE_spe_size=30.4
V_4=vscale*channel_4/chE_spe_size
V_4=V_4[:int(len(V)/wsize)*wsize]
chF_spe_size=30.44
V_5=vscale*channel_5/chF_spe_size
V_5=V_5[:int(len(V)/wsize)*wsize]
chG_spe_size=30.84
V_6=vscale*channel_6/chG_spe_size
V_6=V_6[:int(len(V)/wsize)*wsize]
chH_spe_size=30.3
V_7=vscale*channel_7/chH_spe_size
V_7=V_7[:int(len(V)/wsize)*wsize]
n_channels=9 # including sum
v_matrix = V.reshape(int(V.size/wsize),wsize)
v1_matrix = V_1.reshape(int(V.size/wsize),wsize)
v2_matrix = V_2.reshape(int(V.size/wsize),wsize)
v3_matrix = V_3.reshape(int(V.size/wsize),wsize)
v4_matrix = V_4.reshape(int(V.size/wsize),wsize)
v5_matrix = V_5.reshape(int(V.size/wsize),wsize)
v6_matrix = V_6.reshape(int(V.size/wsize),wsize)
v7_matrix = V_7.reshape(int(V.size/wsize),wsize)
vsum_matrix = v_matrix+v1_matrix+v2_matrix+v3_matrix+v4_matrix+v5_matrix+v6_matrix+v7_matrix
v_matrix_all_ch=[v_matrix,v1_matrix,v2_matrix,v3_matrix,v4_matrix,v5_matrix,v6_matrix,v7_matrix,vsum_matrix]
x=np.arange(0, wsize, 1)
tscale=(8.0/4096.0)
t=tscale*x
t_matrix=np.repeat(t[np.newaxis,:], V.size/wsize, 0)
# One entry per channel
max_ind_array=np.zeros((v_matrix.shape[0],n_channels) )
max_val_array=np.zeros((v_matrix.shape[0],n_channels) )
integral_array=np.zeros((v_matrix.shape[0],n_channels) )
s2_integral_array=np.zeros((v_matrix.shape[0],n_channels) )
s1_ch_array=np.zeros((v_matrix.shape[0],n_channels-1) )
s2_ch_array=np.zeros((v_matrix.shape[0],n_channels-1) )
# One entry per event
s2_area_array=np.zeros(v_matrix.shape[0])
s1_area_array=np.zeros(v_matrix.shape[0])
s2_width_array=np.zeros(v_matrix.shape[0])
s2_height_array=np.zeros(v_matrix.shape[0])
s1_height_array=np.zeros(v_matrix.shape[0])
t_drift_array=np.zeros(v_matrix.shape[0])
s2_found_array=np.zeros(v_matrix.shape[0],dtype='bool')
s1_found_array=np.zeros(v_matrix.shape[0],dtype='bool')


# s2_area_array=[]
# s1_area_array=[]
# s2_width_array=[]
# t_drift_array=[]
# s2_found_array=[]
# s1_found_array=[]

inn=""
center = np.zeros(v_matrix.shape[0], dtype='bool')
print("Total events: ",v_matrix.shape[0])
for i in range(0, int(v_matrix.shape[0])):
    if i%100==0: print("Event #",i)
    t0=time.time()
    # for each channel
    #for j in range(0, n_channels):
        #i = input("Window number between 1 and " + str((V.size/wsize)) + ": ")
        
        #baseline=np.mean(v_matrix_all_ch[j][i,:500]) #avg ~1 us
        #print("baseline: ",baseline)

        
    # Look for events with S1 and S2 from summed channel
    n_top = int((n_channels-1)/2)
    top_channels=np.array(range(n_top),int)
    bottom_channels=np.array(range(n_top,2*n_top),int)
    post_trigger=0.5 # Was 0.2 for data before 11/22/19
    trigger_time_us=event_window*(1-post_trigger)
    trigger_time=int(trigger_time_us/tscale)
    t_offset=int(0.2/tscale)
    s1_window = int(0.5/tscale)
    s2_window = int(5.0/tscale)
    t_min_search=trigger_time-int(event_window/2.1/tscale)# was int(10./tscale)
    t_max_search=trigger_time+int(event_window/2.1/tscale)# was int(22./tscale)
    s1_thresh = 100 # in units of phd; compare to sum of waveform over multiple samples
    s1_range_thresh = 15/chA_spe_size
    s2_thresh = 75#150
    s2_start_frac=0.06 # s2 pulse starts at this fraction of peak height above baseline
    s2_end_frac=0.05 # s2 pulse starts at this fraction of peak height above baseline
    s1_start_frac=0.1
    s1_end_frac=0.1
    s1_max=s1_thresh
    s1_max_ind=-1
    s1_area=-1
    s1_height_range=-1
    s1_start_pos=-1
    s1_end_pos=-1
    s2_max=s2_thresh
    s2_max_ind=-1
    s2_area=-1
    #s2_start_pos=[]
    #s2_end_pos=[]
    s2_width=-1
    s2_height=-1
    s1_height=-1
    t_drift=-1
    s1_found=False
    s2_found=False
    s1_ch_area=[-1]*(n_channels-1)
    s2_ch_area=[-1]*(n_channels-1)
    fiducial = False
 
    baseline_start = int(0./tscale)
    baseline_end = int(2./tscale)
    t0=time.time()
    sum_baseline=np.mean(v_matrix_all_ch[-1][i,baseline_start:baseline_end]) #avg ~us, avoiding trigger
    baselines = [np.mean(ch_j[i,baseline_start:baseline_end] ) for ch_j in v_matrix_all_ch]
    #print("baseline:", baselines)
    #print("sum baseline:", sum_baseline)
    sum_data=v_matrix_all_ch[-1][i,:]-sum_baseline
    ch_data=[ch_j[i,:]-baseline_j for (ch_j,baseline_j) in zip(v_matrix_all_ch,baselines)]
    t0a=time.time()
    #print("ch baseline calc time: ",t0a-t0)
      
    # Do a moving average (sum) of the waveform with different time windows for s1, s2
	# Look for where this value is maximized
	# Look for the s2 using a moving average (sum) of the waveform over a wide window

    # Temporary! Changing just the S2-finding range to be AFTER the trigger (assuming we trigger on the S1)
    small_s2_mode = True
    if small_s2_mode:
        s2_search_min = trigger_time+int(1.0/tscale)
        s2_search_max = int(22.0/tscale)
        s2_max_ind, s2_max=pulse_finder_area(sum_data,s2_search_min,s2_search_max,s2_window)
    else:
        s2_max_ind, s2_max=pulse_finder_area(sum_data,t_min_search,t_max_search,s2_window)
    s2_found=s2_max>s2_thresh
    t1=time.time()
    
    
    #print("pulse finder time: ", t1-t0)
    #print("max area in s2 window: ",s2_max)
    #print("time of max area in s2 window: ",s2_max_ind," in us: ",s2_max_ind*tscale)
    #print("t_min_search: ",t_min_search*tscale,"t_max_search: ",t_max_search*tscale)
    
    #output = smooth(sum_data,t_min_search,t_min_search+s2_window)
    #local_minima = minima(sum_data)
    t2=time.time()
    #local_maxima = maxima(sum_data)
    #print(len(t_matrix[i,:]))
    #print('local_max:', local_maxima)
    #print("smooth time: ", t2-t1)
    #print(output)
    bad_bounds=False
    if s2_found: # Found a pulse (maybe an s2)
        # print("s2 window time: ",s2_max_ind*tscale,"s2 area max: ",s2_max)
        s2_start_pos, s2_end_pos=pulse_bounds(sum_data,s2_max_ind,s2_window,s2_start_frac,s2_end_frac) # had s2_max_ind-t_offset; why?
        #print(s2_start_pos, s2_end_pos)
              
        if(s2_start_pos==s2_end_pos):
            if(s2_start_pos == -1):
                print("could not find start or end of pulse for event ", i)
                bad_bounds=True
                s2_start_pos=s2_max_ind-3
                s2_end_pos=s2_max_ind+3
            else:
                print("pulse start = end = ",s2_start_pos," for event ",i)
                s2_start_pos=s2_start_pos-3
                s2_end_pos=s2_end_pos+3
        elif(s2_start_pos == -1):
            print("could not find start of pulse for event ", i)
            bad_bounds=True
            s2_start_pos=s2_max_ind-3
            s2_end_pos=s2_max_ind+3
            
        # Temporary! Add S2 range requirement to avoid picking up noise pulses
        if small_s2_mode and not bad_bounds:
            s2_height_range=np.max(sum_data[s2_start_pos:s2_end_pos])-np.min(sum_data[s2_start_pos:s2_end_pos]) 
            #print("range: {0}".format(s2_height_range))
            if s2_height_range<s1_range_thresh:
                #print("S2 fails height range cut (probably noise); range: {0}".format(s2_height_range))
                s2_found = False
            
        if bad_bounds: s2_found=False
            
        #s2_start_array, s2_end_array = merged_bounds(output,s2_max_ind-t_offset,s2_window,s2_start_frac,s2_end_frac)
        t3=time.time()
       
        #print('s2_start_array:', s2_start_array)
        #print('s2_end_array:', s2_end_array)
      
        #print("pulse bounds time: ", t3-t2)
        #print("s2 start: ",s2_start_pos*tscale," s2 end: ",s2_end_pos*tscale)
        s2_area=np.sum(sum_data[s2_start_pos:s2_end_pos])
        s2_ch_area = [np.sum(ch[s2_start_pos:s2_end_pos]) for ch in ch_data]
       
      
        s2_width=(s2_end_pos-s2_start_pos)*tscale
        s2_height=np.max(sum_data[s2_start_pos:s2_end_pos])
        
        #s2_area_array.append(s2_area)
        #s2_width_array.append(s2_width)
        #s2_found_array.append(s2_found)
        s2_mean = np.mean(s2_ch_area[:-1])
        s2_top_mean = np.mean(np.array(s2_ch_area)[top_channels])
        fiducial = True
       
        for k in range(0,len(s2_ch_area)-1):
            #print(s2_ch_area[k], s2_mean)
            if not (s2_ch_area[k] > 0.3*s2_top_mean and s2_ch_area[k] < 2*s2_top_mean):
                fiducial = False
                break
        #print("fiducial? ", fiducial)        
        center[i] = fiducial
         	
        # Now look for a prior s1
        #print("t_min_search: ",t_min_search*tscale,"t_max: ",(s2_start_pos-s1_window-t_offset)*tscale)
        s1_max_ind, s1_max=pulse_finder_area(sum_data,t_min_search,s2_start_pos-s1_window-t_offset,s1_window)
        #print("sum baseline: ",sum_baseline)
        #print("s1 area: ",s1_max)
        if s1_max>s1_thresh:
            #print("s1 window time: ",s1_max_ind*tscale,"s1 area max: ",s1_max)    
            s1_start_pos, s1_end_pos = pulse_bounds(sum_data,s1_max_ind,s1_window,s1_start_frac,s1_end_frac) # had s1_max_ind-t_offset; why?
            if s1_start_pos > -1 and s1_end_pos > s1_start_pos:
                #print(s1_start_pos)
                #print(s1_end_pos)
                # Check that we didn't accidentally find noise (related to poor baseline subtraction)
                s1_height_range=np.max(sum_data[s1_start_pos:s1_end_pos])-np.min(sum_data[s1_start_pos:s1_end_pos]) 
                s1_found = s1_height_range>s1_range_thresh
                #print("s1 start: ",s1_start_pos*tscale," s1 end: ",s1_end_pos*tscale)
                if 0.60<t_drift<0.70:
                    #print("s1_max_ind: ",s1_max_ind*tscale," s1_start_pos: ",s1_start_pos*tscale," tdrift: ",t_drift)
                    #print("s1 range: ",s1_height_range)
                    #print("baseline: ",sum_baseline)
                    pass
                if not s1_found:
                    print("under range, s1 range: ",s1_height_range)
                    pass
                else:    
                    t_drift=(s2_start_pos-s1_start_pos)*tscale
                    s1_area=np.sum(sum_data[s1_start_pos:s1_end_pos])
                    s1_ch_area = [np.sum(ch[s1_start_pos:s1_end_pos]) for ch in ch_data]
                    s1_height=np.max(sum_data[s1_start_pos:s1_end_pos])
                    #s1_found_array.append(s1_found)       
	            #s1_area_array.append(s1_area)
	            #t_drift_array.append(t_drift)
    t4=time.time()
    #print("remaining time: ",t4-t3)
	


    s2_area_array[i]=s2_area
    s2_width_array[i]=s2_width
    s2_height_array[i]=s2_height
    s1_height_array[i]=s1_height
    s1_area_array[i]=s1_area
    t_drift_array[i]=t_drift
    s1_found_array[i]=s1_found
    s2_found_array[i]=s2_found
    for j in range(n_channels-1):
        s1_ch_array[i,j] = s1_ch_area[j]
        s2_ch_array[i,j] = s2_ch_area[j]
	 
    # once per event
    #if s1_max_ind>-1 and not s1_height_range>s1_range_thresh:
    #if 1.5<t_drift:
    #if 1.08<t_drift<1.12:
    #merge_start, merge_end = merged_bounds(output,s2_max_ind-t_offset,s2_window,s2_start_frac,s2_end_frac)
    #s2_2 = merge_start > -1
    #print("merge_start:", merge_start)
    #print("merge_end:", merge_end)
    s2_bottom = np.sum(np.array(s2_ch_area)[bottom_channels])
    s2_top = np.sum(np.array(s2_ch_area)[top_channels])
    s2_tba = (s2_top-s2_bottom)/(s2_top+s2_bottom)
    s1_bottom = np.sum(np.array(s1_ch_area)[bottom_channels])
    s1_top = np.sum(np.array(s1_ch_area)[top_channels])
    s1_tba = (s1_top-s1_bottom)/(s1_top+s1_bottom)
    if True and not inn=='q':
    #if s1_found and s2_found:
        fig=pl.figure(1,figsize=(30, 20))
        pl.rc('xtick', labelsize=25)
        pl.rc('ytick', labelsize=25)
        
        ax=pl.subplot2grid((2,2),(0,0))
        pl.title("Top array, event "+str(i))
        pl.grid(b=True,which='major',color='lightgray',linestyle='--')
        ch_labels=['A','B','C','D','E','F','G','H']
        ch_colors=[pl.cm.tab10(ii) for ii in range(n_channels)]
        #ch_colors=[pl.cm.Dark2(ii) for ii in np.linspace(0.2,0.9,n_channels)]
        #ch_colors=['y','cyan','magenta','b','y','cyan','magenta','b']
        if s2_found:
            ax.axvspan(s2_start_pos*tscale, s2_end_pos*tscale, alpha=0.3, color='blue')
        if s1_found:
            ax.axvspan(s1_start_pos*tscale, s1_end_pos*tscale, alpha=0.3, color='green')
        for i_chan in range(n_channels-1):
            if i_chan == (n_channels-1)/2:
                ax=pl.subplot2grid((2,2),(0,1))
                pl.title("Bottom array, event "+str(i))
                pl.grid(b=True,which='major',color='lightgray',linestyle='--')
                
                if s2_found:
                    ax.axvspan(s2_start_pos*tscale, s2_end_pos*tscale, alpha=0.3, color='blue')
                if s1_found:
                    ax.axvspan(s1_start_pos*tscale, s1_end_pos*tscale, alpha=0.3, color='green')
            
            pl.plot(t_matrix[i,:],v_matrix_all_ch[i_chan][i,:],color=ch_colors[i_chan],label=ch_labels[i_chan])
            pl.xlim([trigger_time_us-10,trigger_time_us+10])
            pl.ylim([0, 1000/chA_spe_size])
            pl.xlabel('Time (us)')
            pl.ylabel('Phd/sample')
            pl.legend()
            #triggertime_us = (t[-1]*0.2)
            #pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
            

                    
        ax=pl.subplot2grid((2,2),(1,0),colspan=2)
        pl.plot(t_matrix[i,:],sum_data,'blue')
        pl.xlim([0,event_window])
        pl.ylim([-1, 5])
        pl.xlabel('Time (us)')
        pl.ylabel('Phd/sample')
        pl.title("Sum, event "+ str(i))
        pl.grid(b=True,which='major',color='lightgray',linestyle='--')
        triggertime_us = (t[-1]*0.2)
        #pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        if s2_found:
            ax.axvspan(s2_start_pos*tscale, s2_end_pos*tscale, alpha=0.3, color='blue')
        if s1_found:
            ax.axvspan(s1_start_pos*tscale, s1_end_pos*tscale, alpha=0.3, color='green')
        #if s2_2:
            #ax.axvspan(merge_start*tscale, merge_end*tscale, alpha=0.3, color='yellow')

        #pl.figure()
        #pl.plot(t_matrix[i,:], smooth, 'blue')
        #pl.xlim([0,10])
        #pl.ylim([0,2500])

        #pl.xlabel('Time (us)')
        #pl.ylabel('Millivolts')
        #pl.title('average')
        #pl.show()




        #fig=pl.figure(3,figsize=(20, 20))
            

        #time_array=[]
        #for o in t_matrix[i,:]:
        #    time_array.append(o)
   
          
        #t = np.asarray(time_array)
        #b = (np.diff(np.sign(np.diff(output))) > 0).nonzero()[0] + 1         # list of index locations of maxima
        #c = (np.diff(np.sign(np.diff(output))) < 0).nonzero()[0] + 1         # list of index locations of minima
        #for k in b:
            
        #    if output[k]>1:
        #        print('output[k]:', output[k])
        #        pl.plot(t[k], output[k], marker ='o', color='b')
        #        print("maxima:", k)
        #for a in c:
        #    if output[a]>1:
        #        pl.plot(t[a],output[a],marker='o', color='r')
        #        print("minima:", a)            
        #pl.xlim([0,25])
        #pl.ylim([0,6000/chA_spe_size])
        #pl.xlabel('Time (us)')
        #pl.ylabel('Millivolts')
        #pl.title("SMOOTH")
        #pl.show()


        
        pl.draw()
        pl.show(block=0)
        inn = input("Press enter to continue, q to skip plotting")
        fig.clf()
s2_frac_array = s2_ch_array/s2_area_array.reshape(np.size(s2_area_array),1)
s1_frac_array = s1_ch_array/s1_area_array.reshape(np.size(s1_area_array),1)
#print("s2_frac_Array sum:",np.sum(s2_frac_array, axis=1))
#print("s1_frac_array sum:", np.sum(s1_frac_array, axis=1))
s2_max_frac_array = np.max(s2_frac_array, axis=1)
s1_max_frac_array = np.max(s1_frac_array, axis=1)
s2_top_array = np.sum(s2_ch_array[:,top_channels], axis=1)
s2_bottom_array = np.sum(s2_ch_array[:,bottom_channels], axis=1)
s2_tba = (s2_top_array-s2_bottom_array)/(s2_top_array+s2_bottom_array)
s1_top_array = np.sum(s1_ch_array[:,top_channels], axis=1)
s1_bottom_array = np.sum(s1_ch_array[:,bottom_channels], axis=1)
s1_tba = (s1_top_array-s1_bottom_array)/(s1_top_array+s1_bottom_array)

s2_frac_sum=(np.sum(s2_frac_array, axis=1))[s2_found_array*s1_found_array]
print("s2_frac_sum where not 1:",s2_frac_sum[np.where(s2_frac_sum < 0.99)[0]])
print("s2_found_array:", np.sum(s2_found_array))
print("s1_found_array:", np.sum(s1_found_array))
print("center:", np.sum(center))
print("Events w/ S1+S2 fiducial: ",s2_area_array[s2_found_array*s1_found_array*center].size)
print("Events w/ S1+S2: ",s2_area_array[s2_found_array*s1_found_array].size)
print("S2 Area mean: ", np.mean(s2_area_array[s2_found_array*s1_found_array]))
print("S2 width mean: ", np.mean(s2_width_array[s2_found_array*s1_found_array]))
print("S2 height mean: ", np.mean(s2_height_array[s2_found_array*s1_found_array]))
print("Drift time mean: ", np.mean(t_drift_array[s2_found_array*s1_found_array]))
print("S1 Area mean: ", np.mean(s1_area_array[s2_found_array*s1_found_array]))

pl.figure(figsize=(30,20))
for j in range(0, n_channels-1):
    pl.subplot(4,2,j+1)
    pl.hist2d(t_drift_array[s2_found_array*s1_found_array], s1_frac_array[s2_found_array*s1_found_array][:,j], bins=100, range=[[0,10],[0,1]], norm=mpl.colors.LogNorm())
    pl.minorticks_on()
    pl.xlabel("drift time")
    pl.ylabel("S1_CH_"+ch_labels[j]+"/S1")

pl.figure(figsize=(30,20))
for j in range(0, n_channels-1):
    pl.subplot(4,2,j+1)
    pl.hist2d(t_drift_array[s2_found_array*s1_found_array], s2_frac_array[s2_found_array*s1_found_array][:,j], bins=100,range=[[0,10],[0,1]], norm=mpl.colors.LogNorm())
    pl.minorticks_on()
    pl.xlabel("drift time")
    pl.ylabel("S2_CH_"+ch_labels[j]+"/S2")


# Event selection for summary plots, i.e. analysis cuts
s1_and_s2=s2_found_array*s1_found_array*(s2_width_array>0.2) # cut out events with unrealistically-short s2s
s1_only_like=s2_found_array*np.logical_not(s1_found_array)*(s2_width_array<0.6)*(s2_tba<0)
s2_like=s2_found_array*(s2_width_array>0.6)*(s2_tba>0.)
print("S2-like events (pulse found, long width): ",np.size(s2_area_array[s2_like]))
long_drift = t_drift_array > 3
save_plots=True
cut_name="SmallS2Mode_S1S2_S2TBAcut_LgS2Area"
plot_selection=s1_and_s2*(s2_tba>0.)*(s2_area_array>500.)#s2_found_array#*(s2_tba>0)*(s1_tba<0)*(s2_width_array>1)#s2_like##*(t_drift_array>6)#s2_found_array#
print("events passing plot_selection cuts: ",np.size(s2_area_array[plot_selection]))
    
#pl.figure(figsize=(30, 20))
#for j in range(0, n_channels-1):   
#    pl.subplot(4,2,j+1)
#    pl.hist(s1_ch_array[plot_selection][:,j],bins=100,range=(0,100))
#    #pl.yscale('log')
#    pl.xlabel("S1 area (phd)")
#    pl.title('Ch '+str(j))
#pl.figure(figsize=(30, 20))
#for j in range(0, n_channels-1):   
#    pl.subplot(4,2,j+1)
#    pl.hist(s2_ch_array[plot_selection][:,j],bins=100,range=(0,4000))
#    #pl.yscale('log')
#    pl.xlabel("S2 area (phd)")
#    pl.title('Ch '+str(j))
    
pl.figure(figsize=(30, 20))
for j in range(0, n_channels-1):   
    pl.subplot(4,2,j+1)
    pl.hist(s1_frac_array[plot_selection][:,j],bins=100,range=(0,1))
    #pl.yscale('log')
    pl.xlabel("S1 area fraction")
    pl.title('Ch '+str(j))
if save_plots: pl.savefig(data_dir+"S1_ch_area_frac_"+cut_name+".png")

pl.figure(figsize=(30, 20))
for j in range(0, n_channels-1):   
    pl.subplot(4,2,j+1)
    pl.hist(s2_frac_array[plot_selection][:,j],bins=100,range=(0,1))
    #pl.yscale('log')
    pl.xlabel("S2 area fraction")
    pl.title('Ch '+str(j))
if save_plots: pl.savefig(data_dir+"S2_ch_area_frac_"+cut_name+".png")
    
pl.figure()
pl.hist(s2_max_frac_array[plot_selection],bins=100,range=[0,1])
pl.axvline(x=np.mean(s2_max_frac_array[plot_selection]),ls='--',color='r')
pl.xlabel("S2 area max channel fraction")
if save_plots: pl.savefig(data_dir+"S2_max_ch_frac_"+cut_name+".png")

pl.figure()
pl.hist(s1_max_frac_array[plot_selection],bins=100,range=[0,1])
pl.axvline(x=np.mean(s1_max_frac_array[plot_selection]),ls='--',color='r')
pl.xlabel("S1 area max channel fraction")
if save_plots: pl.savefig(data_dir+"S1_max_ch_frac_"+cut_name+".png")
    
pl.figure()
pl.hist(s2_area_array[plot_selection],bins=100)#,range=(0,1e6/chA_spe_size))
pl.axvline(x=np.mean(s2_area_array[plot_selection]),ls='--',color='r')
pl.xlabel("S2 area (phd)")
if save_plots: pl.savefig(data_dir+"S2_area_"+cut_name+".png")

pl.figure()
pl.hist(s1_area_array[plot_selection],bins=100,range=(0,1000))#(t_drift_array>0.3)
pl.axvline(x=np.mean(s1_area_array[plot_selection]),ls='--',color='r')
pl.xlabel("S1 area (phd)")
if save_plots: pl.savefig(data_dir+"S1_area_"+cut_name+".png")

pl.figure()
pl.hist(s2_tba[plot_selection],bins=100,range=(-1,1))
pl.axvline(x=np.mean(s2_tba[plot_selection]),ls='--',color='r')
pl.xlabel("S2 TBA (t-b)/(t+b)")
if save_plots: pl.savefig(data_dir+"S2_TBA_"+cut_name+".png")

pl.figure()
pl.hist(s1_tba[plot_selection],bins=100,range=(-1,1))
pl.axvline(x=np.mean(s1_tba[plot_selection]),ls='--',color='r')
pl.xlabel("S1 TBA (t-b)/(t+b)")
if save_plots: pl.savefig(data_dir+"S1_TBA_"+cut_name+".png")

pl.figure()
pl.hist(s2_width_array[plot_selection],bins=100)
pl.axvline(x=np.mean(s2_width_array[plot_selection]),ls='--',color='r')
pl.xlabel("S2 width (us)")
if save_plots: pl.savefig(data_dir+"S2_width_"+cut_name+".png")

pl.figure()
pl.hist(s2_height_array[plot_selection],bins=100)
pl.axvline(x=np.mean(s2_height_array[plot_selection]),ls='--',color='r')
pl.xlabel("S2 height (phd/sample)")
if save_plots: pl.savefig(data_dir+"S2_height_"+cut_name+".png")

pl.figure()
pl.hist(s1_height_array[plot_selection],bins=100)
pl.axvline(x=np.mean(s1_height_array[plot_selection]),ls='--',color='r')
pl.xlabel("S1 height (phd/sample)")
if save_plots: pl.savefig(data_dir+"S1_height_"+cut_name+".png")
                                     
pl.figure()
pl.hist(t_drift_array[plot_selection],bins=100)
pl.xlabel("drift time (us)")
if save_plots: pl.savefig(data_dir+"drift_time_"+cut_name+".png")

t_drift_plot=t_drift_array[plot_selection]
s2_width_plot=s2_width_array[plot_selection]
s2_area_plot=s2_area_array[plot_selection]
s2_height_plot=s2_height_array[plot_selection]
s1_area_plot=s1_area_array[plot_selection]
pl.figure()
pl.scatter(t_drift_plot,s2_area_plot)
pl.xlabel("drift time (us)")
pl.ylabel("S2 area (phd)")


drift_bins=np.linspace(0,10,100)
drift_ind=np.digitize(t_drift_plot, bins=drift_bins)
s2_means=np.zeros(np.shape(drift_bins))
s2_std_err=np.ones(np.shape(drift_bins))*0#10000
for i_bin in range(len(drift_bins)):
    found_i_bin = np.where(drift_ind==i_bin) 
    s2_area_i_bin = s2_area_plot[found_i_bin]
    if len(s2_area_i_bin) < 1: continue
    s2_means[i_bin]=np.median(s2_area_i_bin)
    s2_std_err[i_bin]=np.std(s2_area_i_bin)/np.sqrt(len(s2_area_i_bin))
pl.errorbar(drift_bins, s2_means, yerr=s2_std_err, linewidth=3, elinewidth=3, capsize=5, capthick=4, color='red')
pl.ylim(bottom=0)
if save_plots: pl.savefig(data_dir+"S2_area_vs_drift_"+cut_name+".png")

pl.figure()
pl.scatter(s1_area_plot,s2_area_plot)
pl.xlabel("S1 area (phd)")
pl.ylabel("S2 area (phd)")
pl.yscale("log")
if save_plots: pl.savefig(data_dir+"S2_area_vs_S1_area_"+cut_name+".png")

pl.figure()
pl.scatter(t_drift_plot,s2_height_plot)
pl.xlabel("drift time (us)")
pl.ylabel("S2 height (phd/sample)")
s2_height_means=np.zeros(np.shape(drift_bins))
s2_height_std_err=np.ones(np.shape(drift_bins))
for i_bin in range(len(drift_bins)):
    found_i_bin = np.where(drift_ind==i_bin) 
    s2_height_i_bin = s2_height_plot[found_i_bin]
    if len(s2_height_i_bin) < 1: continue
    s2_height_means[i_bin]=np.median(s2_height_i_bin)
    s2_height_std_err[i_bin]=np.std(s2_height_i_bin)/np.sqrt(len(s2_height_i_bin))
pl.errorbar(drift_bins, s2_height_means, yerr=s2_height_std_err, linewidth=3, elinewidth=3, capsize=5, capthick=4, color='red')
pl.ylim(bottom=0)
if save_plots: pl.savefig(data_dir+"S2_height_vs_drift_"+cut_name+".png")

pl.figure()
pl.scatter(s2_area_plot,s2_width_plot)
pl.xlabel("S2 area (phd)")
pl.ylabel("S2 width (us)")
if save_plots: pl.savefig(data_dir+"S2_width_vs_area_"+cut_name+".png")

pl.show()
