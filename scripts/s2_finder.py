import numpy as np
#import pylab as pl
import matplotlib.pyplot as pl
import matplotlib as mpl

def pulse_finder_area(data,t_min_search,t_max_search,window,thresh):
# Assumes data is already baseline-subtracted
    max_area=-1
    max_ind=-1
    for i_start in range(t_min_search,t_max_search):
        area=np.sum(data[i_start:i_start+window])
        if area>max_area and area>thresh:
            max_area=area
            max_ind=i_start
    return (max_ind, max_area)

def pulse_bounds(data,t_min,window,start_frac,end_frac):
# Assumes data is already baseline-subtracted
    start_pos=-1
    end_pos=-1
    peak_val=np.max(data[t_min:t_min+window])
    peak_pos=np.argmax(data[t_min:t_min+window])
    #start_frac: pulse starts at this fraction of peak height above baseline
    for i_start in range(t_min,t_min+window):
        if data[i_start]>peak_val*start_frac:
            start_pos=i_start
            break
    #end_frac: pulse ends at this fraction of peak height above baseline
    for i_start in range(t_min+window,t_min,-1):
        if data[i_start]>peak_val*end_frac:
            end_pos=i_start
            break
    return (start_pos, end_pos)

# set plotting style
mpl.rcParams['font.size']=28
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
# mpl.rcParams['figure.figsize']=[16.0,12.0]

# Read data
channel_0=np.fromfile("A-K40-after-rec.dat", dtype="int16")
channel_1=np.fromfile("B-K40-after-rec.dat", dtype="int16")
channel_2=np.fromfile("C-K40-after-rec.dat", dtype="int16")
channel_3=np.fromfile("D-K40-after-rec.dat", dtype="int16")

#channel_0=np.fromfile("A-thorium.dat", dtype="int16")
#channel_1=np.fromfile("B-thorium.dat", dtype="int16")
#channel_2=np.fromfile("C-thorium.dat", dtype="int16")
#channel_3=np.fromfile("D-thorium.dat", dtype="int16")

#channel_0=np.fromfile("A-K40.dat", dtype="int16")
#channel_1=np.fromfile("B-K40.dat", dtype="int16")
#channel_2=np.fromfile("C-K40.dat", dtype="int16")
#channel_3=np.fromfile("D-K40.dat", dtype="int16")
vscale=(2000.0/16384.0)
wsize=50000
V=vscale*channel_0
V_1=vscale*channel_1
V_2=vscale*channel_2
V_3=vscale*channel_3
n_channels=5 # including sum
v_matrix = V.reshape(int(V.size/wsize),wsize)
v1_matrix = V_1.reshape(int(V.size/wsize),wsize)
v2_matrix = V_2.reshape(int(V.size/wsize),wsize)
v3_matrix = V_3.reshape(int(V.size/wsize),wsize)
v4_matrix = v_matrix+v1_matrix+v2_matrix+v3_matrix
v_matrix_all_ch=[v_matrix,v1_matrix,v2_matrix,v3_matrix,v4_matrix]
x=np.arange(0, wsize, 1)
tscale=(8.0/4096.0)
t=tscale*x
t_matrix=np.repeat(t[np.newaxis,:], V.size/wsize, 0)
max_ind_array=np.zeros((v_matrix.shape[0],n_channels) )
max_val_array=np.zeros((v_matrix.shape[0],n_channels) )
integral_array=np.zeros((v_matrix.shape[0],n_channels) )
s2_integral_array=np.zeros((v_matrix.shape[0],n_channels) )
s2_area_array=[]
s1_area_array=[]
s2_width_array=[]
t_drift_array=[]

print("Total events: ",v_matrix.shape[0])
for i in range (0, v_matrix.shape[0]):
    if i%100==0: print("Event #",i)
    # for each channel
    for j in range(0, n_channels):
        #i = input("Window number between 1 and " + str((V.size/wsize)) + ": ")
        
        baseline=np.mean(v_matrix_all_ch[j][i,:500]) #avg ~1 us
        #print("baseline: ",baseline)
        
        win_min=int(18./tscale)
        win_max=int(21./tscale)
        integral=np.sum(v_matrix_all_ch[j][i,win_min:win_max]-baseline)
        integral_array[i,j]=integral
        # Threshold integral for trigger at 770: 1.9e5
		
		
        #print("Below is max index")
        max_ind=tscale*np.argmax(v_matrix_all_ch[j][i,:])
        max_ind_array[i,j]=max_ind
        #print(max_ind) 
        #print("Below is max value")
        max_val=np.max(v_matrix_all_ch[j][i,:])
        max_val_array[i,j]=max_val
        #print(max_val)
        
    # Look for events with S1 and S2 from summed channel
    s1_window = int(0.5/tscale)
    s2_window = int(1.5/tscale)
    s1_thresh = 600
    s2_thresh = 1e4 # same threshold for now
    s1_max=s1_thresh
    s1_max_ind=-1
    s1_area=-1
    s1_start_pos=-1
    s1_end_pos=-1
    s2_max=s2_thresh
    s2_max_ind=-1
    s2_area=-1
    s2_start_pos=-1
    s2_end_pos=-1
    s2_width=-1
    t_drift=-1

    sum_baseline=np.mean(v4_matrix[i,int(50./tscale):int(60./tscale)]) #avg ~10 us, later in acquisition since there may be odd noise earlier?
    sum_data=v4_matrix[i,:]-sum_baseline
    
    # Do a moving average (sum) of the waveform with different time windows for s1, s2
	# Look for where this value is maximized
	# Look for the s2 using a moving average (sum) of the waveform over a wide window
    t_min_search=int(16./tscale)
    t_max_search=int(25./tscale)
    t_offset=int(0.1/tscale)
    s2_max_ind, s2_max=pulse_finder_area(sum_data,t_min_search,t_max_search,s2_window,s2_thresh)
    if s2_max_ind>-1: # Found a pulse (maybe an s2)
        # print("s2 window time: ",s2_max_ind*tscale,"s2 area max: ",s2_max)
        start_frac=0.05 # pulse starts at this fraction of peak height above baseline
        end_frac=0.05 # pulse starts at this fraction of peak height above baseline
        s2_start_pos, s2_end_pos = pulse_bounds(sum_data,s2_max_ind-t_offset,s2_window,start_frac,end_frac)
        s2_area=np.sum(sum_data[s2_start_pos:s2_end_pos])
        s2_width=(s2_end_pos-s2_start_pos)*tscale
        s2_area_array.append(s2_area)
        s2_width_array.append(s2_width)
        # print("s2 start: ",s2_start_pos*tscale," s2 end: ",s2_end_pos*tscale)
		
        # Now look for a prior s1
        s1_max_ind, s1_max=pulse_finder_area(sum_data,t_min_search,s2_start_pos-s1_window,s1_window,s1_thresh)
        if s1_max_ind>-1:
            # print("s1 window time: ",s1_max_ind*tscale,"s1 area max: ",s1_max)
            s1_start_pos, s1_end_pos = pulse_bounds(sum_data,s1_max_ind-t_offset,s1_window,0.1,0.1)
            # print("s1 start: ",s1_start_pos*tscale," s1 end: ",s1_end_pos*tscale)
            t_drift=(s2_start_pos-s1_start_pos)*tscale
            s1_area=np.sum(sum_data[s1_start_pos:s1_end_pos])
            s1_area_array.append(s1_area)
            t_drift_array.append(t_drift)
		
    # once
    if -1e12>integral:
        pl.figure(1,figsize=(20, 20))
        pl.clf()
        pl.rc('xtick', labelsize=25)
        pl.rc('ytick', labelsize=25)
        
        pl.subplot(2,3,1)
        pl.plot(t_matrix[i,:],v_matrix[i,:],'y')
        pl.xlim([17, 35])
        pl.ylim([0, 1500])
        pl.xlabel('Time (us)')
        pl.ylabel('Millivolts')
        pl.title("A,"+ str(i))
        triggertime_us = (t[-1]*0.2)
        pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        
        pl.subplot(2,3,2)
        pl.plot(t_matrix[i,:],v1_matrix[i,:],'cyan')
        pl.xlim([17, 35])
        pl.ylim([0, 1500])
        pl.xlabel('Time (us)')
        pl.ylabel('Millivolts')
        pl.title("B,"+ str(i))
        triggertime_us = (t[-1]*0.2)
        pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        
        pl.subplot(2,3,3)
        pl.plot(t_matrix[i,:],v2_matrix[i,:],'magenta')
        pl.xlim([17, 35])
        pl.ylim([0, 1500])
        pl.xlabel('Time (us)')
        pl.ylabel('Millivolts')
        pl.title("C,"+ str(i))
        triggertime_us = (t[-1]*0.2)
        pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        
        pl.subplot(2,3,4)
        pl.plot(t_matrix[i,:],v3_matrix[i,:],'blue')
        pl.xlim([17, 35])
        pl.ylim([0, 1500])
        pl.xlabel('Time (us)')
        pl.ylabel('Millivolts')
        pl.title("D,"+ str(i))
        triggertime_us = (t[-1]*0.2)
        pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        
        ax=pl.subplot(2,3,5)
        pl.plot(t_matrix[i,:],v4_matrix[i,:],'blue')
        pl.xlim([17, 35])
        pl.ylim([0, 2500])
        pl.xlabel('Time (us)')
        pl.ylabel('Millivolts')
        pl.title("Sum,"+ str(i))
        triggertime_us = (t[-1]*0.2)
        pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        if s2_max_ind > -1:
            ax.axvspan(s2_start_pos*tscale, s2_end_pos*tscale, alpha=0.5, color='blue')
        if s1_max_ind > -1:
            ax.axvspan(s1_start_pos*tscale, s1_end_pos*tscale, alpha=0.5, color='green')
        
        pl.show()
        inn = input("Press enter to continue")

pl.figure(figsize=(20, 20))
pl.clf()     
for j in range(0, n_channels):   
    pl.subplot(3,2,j+1)
    pl.hist(max_ind_array[:,j],bins=100)
    pl.yscale('log')
    pl.xlabel("Time of max value")
pl.figure(figsize=(20, 20))
pl.clf()
for j in range(0, n_channels):    
    pl.subplot(3,2,j+1)  
    pl.hist(max_val_array[:,j],bins=100)
    pl.xlabel("Max value")
    pl.yscale('log')
pl.figure(figsize=(20, 20))
pl.clf()
for j in range(0, n_channels):    
    pl.subplot(3,2,j+1)
    pl.hist(integral_array[:,j],bins=100,range=(-100,1500))
    pl.xlabel("Pulse integral")
    #pl.yscale('log')
    pl.title('Ch '+str(j))
pl.figure()
pl.clf()
pl.hist(s2_area_array,bins=100)
pl.xlabel("S2 area")
pl.figure()
pl.clf()
pl.hist(s1_area_array,bins=100)
pl.xlabel("S1 area")
pl.figure()
pl.clf()
pl.hist(s2_width_array,bins=100)
pl.xlabel("S2 width (us)")
pl.figure()
pl.clf()
pl.hist(t_drift_array,bins=100)
pl.xlabel("drift time (us)")

pl.show()
