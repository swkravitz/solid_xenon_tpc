import numpy as np
#import pylab as pl
import matplotlib.pyplot as pl
import matplotlib as mpl

def pulse_finder_area(data,t_min_search,t_max_search,window):
# Assumes data is already baseline-subtracted
    max_area=-1
    max_ind=-1
    for i_start in range(t_min_search,t_max_search):
        area=np.sum(data[i_start:i_start+window])
        if area>max_area:
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
        if data[i_start]>max(peak_val*start_frac,3):
            start_pos=i_start
            break
    #end_frac: pulse ends at this fraction of peak height above baseline
    for i_start in range(t_min+window,t_min,-1):
        if data[i_start]>max(peak_val*end_frac,3):
            end_pos=i_start
            break
    return (start_pos, end_pos)

# set plotting style
mpl.rcParams['font.size']=28
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[16.0,12.0]

#channel_0=np.fromfile("wave0.dat", dtype="int16")

#channel_0=np.fromfile("../../082919/chB_49_5V_18mV.dat", dtype="int16")
channel_0=np.fromfile("../../Desktop/crystallize_data/082919_CHB/CHB_49_5V_18mV.dat", dtype="int16")



vscale=(2000.0/16384.0)
wsize=1000
V=vscale*channel_0
# Ensure we have an integer number of events
V=V[:int(len(V)/wsize)*wsize]
n_channels=2 # including sum
v_matrix = V.reshape(int(V.size/wsize),wsize)
v4_matrix = v_matrix
v_matrix_all_ch=[v_matrix,v4_matrix]
x=np.arange(0, wsize, 1)
tscale=(8.0/4096.0)
t=tscale*x
t_matrix=np.repeat(t[np.newaxis,:], V.size/wsize, 0)
# One entry per channel
max_ind_array=np.zeros((v_matrix.shape[0],n_channels) )
max_val_array=np.zeros((v_matrix.shape[0],n_channels) )
integral_array=np.zeros((v_matrix.shape[0],n_channels) )
s2_integral_array=np.zeros((v_matrix.shape[0],n_channels) )
# One entry per event
s2_area_array=np.zeros(v_matrix.shape[0])
s1_area_array=np.zeros(v_matrix.shape[0])
s2_width_array=np.zeros(v_matrix.shape[0])
s2_height_array=np.zeros(v_matrix.shape[0])
t_drift_array=np.zeros(v_matrix.shape[0])
s2_found_array=np.zeros(v_matrix.shape[0],dtype='bool')
s1_found_array=np.zeros(v_matrix.shape[0],dtype='bool')
ratio_array=np.zeros(v_matrix.shape[0])
# s2_area_array=[]
# s1_area_array=[]
# s2_width_array=[]
# t_drift_array=[]
# s2_found_array=[]
# s1_found_array=[]
inn =''
 
print("Total events: ",v_matrix.shape[0])
for i in range(0, int(v_matrix.shape[0])):
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
    s2_window = int(0.5/tscale)
    s1_thresh = 400
    s1_range_thresh = 10
    s2_thresh = 100.
    s1_max=s1_thresh
    s1_max_ind=-1
    s1_area=-1
    s1_height_range=-1
    s1_start_pos=-1
    s1_end_pos=-1
    s2_max=s2_thresh
    s2_max_ind=-1
    s2_area=-1
    s2_start_pos=-1
    s2_end_pos=-1
    s2_width=-1
    s2_height=-1
    t_drift=-1
    s1_found=False
    s2_found=False

    sum_baseline=np.mean(v4_matrix[i,int(1./tscale):int(1.75/tscale)]) #avg 0.5 us, later in acquisition since there may be odd noise earlier?
    sum_data=v4_matrix[i,:]-sum_baseline
    
    # Do a moving average (sum) of the waveform with different time windows for s1, s2
	# Look for where this value is maximized
	# Look for the s2 using a moving average (sum) of the waveform over a wide window
    t_min_search=int(0./tscale)
    t_max_search=int(1.2/tscale)
    t_offset=int(0.00/tscale)
    s2_max_ind, s2_max=pulse_finder_area(sum_data,t_min_search,t_max_search,s2_window)
    #print(s2_max)
    pos=0
    neg=0
    
    for i in range(s2_max_ind, s2_max_ind+s2_window):
        if sum_data[i]>0:
            pos+=sum_data[i]
        elif sum_data[i]<0:
            neg+=abs(sum_data[i])
    ratio = pos/neg
	
    s2_found=s2_max>s2_thresh
    if s2_found: # Found a pulse (maybe an s2)
        # print("s2 window time: ",s2_max_ind*tscale,"s2 area max: ",s2_max)
        start_frac=0.1 # pulse starts at this fraction of peak height above baseline
        end_frac=0.2 # pulse starts at this fraction of peak height above baseline
        s2_start_pos, s2_end_pos = pulse_bounds(sum_data,s2_max_ind-t_offset,s2_window,start_frac,end_frac)
        s2_area=np.sum(sum_data[s2_start_pos:s2_end_pos])
        s2_width=(s2_end_pos-s2_start_pos)*tscale
        s2_height=np.max(sum_data[s2_start_pos:s2_end_pos])
        #s2_area_array.append(s2_area)
        #s2_width_array.append(s2_width)
        #s2_found_array.append(s2_found)
        #print("s2 start: ",s2_start_pos*tscale," s2 end: ",s2_end_pos*tscale)
		
        # Now look for a prior s1                                                   
        s1_max_ind, s1_max=pulse_finder_area(sum_data,t_min_search,s2_start_pos-s1_window-t_offset,s1_window)
       # print("s1 area: ",s1_max)
        if s1_max>s1_thresh:
            # print("s1 window time: ",s1_max_ind*tscale,"s1 area max: ",s1_max)    
            s1_start_pos, s1_end_pos = pulse_bounds(sum_data,s1_max_ind-t_offset,s1_window,0.1,0.1)
            if s1_start_pos > -1 and s1_end_pos > s1_start_pos:
               # print(s1_start_pos)
               # print(s1_end_pos)
                # Check that we didn't accidentally find noise (related to poor baseline subtraction)
                s1_height_range=np.max(sum_data[s1_start_pos:s1_end_pos])-np.min(sum_data[s1_start_pos:s1_end_pos]) 
                s1_found = s1_height_range>s1_range_thresh
                # print("s1 start: ",s1_start_pos*tscale," s1 end: ",s1_end_pos*tscale)
                if 0.60<t_drift<0.70:
                    print("s1_max_ind: ",s1_max_ind*tscale," s1_start_pos: ",s1_start_pos*tscale," tdrift: ",t_drift)
                    print("s1 range: ",s1_height_range)
                    print("baseline: ",sum_baseline)
                if not s1_found:
                    pass
                   # print("under range, s1 range: ",s1_height_range)
                else:    
                    t_drift=(s2_start_pos-s1_start_pos)*tscale
                    s1_area=np.sum(sum_data[s1_start_pos:s1_end_pos])
         	        #s1_found_array.append(s1_found)       
	            #s1_area_array.append(s1_area)
	            #t_drift_array.append(t_drift)
	


    s2_area_array[i]=s2_area
    s2_width_array[i]=s2_width
    s2_height_array[i]=s2_height
    s1_area_array[i]=s1_area
    t_drift_array[i]=t_drift
    s1_found_array[i]=s1_found
    s2_found_array[i]=s2_found
    ratio_array[i]=ratio	 
		
    # once per event
    #if s1_max_ind>-1 and not s1_height_range>s1_range_thresh:
    #if 1.5<t_drift:
    #if 1.08<t_drift<1.12:
    if s2_found and not inn=='q' and (ratio > 1):
    #if s1_found and s2_found:
        print(ratio)
        fig=pl.figure(1,figsize=(10, 10))
        pl.rc('xtick', labelsize=25)
        pl.rc('ytick', labelsize=25)
        
        ax=pl.subplot2grid((1,1),(0,0),colspan=2)
        pl.plot(t_matrix[i,:],sum_data,'blue')
        #pl.xlim([0, 2])
        pl.ylim([-20, 100])
        pl.xlabel('Time (us)')
        pl.ylabel('Millivolts')
        pl.title("Sum,"+ str(i))
        triggertime_us = (t[-1]*0.5)
        
        #pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        if s2_found:
            ax.axvspan(s2_start_pos*tscale, s2_end_pos*tscale, alpha=0.5, color='blue')
        if s1_found:
            ax.axvspan(s1_start_pos*tscale, s1_end_pos*tscale, alpha=0.5, color='green')
            
        pl.draw()
        pl.show(block=0)
        inn = input("Press enter to continue, q to skip plotting")
        fig.clf()

singleCutoff = 1000.
singleMean = np.mean(s2_area_array[s2_found_array*(s2_area_array<singleCutoff)])
print("Events w/ S2: ",s2_area_array[s2_found_array].size)
print("S2 Area mean: ", singleMean)
print("S2 width mean: ", np.mean(s2_width_array[s2_found_array]))
print("S2 height mean: ", np.mean(s2_height_array[s2_found_array]))



pl.figure()
pl.xlim([0,3])
pl.hist(ratio_array, bins=1000)
pl.xlabel("positive_area/negative_area")
pl.show()
    
pl.figure()
pl.hist(s2_area_array[s2_found_array*(ratio_array>2)],bins=100)
pl.axvline(x=singleMean,ls='--',color='r')
pl.text(0.3,0.8,'Mean single count area: {0:g}'.format(singleMean),transform=pl.gca().transAxes)
pl.xlabel("S2 area")


pl.figure()
pl.hist(s2_width_array[s2_found_array],bins=100)
pl.axvline(x=np.mean(s2_width_array[s2_found_array]),ls='--',color='r')
pl.xlabel("S2 width (us)")

pl.figure()
pl.hist(s2_height_array[s2_found_array],bins=100)
pl.axvline(x=np.mean(s2_height_array[s2_found_array]),ls='--',color='r')
pl.xlabel("S2 height (mV)")

pl.figure()
pl.scatter(s2_area_array[s2_found_array],s2_width_array[s2_found_array])
pl.xlabel("S2 area")
pl.ylabel("S2 width (us)")

pl.show()
