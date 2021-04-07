import numpy as np
#import pylab as pl
import matplotlib.pyplot as pl
import matplotlib as mpl
from scipy.stats import norm
from scipy.optimize import curve_fit

def pulse_finder_area(data,t_min_search,t_max_search,window):
# Assumes data is already baseline-subtracted
    weights = np.repeat(1.0, window)#/window #to do avg instead of sum
    data_conv = np.convolve(data, weights, 'same')
        # Search only w/in search range, offset so that max_ind is the start of the window rather than the center
    max_ind=np.argmax(data_conv[int(t_min_search+window/2):int(t_max_search+window/2)])+int(t_min_search)
    return (max_ind, data_conv[max_ind+int(window/2)])

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
    if start_pos == -1:
        print("tmin: ",t_min,"window: ",window,"peak_val: ",peak_val,"peak_pos: ",peak_pos)
    return (start_pos, end_pos)

# set plotting style
mpl.rcParams['font.size']=28
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[16.0,12.0]

data_dir = "/home/xaber/caen/wavedump-3.8.2/data/040721/dark_count_all_channel_Cu_rod_110.0/"
channel_name = "0"
channel_0=np.fromfile(data_dir+"wave"+channel_name+".dat", dtype="int16")

#channel_0=np.fromfile("../../072320/chB_darkcount_calib_50V_-6.1mV/wave0.dat", dtype="int16")

#channel_0=np.fromfile("../../082919/chB_49_5V_18mV.dat", dtype="int16")
#channel_0=np.fromfile("../../Desktop/crystallize_data/110819/chH_51.5V_12mV.dat", dtype="int16")



vscale=(2000.0/16384.0)/4. # smaller digitizer range
wsize=1030 # Number of samples per waveform
V=vscale*channel_0
print("record length: ", len(V))
# Ensure we have an integer number of events
V=V[:int(len(V)/wsize)*wsize]
n_channels=2 # including sum
v_matrix = V.reshape(int(V.size/wsize),wsize)
v4_matrix = v_matrix
v_matrix_all_ch=[v_matrix,v4_matrix]
x=np.arange(0, wsize, 1)
tscale=(8.0/4096.0) # Conversion from samples to us (do not change)
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
    s1_window = int(0.2/tscale)
    s2_window = int(0.15/tscale)
    s1_thresh = 400
    s1_range_thresh = 10
    s2_thresh = 0.
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

    sum_baseline=np.mean(v4_matrix[i,int(0.2/tscale):int(0.5/tscale)]) #avg 0.3 us, later in acquisition since there may be odd noise earlier?
    sum_data=v4_matrix[i,:]-sum_baseline
    
    # Do a moving average (sum) of the waveform with different time windows for s1, s2
	# Look for where this value is maximized
	# Look for the s2 using a moving average (sum) of the waveform over a wide window
    t_min_search=int(0.5/tscale)
    t_max_search=int(1.5/tscale)
    t_offset=int(0.00/tscale)
    #s2_max_ind, s2_max=pulse_finder_area(sum_data,t_min_search,t_max_search,s2_window)
    
    # Find max of waveform, integrate window w/in range nearby
    t_before = int(0.05/tscale)
    t_after = int(0.15/tscale)
    t_baseline = int(0.15/tscale)
    s2_window = t_before+t_after
    s2_max_ind = np.argmax(sum_data)
    s2_max = sum_data[s2_max_ind]
    #s2_start = s2_max_ind-t_before
    #s2_end = s2_max_ind+t_after
    #if s2_start<0 or s2_end>wsize: continue
    s2_start = np.maximum(0, s2_max_ind-t_before)
    s2_end = np.minimum(wsize, s2_max_ind+t_after)
    baseline_start = np.maximum(0, s2_start-t_baseline)
    baseline_correct = np.mean(sum_data[baseline_start:s2_start])
    s2_area = np.sum(sum_data[s2_start:s2_end]-baseline_correct)
    s2_max_ind = s2_start
    
    #print(s2_max)
  
	
    s2_found=True#s2_max>s2_thresh
    #s2_area=s2_max
	
    ratio=-1

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
    if s2_found and not inn=='q':
    #if s1_found and s2_found:
        #print(ratio)
        fig=pl.figure(1,figsize=(20, 10))
        pl.rc('xtick', labelsize=25)
        pl.rc('ytick', labelsize=25)
        
        ax=pl.subplot2grid((1,1),(0,0),colspan=2)
        pl.plot(t_matrix[i,:],sum_data-baseline_correct,'blue')
        pl.grid(b=True,which='major',color='lightgray',linestyle='--',lw=1)
        #pl.xlim([1.5, 2.5])
        pl.xlim([0, 4])
        pl.ylim([-10, 20])
        pl.xlabel('Time (us)')
        pl.ylabel('Millivolts')
        pl.title("Sum,"+ str(i))
        triggertime_us = (t[-1]*0.1)
        
        #ax.axvspan(t_min_search*tscale, t_max_search*tscale, alpha=0.5, color='red')
        pl.plot(np.array([1,1])*triggertime_us,np.array([0,16384]),'k--')
        if s2_found:
            #ax.axvspan(s2_start_pos*tscale, s2_end_pos*tscale, alpha=0.5, color='blue')
            ax.axvspan(s2_max_ind*tscale, (s2_max_ind+s2_window)*tscale, alpha=0.3, color='green')
            ax.text((s2_max_ind+s2_window)*tscale, 0.9 * ax.get_ylim()[1], '{:.1f} mv*samples'.format(s2_area),
                    fontsize=24)
        if s1_found:
            ax.axvspan(s1_start_pos*tscale, s1_end_pos*tscale, alpha=0.5, color='green')
            
        pl.draw()
        pl.show(block=0)
        print("s2 area: ",s2_area)
        inn = input("Press enter to continue, q to skip plotting")
        fig.clf()

singleCutoff = 1000.
singleMean = np.mean(s2_area_array[s2_found_array])
print("Events w/ S2: ",s2_area_array[s2_found_array].size)
print("S2 Area mean: ", singleMean)
print("S2 width mean: ", np.mean(s2_width_array[s2_found_array]))
print("S2 height mean: ", np.mean(s2_height_array[s2_found_array]))



#pl.figure()
#pl.xlim([0,3])
#pl.hist(ratio_array, bins=1000)
#pl.xlabel("positive_area/negative_area")
#pl.show()
def gaussian(x, mean, std,a):
    return a*np.exp(-((x-mean)/std)**2)
bins = np.linspace(-20, 100, 150)
s2_start = 8
s2_end = 18
s2_start_bin = np.digitize(s2_start, bins)
s2_end_bin = np.digitize(s2_end, bins)
pl.figure()
h,b,_=pl.hist(s2_area_array[s2_found_array],bins=bins)
bin_centers = b[:-1] + np.diff(b)/2
popt, _ = curve_fit(gaussian,bin_centers[s2_start_bin:s2_end_bin],h[s2_start_bin:s2_end_bin], p0=[270,80,450])
#print("bin_centers:", bin_centers)
#print("h:", h)
print("popt:",popt)
pl.plot(bin_centers, gaussian(bin_centers, *popt), label ='fit')

pl.grid(b=True,which='major',color='lightgray',linestyle='--')
#pl.yscale("log")
#pl.axvline(x=popt[0],ls='--',color='r')
pl.text(0.3,0.8,'Gaussian Mean: {0:g}'.format(popt[0]),transform=pl.gca().transAxes)
#pl.text(0.3,0.7,'std: {0:g}'.format(popt[1]),transform=pl.gca().transAxes)
#pl.text(0.3,0.6,'Amplitude: {0:g}'.format(popt[2]),transform=pl.gca().transAxes)
pl.xlabel("Pulse Area")


#pl.figure()
#pl.hist(s2_width_array[s2_found_array],bins=100)
#pl.axvline(x=np.mean(s2_width_array[s2_found_array]),ls='--',color='r')
#pl.xlabel("S2 width (us)")

#pl.figure()
#pl.hist(s2_height_array[s2_found_array],bins=100)
#pl.axvline(x=np.mean(s2_height_array[s2_found_array]),ls='--',color='r')
#pl.xlabel("S2 height (mV)")

#pl.figure()
#pl.scatter(s2_area_array[s2_found_array],s2_width_array[s2_found_array])
#pl.xlabel("S2 area")
#pl.ylabel("S2 width (us)")
pl.savefig(data_dir+"dark_counts_pulse_area_hist_channel_"+channel_name+".png")
pl.show()
