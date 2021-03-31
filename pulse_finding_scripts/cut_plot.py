import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import time

import PulseFinderScipy as pf
import PulseQuantities as pq
import PulseClassification as pc

#data_dir = "D:/.shortcut-targets-by-id/11qeqHWCbcKfFYFQgvytKem8rulQCTpj8/crystalize/data/data-202102/022421/Po_6.8g_7.0c_3mV_1.75bar_circ_20min/"
data_dir = "/home/xaber/caen/wavedump-3.8.2/data/032921/Co_ICV_bot_2.8g_3.0c_0.73bar_circ_3min_1030/"
#data_dir = "G:/My Drive/crystalize/data/data-202103/032221/Co_ICVbot_Po_2.8g_3.00c_1.1bar_circ_5min_1720/"

# set plotting style
mpl.rcParams['font.size']=10
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[8.0,6.0]

# use for coloring pulses
pulse_class_colors = np.array(['blue', 'green', 'red', 'magenta', 'darkorange'])
pulse_class_labels = np.array(['Other', 'S1-like LXe', 'S1-like gas', 'S2-like', 'Merged S1/S2'])
pc_legend_handles=[]
for class_ind in range(len(pulse_class_labels)):
    pc_legend_handles.append(mpl.patches.Patch(color=pulse_class_colors[class_ind], label=str(class_ind)+": "+pulse_class_labels[class_ind]))


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

#read RQ
listrq = np.load(data_dir+'rq.npz')

n_events = listrq['n_events'][()]
n_pulses = listrq['n_pulses']
n_s1 = listrq['n_s1']
n_s2 = listrq['n_s2']
s1_before_s2 = listrq['s1_before_s2']
p_area = listrq['p_area']
p_class = listrq['p_class']
drift_Time = listrq['drift_Time']
drift_Time_AS = listrq['drift_Time_AS']
p_max_height = listrq['p_max_height']
p_min_height = listrq['p_min_height']
p_width = listrq['p_width']
p_afs_2l = listrq['p_afs_2l']
p_afs_50 = listrq['p_afs_50']
p_area_ch = listrq['p_area_ch']
p_area_ch_frac = listrq['p_area_ch_frac']
p_area_top = listrq['p_area_top']
p_area_bottom = listrq['p_area_bottom']
p_tba = listrq['p_tba']
p_start = listrq['p_start']
p_end = listrq['p_end']
sum_s1_area = listrq['sum_s1_area']
sum_s2_area = listrq['sum_s2_area']

listrq.close()
#end of RQ read

n_golden = int(np.sum(drift_Time>0))
print("number of golden events found = {0:d} ({1:g}%)".format(n_golden,n_golden*100./n_events))

p_t_rise = tscale*(p_afs_50-p_afs_2l)

# Define some standard cuts for plotting
cut_dict = {}
cut_dict['ValidPulse'] = p_area > 0
cut_dict['PulseClass0'] = p_class == 0
cut_dict['S1'] = (p_class == 1) + (p_class == 2)
cut_dict['S2'] = (p_class == 3) + (p_class == 4)
cut_dict['Co_peak'] = (p_area>30)*(p_area<60)
SS_cut = drift_Time > 0

# Pick which cut from cut_dict to apply here and whether to save plots
save_pulse_plots=True # One entry per pulse
save_S1S2_plots=True # One entry per S1 (S2) pulse
save_event_plots=True # One entry per event
save_2S2_plots=True # One entry per event w/ 2 S2s
pulse_cut_name = 'ValidPulse'#'Co_peak'
pulse_cut = cut_dict[pulse_cut_name]
print("number of pulses found passing cut "+pulse_cut_name+" = {0:d} ({1:g}% of pulses found)".format(np.sum(pulse_cut),np.sum(pulse_cut)*100./np.sum(n_pulses)))
#pulse_cut_name = 'ValidPulse_SS_Evt'
#pulse_cut = pulse_cut*SS_cut[:,np.newaxis] # add second dimension to allow broadcasting

cleanArea = p_area[pulse_cut].flatten()
cleanMax = p_max_height[pulse_cut].flatten()
cleanMin = p_min_height[pulse_cut].flatten()
cleanWidth = p_width[pulse_cut].flatten()
cleanPulseClass = p_class[pulse_cut].flatten()

cleanAFS2l = p_afs_2l[pulse_cut].flatten()
cleanAFS50 = p_afs_50[pulse_cut].flatten()
cleanRiseTime = p_t_rise[pulse_cut].flatten()

cleanAreaCh = p_area_ch[pulse_cut] # pulse_cut gets broadcast to the right shape
cleanAreaChFrac = p_area_ch_frac[pulse_cut]
cleanAreaTop = p_area_top[pulse_cut].flatten()
cleanAreaBottom = p_area_bottom[pulse_cut].flatten()
cleanTBA = p_tba[pulse_cut].flatten()
# Note: TBA can be <-1 or >+1 if one of top or bottom areas is <0 (can still be a valid pulse since total area >0)

s1_cut = pulse_cut*cut_dict['S1']
cleanS1Area = p_area[s1_cut].flatten()
cleanS1RiseTime = p_t_rise[s1_cut].flatten()
cleanS1AreaChFrac = p_area_ch_frac[s1_cut]
cleanS1TBA = p_tba[s1_cut].flatten()
print("number of S1 pulses found = {0:d} ({1:g}% of pulses found)".format(np.sum(s1_cut),np.sum(s1_cut)*100./np.sum(n_pulses)))

s2_cut = pulse_cut*cut_dict['S2']
cleanS2Area = p_area[s2_cut].flatten()
cleanS2RiseTime = p_t_rise[s2_cut].flatten()
cleanS2AreaChFrac = p_area_ch_frac[s2_cut]
cleanS2TBA = p_tba[s2_cut].flatten()
print("number of S2 pulses found = {0:d} ({1:g}% of pulses found)".format(np.sum(s2_cut),np.sum(s2_cut)*100./np.sum(n_pulses)))

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
event_cut_dict["All_Scatter"] = drift_Time_AS > 0
event_cut_dict["MS"] = (n_s1 == 1)*(n_s2 > 1)*s1_before_s2
event_cut_dict["Po"] = (drift_Time>0)*np.any((p_tba<-0.0)*(p_tba>-0.7)*(p_area>1000)*(p_area<4000), axis=1)#np.any((p_tba<-0.85)*(p_tba>-0.91)*(p_area>1500)*(p_area<2700), axis=1) # true if any pulse in event matches these criteria
event_cut_dict["lg_S1"] = (drift_Time>0)*np.any((p_area>1000.)*cut_dict["S1"], axis=1) # true if any S1 has area>1000
event_cut_dict["2S2"] = (n_s2 == 2)

event_cut_name = "SS"#"Po"#"lg_S1"
event_cut = event_cut_dict[event_cut_name] 
cleanSumS1 = sum_s1_area[event_cut]
cleanSumS2 = sum_s2_area[event_cut]
cleanDT = drift_Time[event_cut]
cleanDT_AS = drift_Time_AS[drift_Time_AS>0]
print("number of events found passing cut "+event_cut_name+" = {0:d} ({1:g}%)".format(np.sum(event_cut),np.sum(event_cut)*100./n_events))

# =============================================================
# =============================================================
# now make plots of interesting pulse quantities

# Turns data into (x,y) points of histogram to plot 
def histToPlot(data, bins):
    [histData,bins] = np.histogram(data, bins=bins)
    binCenters = np.array([0.5 * (bins[j] + bins[j+1]) for j in range(len(bins)-1)])
    return binCenters, histData

# For creating basic histograms
def basicHist(data, bins=100, save=False, name="", mean=False, show=False, hRange=[], xlim=[], ylim=[], xlabel="", ylabel="", logx=False, logy=False, area_max_plot=-99999999,legHand=[]):
    pl.figure()
    if len(hRange) > 1: 
        cut = (data>hRange[0])*(data<hRange[1])
        data = data[cut]
        pl.hist(data, bins, range=(hRange[0],hRange[1]) )
    else: pl.hist(data, bins)

    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    if mean and area_max_plot<np.mean(data): pl.axvline(x=np.mean(data), ls='--', color='r')
    if len(xlim) > 1: pl.xlim(xlim[0],xlim[1])
    if len(ylim) > 1: pl.ylim(ylim[0],ylim[1])
    if logx: pl.xscale("log")
    if logy: pl.yscale("log")
    if len(legHand) > 0: pl.legend(handles=legHand)
    if save: pl.savefig(str(data_dir)+str(name)+".png")
    if show: pl.show()

    return

# For creating basic scatter plots
def basicScatter(xdata, ydata, s=[], c=[], save=False, name="", mean=False, show=False, xlim=[], ylim=[], xlabel="", ylabel="", logx=False, logy=False, area_max_plot=-99999999,legHand=[]):
    pl.figure()
    pl.scatter(xdata, ydata, s=s, c=c)

    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    if mean and area_max_plot<np.mean(xdata): pl.axvline(x=np.mean(xdata), ls='--', color='r')
    if len(xlim) > 1: pl.xlim(xlim[0],xlim[1])
    if len(ylim) > 1: pl.ylim(ylim[0],ylim[1])
    if logx: pl.xscale("log")
    if logy: pl.yscale("log")
    if len(legHand) > 0: pl.legend(handles=legHand)
    if save: pl.savefig(str(data_dir)+str(name)+".png")
    if show: pl.show()
    pl.close()

    return


# Plots of all pulses combined (after cuts)
basicHist(cleanTBA, bins=100, hRange=[-1.01,1.01], mean=True, xlabel="TBA", name="TBA_"+pulse_cut_name, save=save_pulse_plots)

basicHist(cleanRiseTime, bins=100, mean=True, logy=True, xlabel="Rise time, 50-2 (us)", name="RiseTime_"+pulse_cut_name, save=save_pulse_plots)

basicHist(np.log10(cleanArea), bins=100, mean=True, xlabel="log10 Pulse area (phd)", name="log10PulseArea_"+pulse_cut_name, save=save_pulse_plots)

area_max_plot=150
basicHist(cleanArea, bins=125, hRange=[0,area_max_plot], mean=True, xlabel="Pulse area (phd)", area_max_plot=area_max_plot, name="PulseArea_Under150phd"+pulse_cut_name, save=save_pulse_plots)

basicHist(cleanPulseClass, legHand=pc_legend_handles, xlabel="Pulse Class", name="PulseClass_"+pulse_cut_name, save=save_pulse_plots)

basicScatter(cleanTBA, cleanRiseTime, s=1.2, c=pulse_class_colors[cleanPulseClass], xlim=[-1.01,1.01], logy=True, ylim=[.01,4], xlabel="TBA", ylabel="Rise time, 50-2 (us)", legHand=pc_legend_handles, name="RiseTime_vs_TBA_"+pulse_cut_name, save=save_pulse_plots)

basicScatter(cleanArea, cleanRiseTime, s=1.2, c=pulse_class_colors[cleanPulseClass], logx=True, logy=True, xlim=[5,10**6], ylim=[.01,4], xlabel="Pulse area (phd)", ylabel="Rise time, 50-2 (us)", legHand=pc_legend_handles, name="RiseTime_vs_PulseArea_"+pulse_cut_name, save=save_pulse_plots)
#xlim=[0.7*min(p_area.flatten()), 1.5*max(p_area.flatten())]

basicScatter(cleanTBA, cleanArea, s=1.2, c=pulse_class_colors[cleanPulseClass], xlim=[-1.01,1.01], ylim=[0, 6000], xlabel="TBA", ylabel="Pulse area (phd)", legHand=pc_legend_handles, name="PulseArea_vs_TBA_"+pulse_cut_name, save=save_pulse_plots)

# Channel fractional area for all pulses
pl.figure()
for j in range(0, n_channels-1):
    pl.subplot(4,2,j+1)
    pl.hist(cleanAreaChFrac[:,j],bins=100,range=(0,1))
    pl.axvline(x=np.mean(cleanAreaChFrac[:,j]), ls='--', color='r')
    #print("ch {0} area frac mean: {1}".format(j,np.mean(cleanAreaChFrac[:,j])))
    #pl.yscale('log')
    pl.xlabel("Pulse area fraction")
    pl.title('Ch '+str(j))
if save_pulse_plots: pl.savefig(data_dir+"pulse_ch_area_frac_"+pulse_cut_name+".png")

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

basicHist(cleanS1TBA, bins=100, hRange=[-1.01,1.01], mean=True, xlabel="S1 TBA", name="S1TBA_"+pulse_cut_name, save=save_S1S2_plots)
basicHist(cleanS2TBA, bins=100, hRange=[-1.01,1.01], mean=True, xlabel="S2 TBA", name="S2TBA_"+pulse_cut_name, save=save_S1S2_plots)

basicHist(np.log10(cleanS1Area), bins=100, mean=True, xlabel="log10 S1 Area", name="log10_S1_"+pulse_cut_name, save=save_S1S2_plots)
basicHist(np.log10(cleanS2Area), bins=100, mean=True, xlabel="log10 S2 Area", name="log10_S2_"+pulse_cut_name, save=save_S1S2_plots)

basicHist(cleanS1Area, bins=125, mean=True, xlabel="S1 area (phd)", name="S1_"+pulse_cut_name, save=save_S1S2_plots)
basicHist(cleanS2Area, bins=500, mean=True, xlabel="S2 area (phd)", name="S2_"+pulse_cut_name, save=save_S1S2_plots)

# Plots of event-level variables
pl.figure()
pl.scatter(cleanSumS1, np.log10(cleanSumS2), s = 1, c=cleanDT)
pl.xlabel("Sum S1 area (phd)")
pl.ylabel("log10 Sum S2 area")
cbar=pl.colorbar()
cbar.set_label("Drift time (us)")
if save_event_plots: pl.savefig(data_dir+"log10_SumS2_vs_SumS1_"+event_cut_name +".png")

basicHist(np.log10(cleanSumS1), bins=100, mean=True, xlabel="log10 Sum S1 area (phd)", name="log10_SumS1_"+event_cut_name, save=save_event_plots)
basicHist(np.log10(cleanSumS2), bins=100, mean=True, xlabel="log10 Sum S2 area (phd)", name="log10_SumS2_"+event_cut_name, save=save_event_plots)

# Only ever plot this for SS events?
basicHist(cleanDT, bins=50, hRange=[0,10], mean=True, xlabel="Drift time (us)", name="DriftTime_"+event_cut_name, save=save_event_plots)

basicHist(cleanDT_AS, bins=50, hRange=[0,10], mean=True, xlabel="Drift time AS (us)", name="DriftTime_AS", save=save_event_plots)

pl.figure() # Only ever plot this for SS events?
pl.scatter(cleanDT, cleanSumS2)
pl.xlabel("Drift time (us)")
pl.ylabel("Sum S2 area")
# Calculate mean vs drift bin
drift_bins=np.linspace(0,13,50)
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

#pl.show()

# Just for 2S2 events
s2_bool_2s2 = cut_dict['S2'][event_cut_dict['2S2']] # get boolean array of S2 pulses, w/in 2s2 events
s2_ind_array = np.array(np.where(s2_bool_2s2)) # convert to an array of indices
first_s2_ind = tuple(s2_ind_array[:,::2]) # 1st pulse per event is entry 0,2,4,...; use tuple for indexing
second_s2_ind = tuple(s2_ind_array[:,1::2]) # 2nd pulse per event is entry 1,3,5,...
n_2s2 = np.sum(event_cut_dict['2S2'])

area_1st_s2 = p_area[event_cut_dict['2S2']][first_s2_ind]
area_2nd_s2 = p_area[event_cut_dict['2S2']][second_s2_ind]

tstart_1st_s2 = p_start[event_cut_dict['2S2']][first_s2_ind]
tstart_2nd_s2 = p_start[event_cut_dict['2S2']][second_s2_ind]
dt_2s2 = tscale*(tstart_2nd_s2 - tstart_1st_s2)

pl.figure()
pl.scatter(np.log10(area_1st_s2), np.log10(area_2nd_s2), c=dt_2s2)
pl.plot([1.5,5],[1.5,5],c='r')
cbar=pl.colorbar()
cbar.set_label("Time between S2s (us)")
pl.xlabel('log10(1st S2 area)')
pl.ylabel('log10(2nd S2 area)')
if save_2S2_plots: pl.savefig(data_dir+"2S2_log10_area2_vs_log10_area1.png")

pl.figure()
pl.hist(area_2nd_s2/area_1st_s2,range=(0,4),bins=50)
pl.xlabel('2nd S2 area/1st S2 area')
if save_2S2_plots: pl.savefig(data_dir+"2S2_area2_over_area1.png")

pl.figure()
pl.hist(dt_2s2,bins=50)
pl.xlabel('Time between S2s (us)')
if save_2S2_plots: pl.savefig(data_dir+"2S2_time_diff.png")
