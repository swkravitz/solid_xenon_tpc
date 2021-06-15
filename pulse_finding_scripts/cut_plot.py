import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from matplotlib.patches import Circle, PathPatch, Rectangle
import time
import matplotlib.colors as mcolors


# Turns data into (x,y) points of histogram to plot
def histToPlot(data, bins):
    [histData,bins] = np.histogram(data, bins=bins)
    binCenters = np.array([0.5 * (bins[j] + bins[j+1]) for j in range(len(bins)-1)])
    return binCenters, histData

# For creating basic histograms
def basicHist(data, bins=100, save=False, name="", mean=False, show=False, hRange=[], xlim=[], ylim=[], xlabel="", ylabel="", logx=False, logy=False, area_max_plot=-99999999,legHand=[],data_dir=None):
    pl.figure()
    if len(hRange) > 1:
        cut = (data>hRange[0])*(data<hRange[1])
        data = data[cut]
        pl.hist(data, bins, range=(hRange[0],hRange[1]), histtype='step' )
    else: pl.hist(data, bins)

    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    if mean and area_max_plot<np.mean(data): pl.axvline(x=np.mean(data), ls='--', color='r')
    if len(xlim) > 1: pl.xlim(xlim[0],xlim[1])
    if len(ylim) > 1: pl.ylim(ylim[0],ylim[1])
    if logx: pl.xscale("log")
    if logy: pl.yscale("log")
    if len(legHand) > 0: pl.legend(handles=legHand)
    if save and data_dir is not None: pl.savefig(str(data_dir)+str(name)+".png")
    if show: pl.show()

    return

# For creating basic scatter plots
def basicScatter(xdata, ydata, s=[], c=[], save=False, name="", mean=False, show=False, xlim=[], ylim=[], xlabel="", ylabel="", logx=False, logy=False, area_max_plot=-99999999,legHand=[],data_dir=None,showsipms=False):
    pl.figure()
    if showsipms:
        w = 0.6 # Photosensitive area in cm
        d_x = 1.23 # cm
        d_y = 1.14 # cm
        r = 1.5 # cm, active TPC radius
        c1 = pl.Circle((0., 0.), r, fill=False, facecolor='grey', edgecolor='grey',alpha=1)
        r1 = pl.Rectangle((d_x/2.-w/2., d_y/2.-w/2.), w, w, fill=False, facecolor='grey', edgecolor='grey',alpha=1)
        r2 = pl.Rectangle((d_x/2.-w/2., -d_y/2.-w/2.), w, w, fill=False, facecolor='grey', edgecolor='grey',alpha=1)
        r3 = pl.Rectangle((-d_x/2.-w/2., -d_y/2.-w/2.), w, w, fill=False, facecolor='grey', edgecolor='grey',alpha=1)
        r4 = pl.Rectangle((-d_x/2.-w/2., d_y/2.-w/2.), w, w, fill=False, facecolor='grey', edgecolor='grey',alpha=1)
        pl.gca().add_patch(c1)
        pl.gca().add_patch(r1)
        pl.gca().add_patch(r2)
        pl.gca().add_patch(r3)
        pl.gca().add_patch(r4)
        pl.axis('square')
    pl.scatter(xdata, ydata, s=s, c=c)

    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    
    if mean and area_max_plot<np.mean(xdata): pl.axvline(x=np.mean(xdata), ls='--', color='r')
    if len(xlim) > 1: pl.xlim(xlim[0],xlim[1])
    if len(ylim) > 1: pl.ylim(ylim[0],ylim[1])
    if logx: pl.xscale("log")
    if logy: pl.yscale("log")
    if len(legHand) > 0: pl.legend(handles=legHand)
    if save and data_dir is not None: pl.savefig(str(data_dir)+str(name)+".png")
    if show: pl.show()
    pl.close()

    return

# For creating heatmaps, i.e. 2D histograms (can be weighted if desired)
def basicHeatmap(xdata, ydata, weights=None, bins=40, save=False, name="", show=False, xlim=[], ylim=[], cmin=1e-12, cmax=None, xlabel="", ylabel="", logx=False, logy=False, legHand=[], data_dir=None):
    pl.figure()
    if len(xlim)>1 and len(ylim)>1: hist_range = [xlim, ylim]
    else: hist_range = None
    _, _, _, image = pl.hist2d(xdata, ydata, bins=bins, range=hist_range, weights=weights, cmin=cmin, cmax=cmax)
    pl.colorbar(image)

    pl.xlabel(xlabel)
    pl.ylabel(ylabel)

    if logx: pl.xscale("log")
    if logy: pl.yscale("log")
    if len(legHand) > 0: pl.legend(handles=legHand)
    if save and data_dir is not None: pl.savefig(str(data_dir)+str(name)+".png")
    if show: pl.show()
    pl.close()

    return

def make_plots(data_dir):
    # set plotting style
    mpl.rcParams['font.size']=10
    mpl.rcParams['legend.fontsize']='small'
    mpl.rcParams['figure.autolayout']=True
    mpl.rcParams['figure.figsize']=[8.0,6.0]
    mpl.rcParams['figure.max_open_warning']=False

    # use for coloring pulses
    pulse_class_colors = np.array(['blue', 'green', 'red', 'magenta', 'darkorange'])
    pulse_class_labels = np.array(['Other', 'S1-like LXe', 'S1-like gas', 'S2-like', 'Merged S1/S2'])
    pc_legend_handles=[]
    for class_ind in range(len(pulse_class_labels)):
        pc_legend_handles.append(mpl.patches.Patch(color=pulse_class_colors[class_ind], label=str(class_ind)+": "+pulse_class_labels[class_ind]))


    # ==================================================================
    # define DAQ and other parameters
    tscale = (8.0/4096.0)     # = 0.002 µs/sample, time scale

    n_sipms = 8
    n_channels = n_sipms+1 # includes sum
    d_between_SiPM_center_x = 1.23 # cm
    d_between_SiPM_center_y = 1.14 # cm
    
    # define top, bottom channels
    n_top = int((n_channels-1)/2)
    top_channels=np.array(range(n_top),int)
    bottom_channels=np.array(range(n_top,2*n_top),int)

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
    center_top_x = listrq['center_top_x']
    center_top_y = listrq['center_top_y']
    center_bot_x = listrq['center_bot_x']
    center_bot_y = listrq['center_bot_y']
    center_bot_x_d = center_bot_x * d_between_SiPM_center_x/2
    center_bot_y_d = center_bot_y * d_between_SiPM_center_y/2
    center_top_x_d = center_top_x * d_between_SiPM_center_x/2
    center_top_y_d = center_top_y * d_between_SiPM_center_y/2

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
    cut_dict['SmallS1'] = cut_dict['S1']*(p_area<500)
    cut_dict['LargeS1'] = cut_dict['S1']*(p_area>500)
    cut_dict['TopCo'] = cut_dict['S1']*(p_tba>0.0)*(p_area>200)
    cut_dict['TopS1'] = cut_dict['S1']*(p_tba>0.75)
    cut_dict['PoS1'] = cut_dict['S1']*(p_tba<-0.75)*(p_tba>-1)*(p_area>5000)*(p_area<30000)
    cut_dict['PoSmallS1'] = cut_dict['S1']*(p_tba<-0.25)*(p_tba>-1)*(p_area>0)*(p_area<2000)
    cut_dict['PoMedS1'] = cut_dict['S1']*(p_tba<-0.25)*(p_tba>-1)*(p_area>2000)*(p_area<5000)
    cut_dict['PoMedLgS1'] = cut_dict['S1']*(p_tba<-0.75)*(p_tba>-1)*(p_area>5000)*(p_area<10000)
    cut_dict['S1_200-400phd'] = cut_dict['S1']*(p_area>200)*(p_area<400)
    cut_dict['S1_0-200phd'] = cut_dict['S1']*(p_area>0)*(p_area<200)
    SS_cut = drift_Time > 0

    # Pick which cut from cut_dict to apply here and whether to save plots
    save_pulse_plots=True # One entry per pulse
    save_S1S2_plots=True # One entry per S1 (S2) pulse
    save_event_plots=True # One entry per event
    save_2S2_plots=True # One entry per event w/ 2 S2s
    save_PoS1_plots=False # One entry per event w/ 1 Po S1, specific to studies of S2 multiplicity
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
    cleanCenterBottomX = center_bot_x_d[pulse_cut]
    cleanCenterBottomY = center_bot_y_d[pulse_cut]
    cleanCenterTopX = center_top_x_d[pulse_cut]
    cleanCenterTopY = center_top_y_d[pulse_cut]

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
    event_cut_dict["Po"] = (drift_Time>4)*np.any((p_tba<-0.0)*(p_tba>-0.7)*(p_area>1000)*(p_area<4000), axis=1)#np.any((p_tba<-0.85)*(p_tba>-0.91)*(p_area>1500)*(p_area<2700), axis=1) # true if any pulse in event matches these criteria
    event_cut_dict["Po_AS"] = (drift_Time_AS>0.)*(drift_Time_AS<1.5)*np.any((p_tba<-0.6)*(p_tba>-1)*(p_area>5000)*(p_area<20000), axis=1)#np.any((p_tba<-0.85)*(p_tba>-0.91)*(p_area>1500)*(p_area<2700), axis=1) # true if any pulse in event matches these criteria
    event_cut_dict["lg_S1"] = (drift_Time>0)*np.any((p_area>1000.)*cut_dict["S1"], axis=1) # true if any S1 has area>1000
    event_cut_dict["2S2"] = (n_s2 == 2)
    event_cut_dict["PoS1"] = (np.sum(cut_dict['PoS1'], axis=1)==1)#*(n_s2>0)

    event_cut_name = "SS"#"Po"#"lg_S1"
    event_cut = event_cut_dict[event_cut_name]
    cleanSumS1 = sum_s1_area[event_cut]
    cleanSumS2 = sum_s2_area[event_cut]
    # For events passing event_cut, get area-weighted avg of TBA for S1s only
    cleanAvgS1TBA = np.sum(p_area[event_cut]*p_tba[event_cut]*s1_cut[event_cut],axis=1)/np.sum(p_area[event_cut]*s1_cut[event_cut],axis=1)
    cleanSumS1TBA = p_tba[event_cut_dict['SS'][:,np.newaxis]*cut_dict['S1']]
    cleanDT = drift_Time[event_cut]
    cleanDT_AS = drift_Time_AS[event_cut]
    print("number of events found passing cut "+event_cut_name+" = {0:d} ({1:g}%)".format(np.sum(event_cut),np.sum(event_cut)*100./n_events))

    # =============================================================
    # =============================================================
    # now make plots of interesting pulse quantities


    # Plots of all pulses combined (after cuts)
    basicHist(cleanTBA, bins=100, hRange=[-1.01,1.01], mean=True, xlabel="TBA", name="TBA_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)

    basicHist(cleanRiseTime, bins=100, mean=True, logy=True, xlabel="Rise time, 50-2 (us)", name="RiseTime_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)

    basicHist(np.log10(cleanArea), bins=100, mean=True, xlabel="log10 Pulse area (phd)", name="log10PulseArea_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)

    area_max_plot=150
    basicHist(cleanArea, bins=125, hRange=[0,area_max_plot], mean=True, xlabel="Pulse area (phd)", area_max_plot=area_max_plot, name="PulseArea_Under150phd"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)

    basicHist(cleanPulseClass, legHand=pc_legend_handles, xlabel="Pulse Class", name="PulseClass_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)

    basicScatter(cleanTBA, cleanRiseTime, s=1.2, c=pulse_class_colors[cleanPulseClass], xlim=[-1.01,1.01], logy=True, ylim=[.01,4], xlabel="TBA", ylabel="Rise time, 50-2 (us)", legHand=pc_legend_handles, name="RiseTime_vs_TBA_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)

    basicScatter(cleanArea, cleanRiseTime, s=1.2, c=pulse_class_colors[cleanPulseClass], logx=True, logy=True, xlim=[5,10**6], ylim=[.01,4], xlabel="Pulse area (phd)", ylabel="Rise time, 50-2 (us)", legHand=pc_legend_handles, name="RiseTime_vs_PulseArea_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)
    #xlim=[0.7*min(p_area.flatten()), 1.5*max(p_area.flatten())]

    basicScatter(cleanTBA, cleanArea, s=1.2, c=pulse_class_colors[cleanPulseClass], xlim=[-1.01,1.01], ylim=[0, 30000], xlabel="TBA", ylabel="Pulse area (phd)", legHand=pc_legend_handles, name="PulseArea_vs_TBA_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)
    basicScatter(cleanTBA, cleanArea, s=1.2, c=pulse_class_colors[cleanPulseClass], xlim=[-1.01,1.01], ylim=[0, 1000], xlabel="TBA", ylabel="Pulse area (phd)", legHand=pc_legend_handles, name="PulseArea_small_vs_TBA_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)

    basicScatter(cleanCenterBottomX, cleanCenterBottomY, s=1.2, c=pulse_class_colors[cleanPulseClass], xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], xlabel="x (cm)", ylabel="y (cm)", legHand=pc_legend_handles, name="BottomCentroid_"+pulse_cut_name, save=save_pulse_plots, data_dir=data_dir, showsipms=True)
    basicHeatmap(cleanCenterBottomX, cleanCenterBottomY, xlim=[-0.7, 0.7], ylim=[-0.7, 0.7], xlabel="x (cm)",
             ylabel="y (cm)", name="BottomCentroidMap_" + pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)
    basicScatter(cleanCenterTopX, cleanCenterTopY, s=1.2, c=pulse_class_colors[cleanPulseClass], xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], xlabel="x (cm)", ylabel="y (cm)", legHand=pc_legend_handles, name="TopCentroid_" + pulse_cut_name, save=save_pulse_plots, data_dir=data_dir, showsipms=True)
    basicHeatmap(cleanCenterTopX, cleanCenterTopY, xlim=[-0.7, 0.7], ylim=[-0.7, 0.7], xlabel="x (cm)", ylabel="y (cm)",
             name="TopCentroidMap_" + pulse_cut_name, save=save_pulse_plots, data_dir=data_dir)

    # Channel fractional area for all pulses
    pl.figure()
    for j in range(0, n_channels-1):
        pl.subplot(4,2,j+1)
        pl.hist(cleanAreaChFrac[:,j],bins=100,range=(0,1), histtype='step')
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
        pl.hist(cleanS1AreaChFrac[:,j],bins=100,range=(0,1), histtype='step')
        pl.axvline(x=np.mean(cleanS1AreaChFrac[:,j]), ls='--', color='r')
        #print("S1 ch {0} area frac mean: {1}".format(j,np.mean(cleanS1AreaChFrac[:,j])))
        #pl.yscale('log')
        pl.xlabel("S1 area fraction")
        pl.title('Ch '+str(j))
    if save_S1S2_plots: pl.savefig(data_dir+"S1_ch_area_frac_"+pulse_cut_name+".png")

    pl.figure()
    for j in range(0, n_channels-1):
        pl.subplot(4,2,j+1)
        pl.hist(cleanS2AreaChFrac[:,j],bins=100,range=(0,1), histtype='step')
        pl.axvline(x=np.mean(cleanS2AreaChFrac[:,j]), ls='--', color='r')
        #print("S2 ch {0} area frac mean: {1}".format(j,np.mean(cleanS2AreaChFrac[:,j])))
        #pl.yscale('log')
        pl.xlabel("S2 area fraction")
        pl.title('Ch '+str(j))
    if save_S1S2_plots: pl.savefig(data_dir+"S2_ch_area_frac_"+pulse_cut_name+".png")

    basicHist(cleanS1TBA, bins=100, hRange=[-1.01,1.01], mean=True, xlabel="S1 TBA", name="S1TBA_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(cleanS2TBA, bins=100, hRange=[-1.01,1.01], mean=True, xlabel="S2 TBA", name="S2TBA_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)

    basicHist(np.log10(cleanS1Area), bins=100, mean=True, xlabel="log10 S1 Area", name="log10_S1_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(np.log10(cleanS2Area), bins=100, mean=True, xlabel="log10 S2 Area", name="log10_S2_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)

    basicHist(cleanS1Area, bins=125, mean=True, xlabel="S1 area (phd)", name="S1_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(cleanS1Area, bins=100, hRange=[0, 200], mean=True, xlabel="S1 area (phd)", name="S1_small_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(cleanS1Area, bins=100, hRange=[0, 1000], mean=True, xlabel="S1 area (phd)", name="S1_med_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(cleanS1Area, bins=100, hRange=[0, 30000], mean=True, xlabel="S1 area (phd)", name="S1_lg_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(cleanS2Area, bins=500, mean=True, xlabel="S2 area (phd)", name="S2_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(cleanS2Area, bins=500, hRange=[0, 1000], mean=True, xlabel="S2 area (phd)", name="S2_small_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(cleanS2Area, bins=500, hRange=[0, 10000], mean=True, xlabel="S2 area (phd)", name="S2_med_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)
    basicHist(cleanS2Area, bins=500, hRange=[0, 500000], mean=True, xlabel="S2 area (phd)", name="S2_lg_"+pulse_cut_name, save=save_S1S2_plots, data_dir=data_dir)

    # Plots of event-level variables
    pl.figure()
    pl.scatter(cleanSumS1, np.log10(cleanSumS2), s = 1, c=cleanDT)
    pl.xlabel("Sum S1 area (phd)")
    pl.ylabel("log10 Sum S2 area")
    cbar=pl.colorbar()
    cbar.set_label("Drift time (us)")
    if save_event_plots: pl.savefig(data_dir+"log10_SumS2_vs_SumS1_"+event_cut_name +".png")

    pl.figure()
    pl.scatter(cleanAvgS1TBA, np.log10(cleanSumS2), s = 1, c=cleanDT_AS)
    pl.xlabel("Avg S1 TBA")
    pl.ylabel("log10 Sum S2 area")
    cbar=pl.colorbar()
    cbar.set_label("Drift time (us)")
    if save_event_plots: pl.savefig(data_dir+"log10_SumS2_vs_S1TBA_"+event_cut_name +".png")

    basicHist(np.log10(cleanSumS1), bins=100, mean=True, xlabel="log10 Sum S1 area (phd)", name="log10_SumS1_"+event_cut_name, save=save_event_plots, data_dir=data_dir)
    basicHist(np.log10(cleanSumS2), bins=100, mean=True, xlabel="log10 Sum S2 area (phd)", name="log10_SumS2_"+event_cut_name, save=save_event_plots, data_dir=data_dir)
    basicHist(cleanSumS2, bins=100, hRange=[0,400000], mean=True, xlabel="Sum S2 area (phd)", name="SumS2_"+event_cut_name, save=save_event_plots, data_dir=data_dir)

    # Only ever plot this for SS events?
    basicHist(cleanDT, bins=50, hRange=[0,10], mean=True, xlabel="Drift time (us)", name="DriftTime_"+event_cut_name, save=save_event_plots, data_dir=data_dir)
    basicHist(drift_Time_AS[drift_Time_AS>0], bins=50, hRange=[0,10], mean=True, xlabel="Drift time AS (us)", name="DriftTime_AS", save=save_event_plots, data_dir=data_dir)
    basicHist(cleanDT_AS, bins=50, hRange=[0,10], mean=True, xlabel="Drift time AS (us)", name="DriftTime_AS_"+event_cut_name, save=save_event_plots, data_dir=data_dir)

    pl.figure() # Only ever plot this for SS events?
    pl.scatter(cleanDT_AS, cleanSumS2)
    pl.xlabel("Drift time (us)")
    pl.ylabel("Sum S2 area")
    # Calculate mean vs drift bin
    drift_bins=np.linspace(0,8,50)
    drift_ind=np.digitize(cleanDT_AS, bins=drift_bins)
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
    
    # S1 area vs drift time w/ TBA as color scale:
    pl.figure()
    pl.scatter(cleanDT, cleanSumS1, s=1.5, c=cleanAvgS1TBA, cmap='plasma' )
    pl.xlabel("Drift Time (us)")
    pl.ylabel("Sum S1 area (phd)")
    pl.xlim(0,9)
    pl.ylim(10,30000)
    pl.yscale('log')
    pl.clim(-1,1)
    cbar=pl.colorbar()
    cbar.set_label("S1 TBA")
    if save_event_plots: pl.savefig(data_dir+"SumS1_vs_DriftTime_"+event_cut_name +".png")
    
    
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
    pl.hist(area_2nd_s2/area_1st_s2,range=(0,4),bins=50, histtype='step')
    pl.xlabel('2nd S2 area/1st S2 area')
    if save_2S2_plots: pl.savefig(data_dir+"2S2_area2_over_area1.png")

    pl.figure()
    pl.hist(dt_2s2,bins=50, histtype='step')
    pl.xlabel('Time between S2s (us)')
    if save_2S2_plots: pl.savefig(data_dir+"2S2_time_diff.png")

    # Events w/ 1 up-going Po-like S1
    if save_PoS1_plots:
        n_lg_s2 = np.sum((p_area>100.)*cut_dict['S2'], axis=1)
        event_cut_dict['PoS1lgS2'] = event_cut_dict['PoS1']*n_lg_s2>0
        n_lg_s2_PoS1 = n_lg_s2[event_cut_dict['PoS1lgS2']]

        s1_bool_PoS1 = cut_dict['PoS1'][event_cut_dict['PoS1lgS2']] # get boolean array of Po S1 pulses, w/in events w/ one Po S1
        s1_ind_array = np.array(np.where(s1_bool_PoS1)) # convert to an array of indices
        s1_ind = tuple(s1_ind_array)
        s1_TBA_PoS1 = p_tba[event_cut_dict['PoS1lgS2']][s1_ind]

        pl.figure()
        pl.scatter(s1_TBA_PoS1, n_lg_s2_PoS1)
        pl.xlabel('S1 TBA')
        pl.ylabel('n_S2')
        TBA_bins=np.linspace(-0.8,0,20)
        TBA_ind=np.digitize(s1_TBA_PoS1, bins=TBA_bins)
        ns2_means=np.zeros(np.shape(TBA_bins))
        ns2_std_err=np.ones(np.shape(TBA_bins))*0#10000
        for i_bin in range(len(TBA_bins)):
            found_i_bin = np.where(TBA_ind==i_bin)
            ns2_i_bin = n_lg_s2_PoS1[found_i_bin]
            if len(ns2_i_bin) < 1: continue
            ns2_means[i_bin]=np.mean(ns2_i_bin) # Median instead of mean, better at ignoring outliers
            ns2_std_err[i_bin]=np.std(ns2_i_bin)/np.sqrt(len(ns2_i_bin))
        pl.errorbar(TBA_bins, ns2_means, yerr=ns2_std_err, linewidth=3, elinewidth=3, capsize=5, capthick=4, color='red')
        pl.ylim(bottom=0)

        pl.savefig(data_dir+"PoS1lgS2_nS2_vs_S1_TBA.png")

# This is what actually gets run when calling cut_plot.py as a script
def main():
    with open("path.txt", 'r') as path:
        #data_dir = path.read()
        data_dir = path.readline().strip() #only read first line of path.txt
        print('\nOpening RQ file in {:s} ...\n'.format(data_dir))

    make_plots(data_dir)

if __name__ == "__main__":
    main()

