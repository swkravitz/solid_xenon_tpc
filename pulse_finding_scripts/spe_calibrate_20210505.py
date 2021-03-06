import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import time
import sys
from scipy import stats

import PulseFinderScipy as pf
import PulseQuantities as pq
import PulseClassification as pc

import PulseFinderSPE_crystalize as pfSPE

data_dir = "/home/xaber/caen/wavedump-3.8.2/data/20210512/20210512_1745_spe_calibration_match/"

chs = [5, 7] # Sipm channels needed to be processed
SPEMode = True
LED = False # LED mode, expects signal at ~2us
plotyn = False # Waveform plotting
saveplot = True # Save RQ plots

# set plotting style
mpl.rcParams['font.size']=12
mpl.rcParams['axes.labelsize']=12
mpl.rcParams['legend.fontsize']='small'
mpl.rcParams['figure.autolayout']=True
mpl.rcParams['figure.figsize']=[8.0,6.0]

# ==================================================================
# define DAQ and other parameters
#wsize = 12500             # size of event window in samples. 1 sample = 2 ns.
#event_window = 2.  # in us
wsize = 1030  # samples per waveform # 12500 for 25 us
vscale = (500.0/16384.0) # vertical scale, = 0.031 mV/ADCC for 0.5 Vpp dynamic range
tscale = (8.0/4096.0)     # = 0.002 µs/sample, time scale

# Set range to look for pulses
if LED:
    left_bound = int(0.7/tscale)
    right_bound = int(1.5/tscale)
else:
    left_bound = 0
    right_bound = wsize

block_size = 5000
n_block = 40
max_evts = n_block*block_size#5000  # 25000 # -1 means read in all entries; 25000 is roughly the max allowed in memory on the DAQ computer
max_pts = -1  # do not change
max_pulses = 6
if max_evts > 0:
    max_pts = wsize * max_evts
load_dtype = "int16"


n_sipms = 1
list_rq = {}

#cleanHeight = np.zeros((n_sipms, max_evts, max_pulses))
#cleanStart = np.zeros((n_sipms, max_evts, max_pulses))

# p_sarea instead of p_int
#cleanSarea = np.zeros((n_sipms, max_evts, max_pulses))
cleanArea = []
# ==================================================================

#load in raw data

t_start = time.time()

for ch_index in chs:
    working_ch = [ch_index]


    # RQ's for SPE analysis
    
    p_start = np.zeros((n_sipms, max_evts, max_pulses), dtype=np.int)
    p_end   = np.zeros((n_sipms, max_evts, max_pulses), dtype=np.int)

    # p_sarea instead of p_int
    p_sarea = np.zeros((n_sipms, max_evts, max_pulses))
    p_area = np.zeros((n_sipms, max_evts, max_pulses))
    p_max_height = np.zeros((n_sipms, max_evts, max_pulses))
    p_width = np.zeros((n_sipms, max_evts, max_pulses))
    p_noisearea = [[]]*n_sipms

    n_pulses = np.zeros((n_sipms, max_evts), dtype=np.int)

    e_fft = np.zeros(wsize, dtype=np.complex128)


    inn=""

    # Loop over blocks
    for j in range(n_block):
        ch_data = []
        
        try:
            ch_data.append(np.fromfile(data_dir + "wave"+str(ch_index)+".dat", dtype=load_dtype, offset = block_size*wsize*j, count=wsize*block_size))
        except:
            print(data_dir + "wave"+str(ch_index)+".dat")
            print ("no data from ch%d"%ch_index)
            ch_data.append(np.zeros(10000000))

        t_end_load = time.time()
        print("Time to load files: ", t_end_load-t_start)

        # scale waveforms to get units of mV/sample
        # then for each channel ensure we 
        # have an integer number of events
        array_dtype = "float32" # using float32 reduces memory for big files, otherwise implicitly converts to float64

        # matrix of all channels including the sum waveform
        v_matrix_all_ch = []
    
        V = vscale * ch_data[0].astype(array_dtype)
        V = V[:int(len(V) / wsize) * wsize]
        V = V.reshape(int(V.size / wsize), wsize) # reshape to make each channel's matrix of events
        v_matrix_all_ch.append(V)
    #        if ch_ind==0: v_sum = np.copy(V)
     #       else: v_sum += V
      #  v_matrix_all_ch.append(v_sum)

        
        # create a time axis in units of µs:
        x = np.arange(0, wsize, 1)
        t = tscale*x
        t_matrix = np.repeat(t[np.newaxis,:], V.size/wsize, 0)

        # Note: if max_evts != -1, we won't load in all events in the dataset
        n_events = int(v_matrix_all_ch[0].shape[0])
        if n_events == 0: break
            

        # Baseline subtraction
        # baseline subtracted (bls) waveforms saved in this matrix:
        v_bls_matrix_all_ch = np.zeros( np.shape(v_matrix_all_ch), dtype=array_dtype) # dims are (chan #, evt #, sample #)
        
        t_end_wfm_fill = time.time()
        print("Time to fill all waveform arrays: ", t_end_wfm_fill - t_end_load)

        print("Events to process: ",n_events)

        
        # For LED, looks 0.5 us before expected range of pulse
        if LED:
            baseline_start = left_bound - int(0.2/tscale)
            baseline_end = left_bound
        
            for b in range(0, n_events):

                baselines = [ np.mean( ch_j[b,baseline_start:baseline_end] ) for ch_j in v_matrix_all_ch ]
            
                ch_data = [ch_j[b,:]-baseline_j for (ch_j,baseline_j) in zip(v_matrix_all_ch,baselines)]
            
                v_bls_matrix_all_ch[:,b,:] = ch_data

        # Does preliminary baseline subtraction
        else:
            for bb in range(n_events):
                baselines = [ pfSPE.findBase1(ch_j[bb,:]) for ch_j in v_matrix_all_ch ]
                ch_data = [ch_j[bb,:]-baseline_j for (ch_j,baseline_j) in zip(v_matrix_all_ch,baselines)]
                
                v_bls_matrix_all_ch[:,bb,:] = ch_data


        # ==================================================================
        # ==================================================================
        # now setup for pulse finding on the baseline-subtracted sum waveform

        print("Running pulse finder on {:d} events...".format(n_events))

        # Loop index key:
        # j = blocks 
        # b = baseline sub 
        # i = events 
        # k = sipms
        # m = start time sorting
        # n = pulses

        # Loop over events
        for i in range(j*block_size, j*block_size+n_events):
            if i%1000==0: print("Event #",i)
            
            # Loop over channels, slowest part of the code
            # Have to do a loop, pulse finder does not like v_bls_matrix_all_ch[:,i,:] 
            for k in working_ch:
            
                # Do a simple integral over desired range
                # If you change the range, make sure to change the plot x_label
                l_area = int(2.06/tscale) #2.06
                r_area = int(2.30/tscale) #2.18
                p_sarea[0,i,0] = pfSPE.simpleSumArea(v_bls_matrix_all_ch[0,i-j*block_size,:],[l_area,r_area])

                # Unfinished FFT calculator. Not sure we need this?
                e_fft += np.abs(np.fft.fft(v_bls_matrix_all_ch[0,i-j*block_size,:] ))

                # Do a proper pulse finder
                if LED:
                    start_times, end_times, peaks, data_conv = pfSPE.findLEDSPEs( v_bls_matrix_all_ch[0, i-j*block_size, left_bound:right_bound] )
                    start_times += left_bound
                    end_times +=left_bound
                    start_times = np.array([start_times])
                    end_times = np.array([end_times])

                # Preliminary baseline should already be done
                else:
                    start_times, end_times, peaks, baselines, data_conv,baselines_start,baselines_end = pfSPE.findDarkSPEs(v_bls_matrix_all_ch[0, i-j*block_size, :])

                base_win = int(0.2/tscale) # 0.2 us
                # Calculate RQ's from pulse finder
                if SPEMode:
                    for n in range(len(start_times)):
                        if n > max_pulses - 1: break
                        if start_times[n] <end_times[n]:
                            p_area[0,i,n] = pq.GetPulseArea(start_times[n],end_times[n],v_bls_matrix_all_ch[0, i-j*block_size, :] - baselines[n])
                            p_max_height[0,i,n] = pq.GetPulseMaxHeight(start_times[n],end_times[n],v_bls_matrix_all_ch[0, i-j*block_size, :]- baselines[n])
                            p_width[0,i,n] = start_times[n] - end_times[n]
                            p_start[0,i,n] = start_times[n]
                            p_end[0,i,n] = end_times[n]
                
                    
                        n_pulses[0,i] += len(start_times)
                elif LED and start_times[0] <end_times[0]:
                    p_area[0,i,0] = pq.GetPulseArea(start_times[0],end_times[0],v_bls_matrix_all_ch[0, i-j*block_size, :])
                    p_max_height[0,i,0] = pq.GetPulseMaxHeight(start_times[0],end_times[0],v_bls_matrix_all_ch[0, i-j*block_size, :])
                    p_width[0,i,0] = start_times[0] - end_times[0]
                    p_start[0,i,0] = start_times[0]
                    p_end[0,i,0] = end_times[0]
                else:
                    start_times[0] = left_bound
                    end_times[0] = left_bound+base_win
                    p_area[0,i,0] = pq.GetPulseArea(start_times[0],end_times[0],v_bls_matrix_all_ch[0, i-j*block_size, :])
                    p_max_height[0,i,0] = pq.GetPulseMaxHeight(left_bound,right_bound,v_bls_matrix_all_ch[0, i-j*block_size, :])
                    p_width[0,i,0] = -left_bound+right_bound
                    p_start[0,i,0] = left_bound
                    p_end[0,i,0] = right_bound
                    

                if SPEMode:
                    noisewin = len(start_times)==0
                    if  noisewin:
                        p_noisearea[0].append(pq.GetPulseArea(base_win,2*base_win,v_bls_matrix_all_ch[0, i-j*block_size, :] - baselines[0]))
                # Plotter
    #            areacut = p_area[0,i,0] > 0
                areacut = np.logical_and(p_area[0,i,0]*tscale*1000>40,p_area[0,i,0]*tscale*1000<200)

                # horizontal lines: @ zero, baseline, height
                
                if not inn == 'q' and plotyn and areacut and k==7:
                    # Plot something
                    fig = pl.figure(1,figsize=(10, 7))
                    pl.grid(b=True,which='major',color='lightgray',linestyle='--')
                    if LED:
                        pl.plot(t_matrix[j,:], v_bls_matrix_all_ch[0,i-j*block_size,:], color='blue')
                        pl.plot(t_matrix[j,left_bound:right_bound], data_conv, color='red')
                    else:
                        pl.plot(t_matrix[j,:], v_bls_matrix_all_ch[0,i-j*block_size,:]-baselines[0], color='blue')
                        pl.plot(t_matrix[j,:], data_conv, color='red')
                    for pulse in range(len(start_times)):
                        if start_times[pulse]!=0:
                            pl.axvspan(start_times[pulse] * tscale, end_times[pulse] * tscale, alpha=0.25, color='green',label="Pulse area = {:0.3f} mV*ns".format(p_area[0,i,pulse]*tscale*1000))
                            if not LED:
                                pl.axvspan(baselines_start[pulse]*tscale, baselines_end[pulse]*tscale, alpha=0.25, color='purple',label="baseline")
                            print ("area:",p_area[0,i,pulse]*tscale,"start:",start_times[pulse]*tscale,"end:",end_times[pulse]*tscale)
                    if SPEMode and noisewin: # plot the area for calculating noise area
                        pl.axvspan(base_win*tscale, 2*base_win*tscale, alpha=0.25, color='orange',label="noise window")
    #                pl.hlines(0.06,0,4,color='orange',label='Height treshhold = 0.1')
                    pl.hlines(0,0,4,color="black")

    #                pl.ylim([-0.5,1])
                    pl.xlim([0,2])
                    pl.title("Channel "+str(k)+", Event "+str(i) )
                    pl.xlabel('Time (us)')
                    pl.ylabel('mV')
                    pl.legend(loc="upper right")
                    pl.draw()
                    pl.ion()
                    pl.show()
    #                pl.show(block=0)
                    inn = input("Press enter to continue, q to stop plotting, evt # to skip to # (forward only)")
                    pl.close()
    #                fig.clf()


                
    # End of pulse finding and plotting event loop

    n_events = i
    print("total number of events processed:", n_events+1)
    if SPEMode:
        for k in working_ch:
            print ("total number of dark pulses on ch%d:"%working_ch[0],np.sum(n_pulses[0,:]))

    # Clean up and Plotting

    def SPE_Sarea_Hist(data,sipm_n):
        pl.figure()
        pl.hist(data, 600)
        pl.vlines(0,0,3*pow(10,3),colors="red")

        r=(-0.02,0.05)
        xticksteps = 0.01
        pl.xticks(np.arange(r[0],r[1]+xticksteps, xticksteps))
        pl.xlim([r[0],r[1]])
        #pl.yscale("log")

        pl.xlabel("Integral from 2.06-2.30 us (mV*us)")
        pl.title("Channel "+str(sipm_n))
        if saveplot: pl.savefig(data_dir+"SPE_int_sipm_lin"+str(sipm_n)+".png")
        return

    def SPE_Area_Hist(data,sipm_n):
        pl.figure()
        nbin=300
        r=(0,0.1)
        #pl.hist(data, bins=nbin,range=r,histtype='step',linewidth='1')
        pl.hist(data, bins=nbin,range=r)#,histtype='step',linewidth='1')
        
        
        #sel = (data>0.02) & (data<0.06)
        #reddata = data*sel
        #final = reddata[reddata>0]
        #mu, sig = stats.norm.fit(final)
        #x = np.linspace(r[0], r[1], 1000)
        #pdf = stats.norm.pdf(x, mu, sig)
        #pl.plot(x, pdf*len(final)/len(data), color='salmon',linewidth=1)
        
        xticksteps = 0.01
        pl.xticks(np.arange(r[0],r[1]+xticksteps, xticksteps))
        pl.title("Channel "+str(sipm_n))
        pl.xlabel("Pulse Area (mV*us)")
        if saveplot: pl.savefig(data_dir+"SPE_area_sipm_"+str(sipm_n)+".png")
        #pl.show()
        return 

    def SPE_Height_Hist(data,sipm_n):
        pl.figure()
        nbin=200
        pl.hist(data, bins=nbin,histtype='step',linewidth='1')
        #pl.xlim([0,0.3])
        pl.title("Channel "+str(sipm_n))
        pl.xlabel("Pulse Height (mV)")
        if saveplot: pl.savefig(data_dir+"SPE_height_sipm_"+str(sipm_n)+".png")
        return    

    def SPE_Start_Hist(data,sipm_n):
        pl.figure()
        nbin=200
        pl.hist(data, bins=nbin,histtype='step',linewidth='1')
        #pl.xlim([0,0.3])
        pl.title("Channel "+str(sipm_n))
        pl.xlabel("Start times (us)")
        if saveplot: pl.savefig(data_dir+"SPE_start_sipm_"+str(sipm_n)+".png")
        return   


    

    # Cuts for RQ's
    cutSarea = (p_sarea[0,:,:]*tscale > -0.05)*(p_sarea[0,:,:]*tscale < 0.05)
    #cleanSarea[ch_index,:,:] = p_sarea[0,cutSarea].flatten()

#    cutArea = (p_area[p,:,:] > 0)*(p_area[p,:,:] < (0.3/tscale) )
    cutArea = p_area[0,:,:] < (0.3/tscale)
    cleanArea.append(p_area[0,cutArea].flatten())
    #cleanHeight[ch_index,:,:] = p_max_height[0,cutArea].flatten()
    #cleanStart[ch_index,:,:] = p_start[0,cutArea].flatten()

        # Make some plots
     #   SPE_Sarea_Hist(cleanSarea*tscale, p)
      #  SPE_Area_Hist(cleanArea*tscale, p)
       # SPE_Height_Hist(cleanHeight, p)
        #SPE_Start_Hist(cleanStart*tscale, p)

        # Save RQ's 

for p in chs:
    #list_rq['p_sarea_'+str(p)] = cleanSarea[p,:,:]
    list_rq['p_area_'+str(p)] = cleanArea[chs.index(p)]
    #list_rq['p_max_height_'+str(p)] = cleanHeight[p,:,:]
    #list_rq['p_start_'+str(p)] = cleanStart[p,:,:]
    if SPEMode:
        list_rq['p_noisearea_'+str(p)] = np.array(p_noisearea[0])

# Save RQ's
#rq = open(data_dir + "randomDC_spe_rq_t-%sC.npz"%temp,'wb')
rq = open(data_dir + "spe.npz",'wb')
np.savez(rq, **list_rq)
rq.close()

    
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#templist = np.array([117.6,110.0,103.1,97.3,90.2])
#templist = np.array([117.6,110.0,103.1])
templist = np.array([100.6])

tscale = (8.0/4096.0)*1000 #ns

#data_dir = "/Volumes/Samsung USB/crystalize/dark_count_all_channel_Cu_rod_%s/"
#data_dir = "/Users/qingxia/Documents/Physics/LZ/SiPM/dark_count_all_ch_calibration_%s/"
filebase = "spe_rq_"
fontsize = 14
impedance = 50 #input impedance
gainconv = pow(10,-3)*pow(10,-9)/(1.602*pow(10,-19)*impedance)
pl.rc('xtick',labelsize=fontsize)
pl.rc('ytick',labelsize=fontsize)
def gauss(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))
def linearfit(x,a,b): 
     return a*x+b
fig = pl.figure()
ax = fig.gca()
for channel in chs:
#for channel in [7]:
    gainlist = []
    gainerrlist = []
    plottemplist = []
    for temp in templist:
         if channel==0:
             #outfile = open(data_dir%str(temp)+"SiPMgain_-%sC.txt"%str(temp),"w")
             outfile = open(data_dir+'SiPMgain.txt', 'w')
             outfile.write("Channel;sphe size (mV*ns);sphe size error;Gain;GainError\n")
         else:
             #outfile = open(data_dir%str(temp)+"SiPMgain_-%sC.txt"%str(temp),"a")
             outfile = open(data_dir+'SiPMgain.txt', 'a')
         peaklist = []
         peakerrlist = []
         if channel==7 and temp ==100.6:
             #rq = np.load(data_dir%str(temp)+"randomDC_"+filebase+"t-%sC.npz"%str(temp))
             rq = np.load(data_dir+"spe.npz")
             area = rq["p_area_%d"%channel]*tscale 
             noisearea = rq["p_noisearea_%d"%channel]*tscale
             area = np.concatenate([area,noisearea[0:5000]])
         else:
             rq = np.load(data_dir+"spe.npz")
             #rq = np.load(data_dir%str(temp)+filebase+"t-%sC.npz"%str(temp))
             area = rq["p_area_%d"%channel]*tscale 
         area = area[area!=0]
#         nbins = 180
         nbins = 100
         uplim = 130
         lowlim = -20
         [data,bins] = np.histogram(area, bins=nbins, range=(lowlim,uplim))
         binCenters = np.array([0.5 * (bins[j] + bins[j+1]) for j in range(len(bins)-1)])

         #find prominent peaks in the dark count spectrum
         min_height = 300
         min_width = 1
         rel_height = 0.5 
         prominence = 50
         peaks, properties = find_peaks(data, height=min_height, width=min_width, rel_height=rel_height, prominence=prominence)
         peakmeans = binCenters[peaks]
         #set the initial gaussain fit range for the peaks
         subfig = pl.figure()
         subax = subfig.gca()
         subax.hist(area,nbins,range=(lowlim,uplim))
         for i in range(len(peakmeans)):
             lab=""
             lab2 = "final fit"
             try:
                 if len(peakmeans)==1:
                     print ("only one peak found on channel %d, T=-%sC"%(channel,str(temp)))
                     fmin = peakmeans[0]-10
                     fmax = peakmeans[0]+10
                 else:
                     if i==0:
                         fmin = peakmeans[i]-(peakmeans[i+1]-peakmeans[i])/2.
                         lab = "initial fit"
                     else:
                         fmin = (peakmeans[i]+peakmeans[i-1])/2.
                     if i==len(peakmeans)-1:
                         fmin = peakmeans[i]-(peakmeans[i]-peakmeans[i-1])/3.
                         fmax = peakmeans[i]+(peakmeans[i]-peakmeans[i-1])/2.
                     else:
                         fmax = (peakmeans[i]+peakmeans[i+1])/2.
             #preliminary guassian fit to the peaks
                 gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
                 x = binCenters[gpts]
                 y = data[gpts]
                 [p0,p0cov] = curve_fit(gauss, xdata=x, ydata=y, p0=[150,peakmeans[i],10],bounds=([0,-100,0],[100000,200,50]))
                 subax.plot(x,gauss(x, *p0), color='orange',label=lab)
             except:
                 print("can't perform intial fit to peak #%d on channel %d"%(i,channel))
                 errfig = pl.figure()
                 errax = errfig.gca()
                 errax.hist(area,nbins,range=(lowlim,uplim))
                 errax.plot(peakmeans,[400]*len(peakmeans),"o",color="orange")
                 errfig.show()
                 input("press Enter to continue")
                 continue
             #set the fit range based on the intial fit
             fmin = p0[1]-1*abs(p0[2])
             fmax = p0[1]+1*abs(p0[2])
             #final fit to the sphe peak
             gpts = np.logical_and(fmin<binCenters, fmax>binCenters)
#             gptsplot = np.logical_and((p0[1]-2*abs(p0[2]))<binCenters, (p0[1]+2*abs(p0[2]))>binCenters)
             x = binCenters[gpts]
#             xplot = binCenters[gptsplot]
             y = data[gpts]
             try:
                 [p,pcov] = curve_fit(gauss, xdata=x, ydata=y, p0=p0,bounds=([0,-100,0],[100000,200,50]))
                 perr = np.sqrt(np.diag(pcov))
                 peaklist.append(p[1])
                 peakerrlist.append(perr[1])
#                 subax.plot(peakmeans,[400]*len(peakmeans),"o",color="orange")
                 #plot pulse area histos and fitted guassians
                 if p[1]>5:
                     lab2 = lab2+" mean=%.2f\nsigma/mean=%.2f"%(p[1],p[2]/p[1])
                 subax.plot(x,gauss(x, *p), label=lab2)
             except:
                 print("can't perform final fit to peak #%d on channel %d"%(i,channel))
                 errfig = pl.figure()
                 errax = errfig.gca()
                 errax.hist(area,nbins,range=(lowlim,uplim))
                 errax.plot(peakmeans,[400]*len(peakmeans),"o",color="orange")
                 errfig.show()
                 input("press Enter to continue")
                 exit(0)
             subax.set_xlim([lowlim,uplim])
             subax.legend(loc="upper right")
             subax.set_title("Channel %d, T=-%s$^\circ$C"%(channel,temp),fontsize=fontsize)
             subax.set_xlabel("Pulse Area (mV*ns)",fontsize=fontsize)
             #pl.savefig(data_dir%str(temp)+"Channel_%d_T-%sC.png"%(channel,str(temp)))
             pl.savefig(data_dir+"Channel_{}.png".format(channel))

             if (i==0 and channel <6) or (i==1 and channel>5):
                 cali_file = open(data_dir+'spe.txt', 'a')
                 cali_file.write("ch{} {:.2f}\n".format(channel, p[1]))
                 cali_file.close()

             pl.ion()
             #subfig.show()
        #input("press Enter to continue")
         pl.close()
         peaklist = np.array(peaklist)
         peakerrlist = np.array(peakerrlist)
         peakindex = np.array(list(range(len(peaklist))))
         if len(peakindex)==0:
             print ("Error: no phe peaks found on channle %d, T=%s"%(channel,temp))
             continue
         elif len(peakindex)<2:
             print ("Warning: only %d phe peak found on channel %d, T=%s"%(len(peakindex),channel,temp))
             plottemplist.append(temp)
             gainlist.append(peaklist[0])
             gainerrlist.append(peakerrlist[0])
         else:
             popt,pcov=curve_fit(linearfit,peakindex,peaklist,p0=(0.0,0.0),sigma=peakerrlist) 
             chi2 = np.sum((popt[0]*peakindex+popt[1] - peaklist) ** 2)/len(peakindex)
             if chi2>1:
                 print ("Warning: bad linear fit on channel %d, T=-%sC"%(channel,str(temp)))
             linfit = pl.figure()
             linfitax = linfit.gca()
             linfitax.set_xlabel("peak index ",fontsize=fontsize)
             linfitax.set_ylabel("pulse area (mV*ns)",fontsize=fontsize)
         #         pl.plot(peakindex,popt[0]*peakindex+popt[1],"b",label="%d*x+%d"%(popt[0],popt[1]))
             linfitax.plot(peakindex,popt[0]*peakindex+popt[1],"b",label="sphe size = %.1f$\pm$%.1f mV*ns\n chi^2/ndf=%.4f\ngain=%d"%(popt[0],np.sqrt(np.diag(pcov))[0],chi2,popt[0]*gainconv))
             linfitax.errorbar(peakindex,peaklist,yerr=peakerrlist,fmt='o',color="b")
             linfitax.set_xticks(peakindex)
             linfitax.legend(loc="lower right")
             linfitax.set_title("Channel%d, T=-%s$^\circ$C"%(channel,temp))
             #linfit.show()
             #linfit.savefig(data_dir%str(temp)+"gain_linearfit_ch%d.png"%(channel))
             linfit.savefig(data_dir+"gain_linearfit_ch{}.png".format(channel))
             #input("press Enter to continue")
             pl.close()
             plottemplist.append(temp)
             gainlist.append(popt[0])
             gainerrlist.append(np.sqrt(np.diag(pcov))[0])
             outfile.write("%d  %f  %f  %f  %f\n"%(channel,gainlist[-1],gainerrlist[-1],gainlist[-1]*gainconv,gainerrlist[-1]*gainconv))
    #plot gain curve for the given channel
    if len(gainlist)>1:
        gainlist = np.array(gainlist)
        plottemplist = np.array(plottemplist)
        gainlist = gainlist*gainconv
        ax.errorbar((plottemplist)*(-1),gainlist,yerr=gainerrlist,fmt='o-',label="channel %d"%channel)
ax.set_xlabel("Temperature ($^\circ$C)",fontsize=fontsize)
ax.set_ylabel("SiPM Gain",fontsize=fontsize)
ax.legend(loc="upper right",prop={"size":10})
ax.set_title("Gain vs. Temperature",fontsize=fontsize)
pl.tight_layout()
pl.clf()
#fig.savefig("/Volumes/Samsung USB/crystalize/gainvstemp.png")
#fig.show()
#input("press Enter to quit")


