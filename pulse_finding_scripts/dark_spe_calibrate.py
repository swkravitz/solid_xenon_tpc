import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
import time
import sys
from scipy import stats

import PulseFinderScipy as pf
import PulseQuantities as pq
import PulseClassification as pc

import PulseFinderSPE as pfSPE

bias = 57
data_dir = "/Volumes/Samsung USB/cold_dark_b%dv/"%bias
#data_dir = "/Users/qingxia/Documents/Physics/LZ/SiPM/"
#data_dir = "C:/Users/ryanm/Documents/Research/Data/sipm_test_210319/cold_dark_b10v_noise/"

SPEMode = True 
LED = False # LED mode, expects signal at ~2us
plotyn = True # Waveform plotting
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
event_window = 4.  # in us
wsize = int(500 * event_window)  # samples per waveform # 12500 for 25 us
vscale = (500.0/16384.0) # vertical scale, = 0.031 mV/ADCC for 0.5 Vpp dynamic range
tscale = (8.0/4096.0)     # = 0.002 µs/sample, time scale

# Set range to look for pulses
if LED:
    left_bound = int(2.0/tscale)
    right_bound = int(2.3/tscale)
else:
    left_bound = 0
    right_bound = wsize

n_sipms = 16


# ==================================================================

#load in raw data

t_start = time.time()

block_size = 5000
n_block = 28
max_evts = n_block*block_size#5000  # 25000 # -1 means read in all entries; 25000 is roughly the max allowed in memory on the DAQ computer
max_pts = -1  # do not change
if max_evts > 0:
    max_pts = wsize * max_evts
load_dtype = "int16"


# RQ's for SPE analysis
max_pulses = 6
p_start = np.zeros((n_sipms, max_evts, max_pulses), dtype=np.int)
p_end   = np.zeros((n_sipms, max_evts, max_pulses), dtype=np.int)

# p_sarea instead of p_int
p_sarea = np.zeros((n_sipms, max_evts, max_pulses))
p_area = np.zeros((n_sipms, max_evts, max_pulses))
p_max_height = np.zeros((n_sipms, max_evts, max_pulses))
p_width = np.zeros((n_sipms, max_evts, max_pulses))

n_pulses = np.zeros((n_sipms, max_evts), dtype=np.int)

e_fft = np.zeros(wsize, dtype=np.complex128)


inn=""

# Loop over blocks
for j in range(n_block):
    ch_data = []
    for ch_ind in range(n_sipms):
        try:
            ch_data.append(np.fromfile(data_dir + "wave"+str(ch_ind)+".dat", dtype=load_dtype, offset = block_size*wsize*j, count=wsize*block_size))
        except:
            print ("no data from ch%d"%ch_ind)
            ch_data.append(np.zeros(10000000))

    t_end_load = time.time()
    print("Time to load files: ", t_end_load-t_start)

    # scale waveforms to get units of mV/sample
    # then for each channel ensure we 
    # have an integer number of events
    array_dtype = "float32" # using float32 reduces memory for big files, otherwise implicitly converts to float64

    # matrix of all channels including the sum waveform
    v_matrix_all_ch = []
    for ch_ind in range(n_sipms):
        V = vscale * ch_data[ch_ind].astype(array_dtype)
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
        if i%500==0: print("Event #",i)
        
        # Loop over channels, slowest part of the code
        # Have to do a loop, pulse finder does not like v_bls_matrix_all_ch[:,i,:] 
        #for k in range(n_sipms):
        for k in range(12,13):
        
            # Do a simple integral over desired range
            # If you change the range, make sure to change the plot x_label
            l_area = int(2.06/tscale) #2.06
            r_area = int(2.30/tscale) #2.18
            p_sarea[k,i,0] = pfSPE.simpleSumArea(v_bls_matrix_all_ch[k,i-j*block_size,:],[l_area,r_area])

            # Unfinished FFT calculator. Not sure we need this?
            e_fft += np.abs(np.fft.fft(v_bls_matrix_all_ch[k,i-j*block_size,:] ))

            # Do a proper pulse finder
            if LED:
                start_times, end_times, peaks, data_conv = pfSPE.findLEDSPEs( v_bls_matrix_all_ch[k, i-j*block_size, left_bound:right_bound] )
                start_times += left_bound
                end_times +=left_bound

            # Preliminary baseline should already be done
            else:
                start_times, end_times, peaks, baselines, data_conv  = pfSPE.findDarkSPEs(v_bls_matrix_all_ch[k, i-j*block_size, :])
                
            # Calculate RQ's from pulse finder
            for n in range(len(start_times)):
                if n > max_pulses - 1: break
                if start_times[n] != 0:
                    p_area[k,i,n] = pq.GetPulseArea(start_times[n],end_times[n],v_bls_matrix_all_ch[k, i-j*block_size, :] - baselines[n])
                    p_max_height[k,i,n] = pq.GetPulseMaxHeight(start_times[n],end_times[n],v_bls_matrix_all_ch[k, i-j*block_size, :]- baselines[n])
                    p_width[k,i,n] = start_times[n] - end_times[n]
                    p_start[k,i,n] = start_times[n]
                    p_end[k,i,n] = end_times[n]
            
            n_pulses[k,i] += len(start_times)

        
            # Plotter
            areacut = p_area[12,i,0] > 0
            #areacut = (p_start[12,i,0] < 0.5/tscale)*p_area[12,i,0] > 0

            # horizontal lines: @ zero, baseline, height
            
            if not inn == 'q' and plotyn and areacut and k==12 and len(start_times)>1:
                # Plot something
                fig = pl.figure(1,figsize=(10, 7))
                pl.grid(b=True,which='major',color='lightgray',linestyle='--')
                pl.plot(t_matrix[j,:], v_bls_matrix_all_ch[k,i-j*block_size,:], color='blue')
                pl.plot(t_matrix[j,:], data_conv, color='red')
                for pulse in range(len(start_times)):
                    pl.axvspan(start_times[pulse] * tscale, end_times[pulse] * tscale, alpha=0.25, color='green')
                    pl.text(end_times[pulse]*tscale,p_max_height[12,i,pulse]*1.1,"Pulse area = {:0.3f} mV*us".format(p_area[12,i,pulse]*tscale) )
                    print ("area:",p_area[12,i,pulse]*tscale,"start:",start_times[pulse],"end:",end_times[pulse])
                #pl.axvspan((start_times)*tscale, (end_times)*tscale, alpha=0.25, color='green')
                #pl.text(end_times*tscale, p_max_height[12,i,0]*1.1,"Pulse area = {:0.3f} mV*us".format(p_area[12,i,0]*tscale) )
                
                pl.hlines(0.06,0,4,color='orange',label='Height treshhold = 0.1')
                pl.hlines(0,0,4,color="black")

                pl.ylim([-0.5,1])
                pl.xlim([0,4])
                pl.title("Channel "+str(k)+", Event "+str(i) )
                pl.xlabel('Time (us)')
                pl.ylabel('mV')
                pl.draw()
                pl.ion()
                pl.show()
#                pl.show(block=0)
                inn = input("Press enter to continue, q to stop plotting, evt # to skip to # (forward only)")
                pl.close()
#                fig.clf()


            
# End of pulse finding and plotting event loop

n_events = i
print("total number of events processed:", n_events)


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


list_rq = {}

#for p in range(n_sipms):
for p in range(12,13):

    # Cuts for RQ's
    cutSarea = (p_sarea[p,:,:]*tscale > -0.05)*(p_sarea[p,:,:]*tscale < 0.05)
    cleanSarea = p_sarea[p,cutSarea].flatten()

    cutArea = (p_area[p,:,:] > 0)*(p_area[p,:,:] < (0.3/tscale) )
    cleanArea = p_area[p,cutArea].flatten()
    cleanHeight = p_max_height[p,cutArea].flatten()
    cleanStart = p_start[p,cutArea].flatten()

    # Make some plots
    SPE_Sarea_Hist(cleanSarea*tscale, p)
    SPE_Area_Hist(cleanArea*tscale, p)
    SPE_Height_Hist(cleanHeight, p)
    SPE_Start_Hist(cleanStart*tscale, p)

    # Save RQ's 
    list_rq['p_sarea_'+str(p)] = cleanSarea
    list_rq['p_area_'+str(p)] = cleanArea
    list_rq['p_max_height_'+str(p)] = cleanHeight
    list_rq['p_start_'+str(p)] = cleanStart

# Save RQ's
rq = open(data_dir + "spe_rq_b%dV.npz"%bias,'wb')
np.savez(rq, **list_rq)
rq.close()

pl.show()
