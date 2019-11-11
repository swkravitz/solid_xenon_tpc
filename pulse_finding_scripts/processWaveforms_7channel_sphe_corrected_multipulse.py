import sys
import csv
#import BinFileReader as bfr
import BinFileReaderNew as bfrn
#import PulseFinderSpheCorrected as pf
#import PulseFinderSpheCorrectedImproved as pf
import PulseFinderSpheCorrectedImproved_v2 as pf
#import PulseFinderSpheCorrectedImproved_v3 as pf
import PulseQuantities as pq
import SpheFileFinder as sp
import numpy as np
from ROOT import TTree, TFile
from array import array

#import matplotlib.pyplot as plt
#import matplotlib.pylab as pylab

#==================================
#Read in the binary file from the DDC10:
infilename = sys.argv[1]
outfiledir = sys.argv[2]

slash_index = infilename.rfind("/")
dot_index = infilename.rfind(".")
underscore_index = infilename.rfind("_")

outfilename = infilename[slash_index+1:dot_index] + ".reduced.root"
filenumber = infilename[underscore_index+1:dot_index]
inputdir = infilename[0:slash_index]
logfilename = "log_" + filenumber + ".txt"

inputdate_str = inputdir[inputdir.rfind("_")+1:]

#==================================

print "Reading in file: %s" % infilename
number_of_channels = 2
number_of_pmts = 1
#waveforms, number_of_events = bfr.ReadBinFile(infilename,number_of_channels)
waveforms, number_of_events = bfrn.Read_DDC10_BinWaveCap_ChSel(infilename)
# We load the data in the bin files to a numpy array.
# This numpy array is 3-dimensional: 
# D1 = channel#, 
# D2 = event number
# D3 = sample number
#
# PMT numbering and channel numbers (as stored in .bin file):
# Updated August 19, 2016 @ 13:15
# KB0060 = PMT1, LE channel 1, HE channel 4 
# KB0064 = PMT2, LE channel 2, HE channel 5
# KB0089 = PMT3, LE channel 3, HE channel 6
# Trigger on channel 0

number_of_samples = waveforms.shape[2]

print "Got %d events, each with length %d samples." % (number_of_events,number_of_samples)
#==================================
#Grab livetime from log file
#make a numpy array to put livetimes
livetimes = np.zeros(number_of_events)
deadtimes = np.zeros(number_of_events)
eventtimes = np.zeros(number_of_events,dtype=np.uint64)
line_count=1
# open log file and get livetimes
try:
    f = csv.reader(open(inputdir+"/"+logfilename))
    print "Found log file!"
    #for line in f:
    #    if line[0]=="pre ":
    #        livetimes[ int(line[1]) ] = float(line[2])
    
    #skip the header info on first 5 lines:
    for skip in range(5):
        next(f)
    
    for line in f:
        if line_count<=number_of_events:            
            eventtimes[ int(line[0])-1 ] = np.uint64(line[1]) #leaving in CPU cycles
            livetimes[ int(line[0])-1 ] = float(line[2]) / 600e6
            deadtimes[ int(line[0])-1 ] = float(line[3]) / 600e6
            line_count += 1

    #f.close()
except IOError:
    print "Couldn't find or open the log file!"
    print "Processing data anyways..."

#==================================
#Decide what sphe file to use based on date of input
#Using premade function:
print "Searching for sphe size file for date: %s"%inputdate_str
sphe_file_str = sp.GetSpheFile( inputdate_str )
print "Using sphe size file: %s"%sphe_file_str
#Get sphe sizes from text file
#store in numpy array
sphe_size_array = np.zeros(2)
line_counter=1
# open file
try:
    #f_sphe = csv.reader(open("/net/cms24/cms24r0/sjh/SamplerCode_git/data_processing_scripts/pmt_sphe_files/sphe_sizes_20190103_2000V.txt"))
    f_sphe = csv.reader(open("/net/cms24/cms24r0/sjh/SamplerCode_git/data_processing_scripts/pmt_sphe_files/"+sphe_file_str))
    
    for line in f_sphe:
        sphe_size_array[ line_counter-1 ] = float(line[0])
        line_counter += 1

    #f_sphe.close()
except IOError:
    print "Couldn't find or open the sphe size file! Exiting..."
    exit(1)

#==================================
#subtract baseline from first 150 samples
s = waveforms.shape
waveforms_bls = np.zeros(s)
for channel in range(number_of_channels):
    for evt in range(number_of_events):
        waveforms_bls[channel,evt,:] = waveforms[channel,evt,:] - np.mean(waveforms[channel,evt,150:180])
        #waveforms_bls[channel,evt,0:150] = np.mean(waveforms_bls[channel,evt,150:180])

#Take each channel, give them names and apply the sphe sizes:
#Units of each waveform height is now in phe/sample
pmt1_sphe_size_le = float(sphe_size_array[0])
#pmt2_sphe_size_le = float(sphe_size_array[1])
#pmt3_sphe_size_le = float(sphe_size_array[2])

#pmt1_sphe_size_he = float(sphe_size_array[1])
pmt1_sphe_size_he = pmt1_sphe_size_le/40.0
#pmt2_sphe_size_he = pmt2_sphe_size_le/40.0
#pmt3_sphe_size_he = pmt3_sphe_size_le/40.0

print "Using the following sphe sizes:"
print "PMT 1, LE: %0.2f mV*ns" % pmt1_sphe_size_le
print "PMT 1, HE: %0.2f mV*ns" % pmt1_sphe_size_he

waveforms_bls_pmt1_LE = waveforms_bls[0,:,:]*0.122*(10/pmt1_sphe_size_le)
#waveforms_bls_pmt1_LE = waveforms_bls[0,:,:]*0.122

waveforms_bls_pmt1_HE = waveforms_bls[1,:,:]*0.122*(10/pmt1_sphe_size_he)

#Make the summed pulse for LE and HE
waveforms_bls_sum_LE = waveforms_bls_pmt1_LE
waveforms_bls_sum_HE = waveforms_bls_pmt1_HE

#===================================================
# Run pulse finder on waveforms
# Extracting pulse starting, ending and max samples
# and pulse area
# Save results to ROOT file

events_to_run = number_of_events
number_of_pulses_per_event = 3

start = np.zeros(number_of_pulses_per_event,dtype=np.int)
end = np.zeros(number_of_pulses_per_event,dtype=np.int)
found = np.zeros(number_of_pulses_per_event,dtype=np.int)

start_he = np.zeros(number_of_pulses_per_event,dtype=np.int)
end_he = np.zeros(number_of_pulses_per_event,dtype=np.int)
found_he = np.zeros(number_of_pulses_per_event,dtype=np.int)

#variables to hold data for output branch
event_length = array( 'i',[number_of_samples]) #event length in samples
event_livetime = array( 'f',[0]) #livetime of event in seconds
event_deadtime = array( 'f',[0]) #deadtime of event in seconds
event_time = array( 'L',[0]) #event time in cpu cycles
n_pulses_per_event = array( 'i',[number_of_pulses_per_event])

# Low energy channel variables for each PMT:
 # PMT1:
#p_start_pmt1_le = array( 'i',np.zeros(number_of_pulses_per_event))                                # pulse start in samples
#p_end_pmt1_le =  array( 'i',np.zeros(number_of_pulses_per_event))                                 # pulse end in samples
p_max_pmt1_le =  array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))                                 # pulse max in samples
area_pmt1_le =  array( 'f',np.zeros(number_of_pulses_per_event))                                 # pulse area in mV*ns
max_height_pmt1_le =  array( 'f',np.zeros(number_of_pulses_per_event))                           # max pulse height in mV
min_height_pmt1_le =  array( 'f',np.zeros(number_of_pulses_per_event))                           # min pulse height in mV
waveform_array_pmt1_le = array( 'f', waveforms_bls_pmt1_LE[0,:]) # event waveform

# High energy channel variables for each PMT:
 # PMT1:
#p_start_pmt1_he = array( 'i',np.zeros(number_of_pulses_per_event))
#p_end_pmt1_he =  array( 'i',np.zeros(number_of_pulses_per_event))
p_max_pmt1_he =  array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
area_pmt1_he =  array( 'f',np.zeros(number_of_pulses_per_event))
max_height_pmt1_he =  array( 'f',np.zeros(number_of_pulses_per_event))
min_height_pmt1_he =  array( 'f',np.zeros(number_of_pulses_per_event))
waveform_array_pmt1_he = array( 'f', waveforms_bls_pmt1_HE[0,:])

#Sum pulse qunatities:
p_found = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))       # pulse finder flag
p_start_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))       # pulse start in samples
p_end_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))         # pulse end in samples
p_max_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))         # pulse max in samples

area_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))                       # pulse area
max_height_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))                 # max pulse height in mV
min_height_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))                 # min pulse height in mV

afs_2l_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))            # samples of fractional areas
afs_2r_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_1_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_25_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_50_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_75_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_99_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))

hfs_10l_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))            # samples of fractional heights
hfs_50l_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
hfs_10r_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))            
hfs_50r_sum_le = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))

ta_10l_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))            # tail area
ta_15l_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))
ta_20l_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))
ta_25l_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))
ta_30l_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))
ta_35l_sum_le = array( 'f',np.zeros(number_of_pulses_per_event))

mean_time_le = array( 'f',np.zeros(number_of_pulses_per_event))                      # weighted mean time of pulse in samples
rms_time_le = array( 'f',np.zeros(number_of_pulses_per_event))                       # rms about mean time in samples
waveform_array_sum_le = array( 'f', waveforms_bls_sum_LE[0,:]) # event waveform

 # High energy
p_found_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))       # pulse finder flag
p_start_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))                   # pulse start in samples
p_end_sum_he =  array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))                    # pulse end in samples
p_max_sum_he =  array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))        # pulse max in samples
area_sum_he =  array( 'f',np.zeros(number_of_pulses_per_event))                      # pulse area in mV*ns
max_height_sum_he =  array( 'f',np.zeros(number_of_pulses_per_event))                # max pulse height in mV
min_height_sum_he =  array( 'f',np.zeros(number_of_pulses_per_event))                # min pulse height in mV

afs_2l_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))            # samples of fractional areas
afs_2r_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_1_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_25_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_50_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_75_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
afs_99_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))

hfs_10l_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))            # samples of fractional heights
hfs_50l_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))
hfs_10r_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))            
hfs_50r_sum_he = array( 'i',np.zeros(number_of_pulses_per_event,dtype=np.int))

mean_time_he = array( 'f',np.zeros(number_of_pulses_per_event))                      # weighted mean time of pulse in samples
rms_time_he = array( 'f',np.zeros(number_of_pulses_per_event))                       # rms about mean time in samples
waveform_array_sum_he = array( 'f', waveforms_bls_sum_HE[0,:]) # event waveform

# Set up the output ROOT file:
outfile = TFile( outfiledir+outfilename, "RECREATE" )

#output tree with branches:
outtree = TTree("events","Tree Containing Screener Pulse Quantities")
outtree.Branch("event_length_samples", event_length, "event_length_samples/I")
outtree.Branch("num_pulses_per_event", n_pulses_per_event, "num_pulses_per_event/I")
outtree.Branch("event_livetime_seconds", event_livetime, "event_livetime_seconds/F")
outtree.Branch("event_deadtime_seconds", event_deadtime, "event_deadtime_seconds/F")
outtree.Branch("event_time_cycles", event_time, "event_time_cycles/l")
#Low Energy:
#outtree.Branch("pmt1_pulse_start_le", p_start_pmt1_le, "pmt1_pulse_start_le[num_pulses_per_event]/I")
#outtree.Branch("pmt1_pulse_end_le", p_end_pmt1_le, "pmt1_pulse_end_le[num_pulses_per_event]/I")
outtree.Branch("pmt1_pulse_max_le", p_max_pmt1_le, "pmt1_pulse_max_le[num_pulses_per_event]/I")
outtree.Branch("pmt1_pulse_area_le", area_pmt1_le, "pmt1_pulse_area_le[num_pulses_per_event]/F")
outtree.Branch("pmt1_pulse_max_height_le", max_height_pmt1_le, "pmt1_pulse_max_height_le[num_pulses_per_event]/F")
outtree.Branch("pmt1_pulse_min_height_le", min_height_pmt1_le, "pmt1_pulse_min_height_le[num_pulses_per_event]/F")
outtree.Branch("pmt1_waveform_le", waveform_array_pmt1_le, "pmt1_waveform_le[event_length_samples]/F")

#High Energy:
#outtree.Branch("pmt1_pulse_start_he", p_start_pmt1_he, "pmt1_pulse_start_he[num_pulses_per_event]/I")
#outtree.Branch("pmt1_pulse_end_he", p_end_pmt1_he, "pmt1_pulse_end_he[num_pulses_per_event]/I")
outtree.Branch("pmt1_pulse_max_he", p_max_pmt1_he, "pmt1_pulse_max_he[num_pulses_per_event]/I")
outtree.Branch("pmt1_pulse_area_he", area_pmt1_he, "pmt1_pulse_area_he[num_pulses_per_event]/F")
outtree.Branch("pmt1_pulse_max_height_he", max_height_pmt1_he, "pmt1_pulse_max_height_he[num_pulses_per_event]/F")
outtree.Branch("pmt1_pulse_min_height_he", min_height_pmt1_he, "pmt1_pulse_min_height_he[num_pulses_per_event]/F")
outtree.Branch("pmt1_waveform_he", waveform_array_pmt1_he, "pmt1_waveform_he[event_length_samples]/F")

#Sum Pulses:
# Low Energy:
outtree.Branch("sum_pulse_found", p_found, "sum_pulse_found[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_start_le", p_start_sum_le, "sum_pulse_start_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_end_le", p_end_sum_le, "sum_pulse_end_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_max_le", p_max_sum_le, "sum_pulse_max_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_area_le", area_sum_le, "sum_pulse_area_le[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_max_height_le", max_height_sum_le, "sum_pulse_max_height_le[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_min_height_le", min_height_sum_le, "sum_pulse_min_height_le[num_pulses_per_event]/F")

outtree.Branch("sum_pulse_afs_2l_le", afs_2l_sum_le, "sum_pulse_afs_2l_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_2r_le", afs_2r_sum_le, "sum_pulse_afs_2r_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_1_le", afs_1_sum_le, "sum_pulse_afs_1_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_25_le", afs_25_sum_le, "sum_pulse_afs_25_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_50_le", afs_50_sum_le, "sum_pulse_afs_50_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_75_le", afs_75_sum_le, "sum_pulse_afs_75_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_99_le", afs_99_sum_le, "sum_pulse_afs_99_le[num_pulses_per_event]/I")

outtree.Branch("sum_pulse_hfs_10l_le", hfs_10l_sum_le, "sum_pulse_hfs_10l_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_hfs_50l_le", hfs_50l_sum_le, "sum_pulse_hfs_50l_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_hfs_10r_le", hfs_10r_sum_le, "sum_pulse_hfs_10r_le[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_hfs_50r_le", hfs_50r_sum_le, "sum_pulse_hfs_50r_le[num_pulses_per_event]/I")

outtree.Branch("sum_pulse_ta_10l_le", ta_10l_sum_le, "sum_pulse_ta_10l_le[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_ta_15l_le", ta_15l_sum_le, "sum_pulse_ta_15l_le[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_ta_20l_le", ta_20l_sum_le, "sum_pulse_ta_20l_le[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_ta_25l_le", ta_25l_sum_le, "sum_pulse_ta_25l_le[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_ta_30l_le", ta_30l_sum_le, "sum_pulse_ta_30l_le[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_ta_35l_le", ta_35l_sum_le, "sum_pulse_ta_35l_le[num_pulses_per_event]/F")

outtree.Branch("sum_pulse_mean_samples_le", mean_time_le, "sum_pulse_mean_samples_le[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_rms_samples_le", rms_time_le, "sum_pulse_rms_samples_le[num_pulses_per_event]/F")
outtree.Branch("sum_waveform_le", waveform_array_sum_le, "sum_waveform_le[event_length_samples]/F")

# High Energy:
outtree.Branch("sum_pulse_found_he", p_found_he, "sum_pulse_found_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_start_he", p_start_sum_he, "sum_pulse_start_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_end_he", p_end_sum_he, "sum_pulse_end_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_max_he", p_max_sum_he, "sum_pulse_max_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_area_he", area_sum_he, "sum_pulse_area_he[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_max_height_he", max_height_sum_he, "sum_pulse_max_height_he[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_min_height_he", min_height_sum_he, "sum_pulse_min_height_he[num_pulses_per_event]/F")

outtree.Branch("sum_pulse_afs_2l_he", afs_2l_sum_he, "sum_pulse_afs_2l_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_2r_he", afs_2r_sum_he, "sum_pulse_afs_2r_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_1_he", afs_1_sum_he, "sum_pulse_afs_1_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_25_he", afs_25_sum_he, "sum_pulse_afs_25_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_50_he", afs_50_sum_he, "sum_pulse_afs_50_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_75_he", afs_75_sum_he, "sum_pulse_afs_75_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_afs_99_he", afs_99_sum_he, "sum_pulse_afs_99_he[num_pulses_per_event]/I")

outtree.Branch("sum_pulse_hfs_10l_he", hfs_10l_sum_he, "sum_pulse_hfs_10l_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_hfs_50l_he", hfs_50l_sum_he, "sum_pulse_hfs_50l_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_hfs_10r_he", hfs_10r_sum_he, "sum_pulse_hfs_10r_he[num_pulses_per_event]/I")
outtree.Branch("sum_pulse_hfs_50r_he", hfs_50r_sum_he, "sum_pulse_hfs_50r_he[num_pulses_per_event]/I")

outtree.Branch("sum_pulse_mean_samples_he", mean_time_he, "sum_pulse_mean_samples_he[num_pulses_per_event]/F")
outtree.Branch("sum_pulse_rms_samples_he", rms_time_he, "sum_pulse_rms_samples_he[num_pulses_per_event]/F")
outtree.Branch("sum_waveform_he", waveform_array_sum_he, "sum_waveform_he[event_length_samples]/F")


#Run the pulse finder on each evt
print "Running pulse finder on %d events" % events_to_run

#make copy of waveform arrays:
waveforms_bls_sum_LE_cpy = waveforms_bls_sum_LE.copy()
waveforms_bls_sum_HE_cpy = waveforms_bls_sum_HE.copy()

for evt in range(events_to_run):

    # first save basic event info:
    event_length[0] = number_of_samples
    n_pulses_per_event[0] = number_of_pulses_per_event
    event_livetime[0] = livetimes[evt]
    event_deadtime[0] = deadtimes[evt]
    event_time[0] = eventtimes[evt]
    
    # waveforms:
    waveform_array_pmt1_le[:] = array( 'f', waveforms_bls_pmt1_LE[evt,:])
    #waveform_array_pmt2_le[:] = array( 'f', waveforms_bls_pmt2_LE[evt,:])
    #waveform_array_pmt3_le[:] = array( 'f', waveforms_bls_pmt3_LE[evt,:])
    waveform_array_sum_le[:] = array( 'f', waveforms_bls_sum_LE[evt,:])
    #waveform_array_trig[:] = array( 'f', waveforms_bls_trig[evt,:])
    waveform_array_pmt1_he[:] = array( 'f', waveforms_bls_pmt1_HE[evt,:])
    #waveform_array_pmt2_he[:] = array( 'f', waveforms_bls_pmt2_HE[evt,:])
    #waveform_array_pmt3_he[:] = array( 'f', waveforms_bls_pmt3_HE[evt,:])
    waveform_array_sum_he[:] = array( 'f', waveforms_bls_sum_HE[evt,:])
    
    # trigger pulse info (trigger pulse waveform in ADC counts):
    #p_start_trig[0],p_end_trig[0],p_found_trig[0] = pf.findaPulse(evt,waveforms_bls_trig)
    #if p_found_trig[0] == 1:
    #    area_trig[0] = pq.GetPulseArea( p_start_trig[0],p_end_trig[0]+1,waveforms_bls_trig[evt,:] )
    #    p_max_trig[0] = pq.GetPulseMaxSample( p_start_trig[0],p_end_trig[0]+1,waveforms_bls_trig[evt,:] )
    #    max_height_trig[0] = pq.GetPulseMaxHeight( p_start_trig[0],p_end_trig[0]+1,waveforms_bls_trig[evt,:] ) * 0.122
    #    min_height_trig[0] = pq.GetPulseMinHeight( p_start_trig[0],p_end_trig[0]+1,waveforms_bls_trig[evt,:] ) * 0.122
    
    # Now loop over number of pulses per event and save pulse quantities along the way
    # for now, pulse quantities all derived using sum pulse start and end times
    for p in range(number_of_pulses_per_event):
        
        start[p],end[p],found[p] = pf.findaPulse(evt,waveforms_bls_sum_LE_cpy)
        #start_he[p],end_he[p],found_he[p] = pf.findaPulse(evt,waveforms_bls_sum_HE_cpy)
        
        # Clear the waveform array of this pulse
        if found[p] == 1:
            waveforms_bls_sum_LE_cpy[evt,:] = pq.ClearWaveform( start[p], end[p]+1, waveforms_bls_sum_LE_cpy[evt,:] )
        #if found_he[p] == 1:
        #    waveforms_bls_sum_HE_cpy[evt,:] = pq.ClearWaveform( start_he[p], end_he[p]+1, waveforms_bls_sum_HE_cpy[evt,:] )
    
    # Sort pulses by start times, not areas
    startinds = np.argsort(start)
    pp = int(0)
    for p_index in startinds:
    #for p in range(number_of_pulses_per_event):
        p_found[pp] = found[p_index]
        p_start_sum_le[pp] = start[p_index]
        p_end_sum_le[pp] = end[p_index]
        
        if p_found[pp] == 1:
            # -------- Low Energy Pulses --------
            # PMT 1
            area_pmt1_le[pp] = pq.GetPulseArea( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_pmt1_LE[evt,:] )
            p_max_pmt1_le[pp] = pq.GetPulseMaxSample( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_pmt1_LE[evt,:] )
            max_height_pmt1_le[pp] = pq.GetPulseMaxHeight( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_pmt1_LE[evt,:] )
            min_height_pmt1_le[pp] = pq.GetPulseMinHeight( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_pmt1_LE[evt,:] )
            
            # SUM
            area_sum_le[pp] = pq.GetPulseArea( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_LE[evt,:] )
            p_max_sum_le[pp] = pq.GetPulseMaxSample( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_LE[evt,:] )
            max_height_sum_le[pp] = pq.GetPulseMaxHeight( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_LE[evt,:] )
            min_height_sum_le[pp] = pq.GetPulseMinHeight( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_LE[evt,:] )
            
            (afs_2l_sum_le[pp], afs_2r_sum_le[pp], afs_1_sum_le[pp], afs_25_sum_le[pp], afs_50_sum_le[pp], afs_75_sum_le[pp], afs_99_sum_le[pp]) = pq.GetAreaFractionSamples( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_LE[evt,:] )
            
            hfs_10l_sum_le[pp], hfs_50l_sum_le[pp], hfs_10r_sum_le[pp], hfs_50r_sum_le[pp] = pq.GetHeightFractionSamples( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_LE[evt,:] )
            #   Using height fractions for mean and RMS
            mean_time_le[pp], rms_time_le[pp] = pq.GetPulseMeanAndRMS( hfs_10l_sum_le[pp], hfs_10r_sum_le[pp]+1, waveforms_bls_sum_LE[evt,:] )
            
            ta_10l_sum_le[pp], ta_15l_sum_le[pp], ta_20l_sum_le[pp], ta_25l_sum_le[pp], ta_30l_sum_le[pp], ta_35l_sum_le[pp] = pq.GetTailArea( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_LE[evt,:] )
            # ------------------------------------
            
            # -------- High Energy Pulses --------
            # PMT 1
            area_pmt1_he[pp] = pq.GetPulseArea( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_pmt1_HE[evt,:] )
            p_max_pmt1_he[pp] = pq.GetPulseMaxSample( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_pmt1_HE[evt,:] )
            max_height_pmt1_he[pp] = pq.GetPulseMaxHeight( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_pmt1_HE[evt,:] )
            min_height_pmt1_he[pp] = pq.GetPulseMinHeight( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_pmt1_HE[evt,:] )
            
            # SUM
            area_sum_he[pp] = pq.GetPulseArea( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_HE[evt,:] )
            p_max_sum_he[pp] = pq.GetPulseMaxSample( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_HE[evt,:] )
            max_height_sum_he[pp] = pq.GetPulseMaxHeight( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_HE[evt,:] )
            min_height_sum_he[pp] = pq.GetPulseMinHeight( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_HE[evt,:] )
            
            (afs_2l_sum_he[pp], afs_2r_sum_he[pp], afs_1_sum_he[pp], afs_25_sum_he[pp], afs_50_sum_he[pp], afs_75_sum_he[pp], afs_99_sum_he[pp]) = pq.GetAreaFractionSamples( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_HE[evt,:] )
            
            hfs_10l_sum_he[pp], hfs_50l_sum_he[pp], hfs_10r_sum_he[pp], hfs_50r_sum_he[pp] = pq.GetHeightFractionSamples( p_start_sum_le[pp], p_end_sum_le[pp]+1, waveforms_bls_sum_HE[evt,:] )
            #   Using height fractions for mean and RMS
            mean_time_he[pp], rms_time_he[pp] = pq.GetPulseMeanAndRMS( hfs_10l_sum_le[pp], hfs_10r_sum_le[pp]+1, waveforms_bls_sum_HE[evt,:] )
            # ------------------------------------
            
        pp = pp + 1
        # end second (sorted) pulse loop
    
    # Sort pulses by start times, not areas
    # use this if running pulse finder individually on HE pulses
#    startinds_he = np.argsort(start_he)
#    pp = int(0)
#    for p_index_he in startinds_he:
#        p_found_he[pp] = found_he[p_index_he]
#        p_start_sum_he[pp] = start_he[p_index_he]
#        p_end_sum_he[pp] = end_he[p_index_he]
        
#        if p_found_he[pp] == 1:
            # -------- High Energy Pulses --------
            # PMT 1
#            area_pmt1_he[pp] = pq.GetPulseArea( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_pmt1_HE[evt,:] )
#            p_max_pmt1_he[pp] = pq.GetPulseMaxSample( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_pmt1_HE[evt,:] )
#            max_height_pmt1_he[pp] = pq.GetPulseMaxHeight( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_pmt1_HE[evt,:] )
#            min_height_pmt1_he[pp] = pq.GetPulseMinHeight( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_pmt1_HE[evt,:] )
            
            # SUM
#            area_sum_he[pp] = pq.GetPulseArea( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_sum_HE[evt,:] )
#            p_max_sum_he[pp] = pq.GetPulseMaxSample( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_sum_HE[evt,:] )
#            max_height_sum_he[pp] = pq.GetPulseMaxHeight( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_sum_HE[evt,:] )
#            min_height_sum_he[pp] = pq.GetPulseMinHeight( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_sum_HE[evt,:] )
            
#            (afs_2l_sum_he[pp], afs_2r_sum_he[pp], afs_1_sum_he[pp], afs_25_sum_he[pp], afs_50_sum_he[pp], afs_75_sum_he[pp], afs_99_sum_he[pp]) = pq.GetAreaFractionSamples( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_sum_HE[evt,:] )
            
#            hfs_10l_sum_he[pp], hfs_50l_sum_he[pp], hfs_10r_sum_he[pp], hfs_50r_sum_he[pp] = pq.GetHeightFractionSamples( p_start_sum_he[pp], p_end_sum_he[pp]+1, waveforms_bls_sum_HE[evt,:] )
            #   Using height fractions for mean and RMS
#            mean_time_he[pp], rms_time_he[pp] = pq.GetPulseMeanAndRMS( hfs_10l_sum_he[pp], hfs_10r_sum_he[pp]+1, waveforms_bls_sum_HE[evt,:] )
            # ------------------------------------
        
#        pp = pp + 1
        # end second (sorted) pulse loop, high energy
    
    
    #Fill output tree for this event
    outtree.Fill()
    
    if evt % 500 == 0:
        print "Done w/ event %d" % evt

# end of event loop

print "Writing to file..."
outfile.Write()
print "Closing file..."
outfile.Close()
print "=============== Done ==============="
print " "
