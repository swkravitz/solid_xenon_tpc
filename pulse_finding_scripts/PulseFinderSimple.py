import numpy as np
import time


def wfm_convolve(data, window, avg=False):
    # Assumes data is already baseline-subtracted
    weights = np.repeat(1.0, window)
    if avg: weights /= window  # do avg instead of sum
    return np.convolve(data, weights, 'same')


def pulse_finder_area(data, t_min_search, t_max_search, window):
    # Assumes data is already baseline-subtracted
    if t_max_search < t_min_search + 1:
        return (-1, -1)

    data_conv = wfm_convolve(data, window)
    # Search only w/in search range, offset so that max_ind is the start of the window rather than the center
    max_ind = np.argmax(data_conv[int(t_min_search + window / 2):int(t_max_search + window / 2)]) + int(t_min_search)
    return (max_ind, data_conv[max_ind + int(window / 2)])


#function to find start and end of a pulse in event
#inputs: event number and baseline subtracted waveform numpy array
#ouputs: sample number for start, end, and max of pulse

def findaPulse( evt, waveforms_bls ):
    
    numSamples = len(waveforms_bls[0,:])
    boxAreaRolling = np.zeros(numSamples)
    tmpBoxArea = 0
    areaStepSize = 5 # higher can help improve speed of initial search
    
    boxWidth = 50 # in samples
    nLookAhead = 1
    nLookBefore = 1
    
    noiseThreshold = 0.8
    
    quietCut = 3
    
    width_cut = 8000
    
    #min_area = 1 # box area must be at least this big
    
    pulse_start = 99999
    pulse_end = 99999
    foundPulse = int(0) # flag indicating a pulse was found
    
    #t0 = time.time()
    #first loop through event and find box areas
    #for t in range(0,numSamples-boxWidth-1,areaStepSize):
    #    tmpBoxArea = np.sum(waveforms_bls[evt,t:t+boxWidth])
    #    boxAreaRolling[t] = tmpBoxArea
        
    #find max area and the beginning of box that encloses that area
    #max_boxArea_value = np.max(boxAreaRolling)
    #max_boxArea_sample = np.argmax(boxAreaRolling)
    #print('max area = {:0.1f} phd located at {:d}'.format(max_boxArea_value,max_boxArea_sample))
    # find waveform max value and location of max value inside box
    #max_pulse_height = np.max(waveforms_bls[evt,max_boxArea_sample:max_boxArea_sample+boxWidth])
    #max_pulse_sample = max_boxArea_sample + np.argmax(waveforms_bls[evt,max_boxArea_sample:max_boxArea_sample+boxWidth])
    #t1 = time.time()
    #print(t1-t0)
    
    # alternate way:
    t0 = time.time()
    boxAreaRolling = wfm_convolve( waveforms_bls[evt,:] , boxWidth )
    # find max area and the beginning of box that encloses that area
    max_boxArea_value = np.max(boxAreaRolling)
    max_boxArea_sample = np.argmax(boxAreaRolling)
    #print('max area = {:0.1f} phd located at {:d}'.format(max_boxArea_value,max_boxArea_sample))
    # find waveform max value and location of max value inside box
    #max_pulse_sample = max_boxArea_sample
    max_pulse_sample = (max_boxArea_sample-int(boxWidth/2)) + np.argmax(waveforms_bls[evt,(max_boxArea_sample-int(boxWidth/2)):(max_boxArea_sample+int(boxWidth/2))])
    max_pulse_height = waveforms_bls[evt,max_pulse_sample]
    
    #t1 = time.time()
    #print(t1-t0)
    
    
    # FINDING PULSE START
    #start at max value of pulse and step backward to find point where pulse avg goes
    #below threshold for at least quietCut to find beginning of pulse
    quietSamples = 0
    for i in range(max_pulse_sample-1):
        avg = np.mean( waveforms_bls[evt,max_pulse_sample-(i+1)-nLookBefore:max_pulse_sample-(i+1)+nLookAhead] )
        if avg < noiseThreshold:
            quietSamples = quietSamples + 1
        else:
            quietSamples = 0
        if quietSamples > quietCut or (quietSamples > 0 and i == max_pulse_sample-nLookBefore-1): #start of pulse
            pulse_start = max_pulse_sample - i + quietSamples - 2
            break
            if pulse_start < 0:
                pulse_start = 0
    #print("  First Found Pulse Start: ",pulse_start)
    
    # FINDING PULSE END
    #now start at max value of pulse and step forword to find point where pulse avg goes
    #below threshold for at least quietCut to find end of pulse
    quietSamples = 0
    for i in range(numSamples-max_pulse_sample-1):
        avg = np.mean( waveforms_bls[evt,max_pulse_sample+(i+1)-nLookBefore:max_pulse_sample+(i+1)+nLookAhead] )
        if avg < noiseThreshold:
            quietSamples = quietSamples + 1
        else:
            quietSamples = 0
        if quietSamples > quietCut or (quietSamples > 0 and i == numSamples-max_pulse_sample-nLookAhead-2):
            #end of pulse
            pulse_end = max_pulse_sample + i - quietSamples + 1 # was + 2
            break
            if pulse_end > numSamples-1:
                pulse_end = numSamples-1
    #print("  First Found Pulse End: ",pulse_end)
    
    # CHECK THAT A VALID PULSE WAS FOUND BEFORE ADDITIONAL STUFF IS DONE ----------------
    if pulse_end!=99999 and pulse_start!=99999:
        
        foundPulse = 1
    
        # ----- MANUAL OVERRIDE OF PULSE WIDTH--------------------------
        #if width is still greater than cut value, set pulse_end manually:
        if (pulse_end-pulse_start)>width_cut:
            pulse_end = pulse_start + width_cut    
    else:#if here, valid pulse not found, flag set to zero
        foundPulse = 0
    
    #if pulse_start < 0:
    #    pulse_start = 0
    #if pulse_end < 0:
    #    pulse_end = numSamples-1
    #return pulse_start, pulse_end;
    return pulse_start,pulse_end,foundPulse
