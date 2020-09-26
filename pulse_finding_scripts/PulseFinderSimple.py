import numpy as np

#function to find start and end of a pulse in event
#inputs: event number and baseline subtracted waveform numpy array
#ouputs: sample number for start, end, and max of pulse

def findaPulse( evt, waveforms_bls ):
    
    numSamples = len(waveforms_bls[0,:])
    boxAreaRolling = np.zeros(numSamples)
    tmpBoxArea = 0
    areaStepSize = 5 # higher can help improve speed of initial search
    
    boxWidth = 1000
    nLookAhead = 1
    nLookBefore = 1
    
    noiseThreshold = 0.8
    
    quietCut = 3
    
    width_cut = 8000
    
    pulse_start = 99999
    pulse_end = 99999
    foundPulse = int(0) # flag indicating a pulse was found
    
    
    #first loop through event and find box areas
    for t in range(0,numSamples-boxWidth-1,areaStepSize):
        
        #for i in range(t,t+boxWidth):
        #    tmpBoxArea = tmpBoxArea + waveforms_bls[evt,i]
        tmpBoxArea = np.sum(waveforms_bls[evt,t:t+boxWidth])
        
        boxAreaRolling[t] = tmpBoxArea
        
    #find max area and the beginning of box that encloses that area
    max_boxArea_value = np.max(boxAreaRolling)
    max_boxArea_sample = np.argmax(boxAreaRolling)
    
    #now find waveform max value and location of max value inside box
    max_pulse_height = np.max(waveforms_bls[evt,max_boxArea_sample:max_boxArea_sample+boxWidth])
    max_pulse_sample = max_boxArea_sample + np.argmax(waveforms_bls[evt,max_boxArea_sample:max_boxArea_sample+boxWidth])
    
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
