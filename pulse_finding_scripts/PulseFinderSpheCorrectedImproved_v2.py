import numpy as np

#function to find start and end of a pulse in event
#inputs: event number and baseline subtracted waveform numpy array
#ouputs: sample number for start, end and max of pulse

def findaPulse( evt, waveforms_bls ):
    
    numSamples = len(waveforms_bls[0,:])
    boxAreaRolling = np.zeros(numSamples)
    tmpBoxArea = 0
    
    boxWidth = 12
    nLookAhead = 1
    nLookBefore = 1
    #noiseThreshold = 5*0.122*(10/417.7)# = 0.0146 currently
    #noiseThreshold = 0.03
    #noiseThreshold = 0.02
    
    #noiseThreshold = 5*0.122 #=0.61
    #noiseThreshold = 0.026
    #noiseThreshold = 0.03
    noiseThreshold = 0.06
    
    quietCut = 3
    
    width_cut = 60
    
    pulse_start = 99999
    pulse_end = 99999
    foundPulse = 0 # flag indicating a pulse was found
    
    
    #first loop through event and find box areas
    for t in range(numSamples-boxWidth-1):
        
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
        
        # MODIFY PULSE END IF WIDTH TOO LARGE
        # if pulse width too large, try again with higher threshold
        if (pulse_end-pulse_start)>width_cut: # and pulse_end!=99999
            noiseThreshold+=0.04
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
        
        
        # ------------ SPLITTING MERGED PULSES CODE ------------
        #if start and end aren't at the edges of the waveform and the found pulse is "wide", run extra bit here
        #to try and split up merged pulses
        flag_falling_r=0
        index_falling_r=0
        flag_rising_r=0
        index_rising_r=0
        flag_falling_l=0
        index_falling_l=0
        flag_rising_l=0
        index_rising_l=0
        pulse_end_try=0
        try_again_flag = 0
        second_pass_start = 0
        
        pulse_end_og = pulse_end #record OG pulse end before running splitting code
        
        rise_length=0
        length_cut=1
        
        min_sep = 35 #was 25
        
        run_split_width = 40 # was 65, April 20 moved to 40
        height_frac_falling = 0.4
        #height_frac_rising = 0.08
        #height_frac_rising = 0.045 # April 18 edit
        height_frac_rising = 0.04 # April 20 edit
        
        height_frac_falling2 = 0.02
        height_frac_rising2 = 0.02
        
        found_pulse_width = pulse_end - pulse_start
        #nLookBeforeMod = nLookBefore + 2
        #nLookAheadMod = nLookAhead + 2
        nLookBeforeMod = nLookBefore + 0 # April 18 edit
        nLookAheadMod = nLookAhead + 0 # April 18 edit
        
        if pulse_start!=0 and pulse_end!=numSamples-1 and found_pulse_width>run_split_width:
            # STEPPING FORWARD:
            #first start at max of found pulse, step forward in time computing an average
            for i in range(pulse_end-max_pulse_sample-nLookAheadMod):# i is relative to pulse max
                if i==0:
                    prev_newavg = 10000
                else:
                    prev_newavg = newavg
                newavg = np.mean( waveforms_bls[evt,max_pulse_sample+(i)-nLookBeforeMod:max_pulse_sample+(i)+nLookAheadMod] )
                #test if pulse is falling:
                if newavg < height_frac_falling*max_pulse_height and flag_falling_r==0:
                    flag_falling_r = 1
                    index_falling_r = i
                #test if pulse is rising after falling:
                if newavg > height_frac_rising*max_pulse_height and newavg>prev_newavg and flag_falling_r==1 and flag_rising_r==0:
                    flag_rising_r = 1
                    index_rising_r = i
                #record how long pulse is rising:
                if flag_rising_r==1 and newavg >= height_frac_rising*max_pulse_height and newavg>prev_newavg:
                    rise_length+=1
                #break if pulse goes back down:
                if flag_rising_r==1 and newavg < height_frac_rising*max_pulse_height:
                    break
            #if pulse rose long enough, find min between them and record that as new pulse end:
            if rise_length > length_cut:
                minpoint = np.argmin(waveforms_bls[evt,max_pulse_sample+index_falling_r:max_pulse_sample+index_rising_r])
                #pulse_end = max_pulse_sample + index_falling_r + minpoint
                pulse_end_try = max_pulse_sample + index_falling_r + minpoint
                #if new end is too close to pulse max, don't keep it
                if (pulse_end_try-max_pulse_sample)>min_sep:
                    pulse_end = pulse_end_try
                else:
                    pulse_end = pulse_end
                    try_again_flag = 1
                    second_pass_start = max_pulse_sample+index_rising_r
            
            # if pulse end not changed, run the afterpulsing finder to be sure
            if pulse_end == pulse_end_og:
                try_again_flag = 1
                second_pass_start = max_pulse_sample+min_sep
            
            # TRY AGAIN FOR AFTERPULSES
            if try_again_flag==1:
                flag_falling_r=0
                index_falling_r=0
                flag_rising_r=0
                index_rising_r=0
                rise_length = 0
                for i in range(pulse_end-second_pass_start-nLookAheadMod):# i is relative to rising index from first pass
                    if i==0:
                        prev_newavg = 10000
                    else:
                        prev_newavg = newavg
                    newavg = np.mean( waveforms_bls[evt,second_pass_start+(i)-nLookBeforeMod:second_pass_start+(i)+nLookAheadMod] )
                    #newavg = waveforms_bls[evt,second_pass_start+i]
                    #test if pulse is falling:
                    if newavg < height_frac_falling2*max_pulse_height and flag_falling_r==0:
                        flag_falling_r = 1
                        index_falling_r = i
                    #test if pulse is rising after falling:
                    if newavg > height_frac_rising2*max_pulse_height and newavg>prev_newavg and flag_falling_r==1 and flag_rising_r==0:
                        flag_rising_r = 1
                        index_rising_r = i
                    #record how long pulse is rising:
                    if flag_rising_r==1 and newavg >= height_frac_rising2*max_pulse_height and newavg>prev_newavg:
                        rise_length+=1
                    #break if pulse goes back down:
                    if flag_rising_r==1 and newavg < height_frac_rising2*max_pulse_height:
                        break
                #if pulse rose long enough, find min between them and record that as new pulse end:
                if rise_length > length_cut:
                    minpoint = np.argmin(waveforms_bls[evt,second_pass_start+index_falling_r:second_pass_start+index_rising_r])
                    pulse_end = second_pass_start + index_falling_r + minpoint
            
            
            rise_length = 0
            # STEPPING BACKWARDS
            #first start at max of found pulse, step backwards in time computing an average
            for i in range(max_pulse_sample-pulse_start-nLookBeforeMod):# i is relative to pulse max
                if i==0:
                    prev_newavg = 10000
                else:
                    prev_newavg = newavg
                newavg = np.mean( waveforms_bls[evt,max_pulse_sample-(i)-nLookBeforeMod:max_pulse_sample-(i)+nLookAheadMod] )
                #test if pulse is falling:
                if newavg < height_frac_falling*max_pulse_height and flag_falling_l==0:
                    flag_falling_l = 1
                    index_falling_l = i
                #test if pulse is rising after falling:
                if newavg > height_frac_rising*max_pulse_height and newavg>prev_newavg and flag_falling_l==1 and flag_rising_l==0:
                    flag_rising_l = 1
                    index_rising_l = i
                #record how long pulse is rising:
                if flag_rising_l==1 and newavg >= height_frac_rising*max_pulse_height and newavg>prev_newavg:
                    rise_length+=1
                #break if pulse goes back down:
                if flag_rising_l==1 and newavg < height_frac_rising*max_pulse_height:
                    break
            #if pulse rose long enough, find min between them and record that as new pulse end:
            if rise_length > length_cut:
                minpoint = np.argmin(waveforms_bls[evt,max_pulse_sample-index_rising_l:max_pulse_sample-index_falling_l])
                pulse_start = max_pulse_sample - index_rising_l + minpoint
        
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
