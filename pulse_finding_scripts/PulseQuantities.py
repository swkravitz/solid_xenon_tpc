import numpy as np

# functions to calculate interesting quantities for pulses
# inputs: pulse start and end samples and baseline subtracted waveform numpy array
# outputs: area, max height, max height sample, min height, mean and RMS between the start and end times

def GetPulseArea( p_start, p_end, waveforms_bls ):
    
    area = -999.
    
    try:
        area = np.sum( waveforms_bls[p_start:p_end] )
    except ValueError:
        area = -999.
    
    return area

def GetPulseAreaChannel(p_start, p_end, waveform_bls):
    # This one needs improving...
    area = []
    n_sipms = np.shape(waveform_bls)[0]-1
    for i in range(n_sipms):
        area.append(np.sum(waveform_bls[i, p_start:p_end] ) )
    while len(area) < n_sipms:
        area.append(0)

    return area

def GetPulseMaxSample( p_start, p_end, waveforms_bls ):
    
    max_height_sample = int(-999)
    
    if p_start != p_end:
        try:
            max_height_sample = p_start + np.argmax( waveforms_bls[p_start:p_end] )
        except ValueError:
            max_height_sample = int(-999)
    else:
        max_height_sample = int(-999)
    
    return max_height_sample

def GetPulseMaxHeight( p_start, p_end, waveforms_bls ):
    
    max_height = -999.
    max_height_sample = int(-999)

    if p_start != p_end:
        try:
            max_height_sample = p_start + np.argmax( waveforms_bls[p_start:p_end] )
            max_height = waveforms_bls[max_height_sample]
        except ValueError:
            max_height_sample = int(-999)
            max_height = -999.
    else:
        max_height_sample = int(-999)
        max_height = -999.
    
    return max_height

def GetPulseMinHeight( p_start, p_end, waveforms_bls ):
    
    min_height = 999.
    
    try:
        min_height = np.min( waveforms_bls[p_start:p_end] )
    except ValueError:
        min_height = 999.
    
    return min_height

def GetPulseMeanAndRMS( p_start, p_end, waveforms_bls ):
    
    mean = int(-999)
    rms = int(-999)
    
    try:
        # Calculate mean and rms times:
        
        # First step is to make an array that is the pulse just between the start and end times
        pulse_waveform = waveforms_bls[p_start:p_end]
        
        # Now make an array that contains the sample numbers of the found pulse
        # i.e. its first value is the pulse start and the last is the pulse end
        #pulse_sample_nums = np.arange(p_start_sum_le[0],p_end_sum_le[0]+1)
        pulse_sample_nums = np.arange( p_end - p_start )
        
        # Mean time:
        mean = np.dot( pulse_waveform,pulse_sample_nums ) / np.sum( pulse_waveform )
        
        #RMS time:
        rms = np.sqrt( np.dot( pulse_waveform, np.square(pulse_sample_nums-mean) ) / np.sum( pulse_waveform ) )
        
        #print "start = %d" % p_start
        #print "end = %d" % p_end
        #print "mean = %f" % mean
        #print "rms = %f" % rms
        
    except ValueError:
        mean = int(-999)
        rms = int(-999)
    
    return mean, rms    

# function to calculate samples at 10%, 50% and 90% area points, integrating from the left
# currently returns 2% area from left and right and 50% area from left
def GetAreaFractionLoop( p_start, p_end, waveforms_bls ):
    
    afs_2l = int(999999) 
    afs_2r = int(999999)
    afs_50 = int(999999)
    afs_1 = int(999999)
    afs_25 = int(999999)
    afs_75 = int(999999)
    afs_99 = int(999999)
    #afs_98 = int(9999)
    
    p_area = GetPulseArea( p_start, p_end, waveforms_bls )
    """if p_area < 0.:
        afs_2l = int(-999)
        afs_2r = int(-999)
        #afs_10 = int(-999)
        afs_50 = int(-999)
        afs_1 = int(-999)
        afs_25 = int(-999)
        afs_75 = int(-999)
        afs_99 = int(-999)
        #afs_90 = int(-999)
        #afs_98 = int(-999)
        return afs_2l,afs_2r,afs_1,afs_25,afs_50,afs_75,afs_99"""
    
    #move through pulse sample-by-sample from left
    rolling_area = 0.
    area_fraction = 0.
    for i in range(int(p_start),int(p_end)):
        
        rolling_area = np.sum(waveforms_bls[p_start:i])
        area_fraction = rolling_area/p_area
        
        if (area_fraction >= 0.01) and (i < afs_1):
            afs_1 = int(i)
        if (area_fraction >= 0.02) and (i < afs_2l):
            afs_2l = int(i)
        #if (area_fraction >= 0.10) and (i < afs_10):
        #    afs_10 = int(i)
        if (area_fraction >= 0.25) and (i < afs_25):
            afs_25 = int(i)
        if (area_fraction >= 0.50) and (i < afs_50):
            afs_50 = int(i)
        if (area_fraction >= 0.75) and (i < afs_75):
            afs_75 = int(i)
        if (area_fraction >= 0.99) and (i < afs_99):
            afs_99 = int(i)
            break
        #if (area_fraction >= 0.90) and (i < afs_90):
        #    afs_90 = int(i)
        #if (area_fraction >= 0.98) and (i < afs_98):
        #    afs_98 = int(i)
    
    #move through pulse sample-by-sample from right
    #print("from the right...")
    rolling_area = 0.
    area_fraction = 0.
    for i in range(int(p_end-1),int(p_start-1),-1): #gives [p_end-1,p_start-1) and so includes p_start
        
        rolling_area = np.sum(waveforms_bls[i:p_end-1])
        area_fraction = rolling_area/p_area
        #print("i=",i,"area =",rolling_area,"area fraction = ",area_fraction)
        if (area_fraction >= 0.02):
            afs_2r = int(i)
            break
    
    return afs_2l,afs_1,afs_25,afs_50,afs_75,afs_99

# function to calculate samples at 10% and 50% max height point looking from left and the right
def GetHeightFractionSamples( p_start, p_end, waveforms_bls ):
    
    hfs_10l = int(999999) # looking from left
    hfs_50l = int(999999) # looking from left
    hfs_10r = int(-999999) # looking from right
    hfs_50r = int(-999999) # looking from right
    
    p_max_height = GetPulseMaxHeight( p_start, p_end, waveforms_bls )
    if p_max_height < 0.:
        hfs_10l = int(-99999) # looking from left
        hfs_50l = int(-99999) # looking from left
        hfs_10r = int(-99999) # looking from right
        hfs_50r = int(-99999) # looking from right
        return hfs_10l, hfs_50l, hfs_10r, hfs_50r

    # only consider the part of the waveform in this pulse, start of pulse is sample 0
    height_fractions = waveforms_bls[int(p_start):int(p_end)]/p_max_height
    # get first and last index where height is greater than x% of max
    hfs_10 = np.argwhere(height_fractions >= 0.10)
    hfs_10l = hfs_10[0]
    hfs_10r = hfs_10[-1]

    hfs_50 = np.argwhere(height_fractions >= 0.50)
    hfs_50l = hfs_50[0]
    hfs_50r = hfs_50[-1]
    
    return hfs_10l, hfs_50l, hfs_10r, hfs_50r


def GetAreaFraction(p_start, p_end, waveform_bls):

    p_area = GetPulseArea( p_start, p_end, waveform_bls )

    if p_start != p_end:
        p_afc = np.cumsum( waveform_bls[p_start:p_end] )/p_area
    else:
        return -1,-1,-1,-1,-1,-1
        

    afs_1 = np.argmax(p_afc >= 0.01) + p_start
    afs_2l = np.argmax(p_afc >= 0.02) + p_start
    afs_25 = np.argmax(p_afc >= 0.25) + p_start
    afs_50 = np.argmax(p_afc >= 0.50) + p_start
    afs_75 = np.argmax(p_afc >= 0.75) + p_start
    afs_99 = np.argmax(p_afc >= 0.99) + p_start
    
    return afs_2l,afs_1,afs_25,afs_50,afs_75,afs_99


############################################################

def ClearWaveform( p_start, p_end, waveforms_bls ):
    
    waveforms_new = waveforms_bls.copy()
    waveforms_new[p_start:p_end] = 0.
    
    return waveforms_new

def GetTailArea( p_start, p_end, waveforms_bls ):
    
    area10 = -999.
    area15 = -999.
    area20 = -999.
    area25 = -999.
    area30 = -999.
    area35 = -999.
    
    width = p_end-p_start
    
    try:
        if( width>10 ):
            area10 = GetPulseArea( p_start+10, p_end, waveforms_bls )
        if( width>15 ):
            area15 = GetPulseArea( p_start+15, p_end, waveforms_bls )
        if( width>20 ):
            area20 = GetPulseArea( p_start+20, p_end, waveforms_bls )
        if( width>25 ):
            area25 = GetPulseArea( p_start+25, p_end, waveforms_bls )
        if( width>30 ):
            area30 = GetPulseArea( p_start+30, p_end, waveforms_bls )
        if( width>35 ):
            area35 = GetPulseArea( p_start+35, p_end, waveforms_bls )
        
    except ValueError:
        area10 = -999.
        area15 = -999.
        area20 = -999.
        area25 = -999.
        area30 = -999.
        area35 = -999.
    
    return area10, area15, area20, area25, area30, area35

# return list of more precise baselines per channel, estimated separately for each pulse
# pulse_baselines[i_pulse][j_channel]
# waveform_bls is the full list of waveforms per channel for a single event
def GetBaselines(p_starts, p_ends, waveform_bls):

    baseline_window = 1000 # take mean over this many samples
    rms_max_scale = 5 # do not trust baseline estimate from windows w/ RMS this much larger than from start of event
    baselines = [ np.mean( ch_j[0:baseline_window] ) for ch_j in waveform_bls ]
    baseline_sum_rms = np.std( waveform_bls[-1][0:baseline_window] )

    pulse_baselines = [baselines]

    for ii in range(1,len(p_starts)):
        if (p_starts[ii]-p_ends[ii-1]) < baseline_window:
            pulse_baselines.append(baselines)
            continue
        else:
            baseline_data = [ch_j[(p_starts[ii]-baseline_window):p_starts[ii]] for ch_j in waveform_bls]
            this_baseline_rms = np.std( baseline_data[-1] )
            if this_baseline_rms > rms_max_scale*baseline_sum_rms: # just use previous baseline if RMS is bad
                pulse_baselines.append(baselines)
                continue
            baselines = [np.mean( ch_j ) for ch_j in baseline_data]
            pulse_baselines.append(baselines)

    return pulse_baselines
