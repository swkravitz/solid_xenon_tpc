import numpy as np

def ClassifyPulses(tba, t_rise, n_pulses, p_area):
    max_pulses = np.size(tba)
    classification = np.zeros(max_pulses)

    # May want everything below rise time of 0.2 to be S1-like, can subdivide S1 -> gas vs liquid?
    # Tricky bc discrimination space for gas-like pulses looks pretty different when voltages are on vs off
    # Clearly want to keep green as normal S1, always below ~0.125
    max_t_rise = (1.5/100/np.log10(2))*(np.sign(p_area)*(np.absolute(p_area)**(np.log10(2)/2.5)))
    case1 = (tba < 0)*(t_rise < max_t_rise) # normal S1s
    case2 = (tba >= 0)*(t_rise < max_t_rise) # top-focused S1s; e.g. in gas or LXe above top array
    case3 = (tba > -0.25)*(t_rise >= max_t_rise) # normal-ish S2s
    case4 = (tba <= -0.25)*(t_rise >= max_t_rise) # Unclear; possible S1/S2 merged pulses
    
    classification[case1] = 1
    classification[case2] = 2
    classification[case3] = 3
    classification[case4] = 4
    #classification[case5] = 5
    #classification[case6] = 6
    classification[n_pulses:] = 0 # Set pulse class to 0 for empty pulses

    
    return classification

