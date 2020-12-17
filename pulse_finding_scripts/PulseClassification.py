import numpy as np

def ClassifyPulses(tba, t_rise, n_pulses):
    max_pulses = np.size(tba)
    classification = np.zeros(max_pulses)

    # May want everything below rise time of 0.2 to be S1-like, can subdivide S1 -> gas vs liquid?
    # Tricky bc discrimination space for gas-like pulses looks pretty different when voltages are on vs off
    # Clearly want to keep green as normal S1, always below ~0.125
    case1 = (tba < -0.65)*(t_rise < 0.125)
    case2 = (tba > -0.6)*(tba < -0.25)*(t_rise < 0.25)
    case3 = (tba > -0.65)*(tba < -0.25)*(t_rise > 1.05)
    case4 = (tba > -0.4) * (tba < -0.15) * (t_rise > 0.65) * (t_rise < 1.05)
    case5 = (tba > 0.15)*(tba < 0.45)
    case6 = (tba > 0.45)
    
    classification[case1] = 1
    classification[case2] = 2
    classification[case3] = 3
    classification[case4] = 4
    classification[case5] = 5
    classification[case6] = 6
    classification[n_pulses:] = 0 # Set pulse class to 0 for empty pulses

    
    return classification

