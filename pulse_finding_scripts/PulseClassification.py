import numpy as np

def ClassifyPulses(p_area_tba, p_afs_2l, p_afs_50, tscale):
    classification = np.zeros(4)

    fMt = tscale*(p_afs_50 - p_afs_2l)
    tba = p_area_tba

    case1 = (p_area_tba < -0.65)*(fMt < 0.125)
    case2 = (tba > -0.6)*(tba < -0.25)*(fMt < 0.25)
    case3 = (tba > -0.65)*(tba < -0.25)*(fMt > 1.05)
    case4 = (tba > -0.4)*(tba < -0.15)*(fMt > 0.65)*(fMt < 1.05)
    case5 = (tba > 0.15)*(tba < 0.45)
    case6 = (tba > 0.45)
    
    classification[case1] = 1
    classification[case2] = 2
    classification[case3] = 3
    classification[case4] = 4
    classification[case5] = 5
    classification[case6] = 6

    
    return classification

