
import numpy as np
from scipy import special

sqrt2 = np.sqrt(2)

def tarnon_2_scoreslabels(tar,non):
    """
    Concatenates the two given score vectors into a single vector, with tar scores
    first. Construct an (integer) label vector of the same length, with 1's for
    tar scores and 0's for non scores.
    
    Returns scores, labels
    """
    scores = np.concatenate((tar,non))
    labels = np.zeros_like(scores,dtype=int)
    labels[:len(tar)] = 1.0
    return scores, labels

def scoreslabels_2_tarnon(scores,labels):
    """
    Splits the given scores vector according to the vector of binary labels.
    Labels for tar scores should equate to 1 (e.g. 1 or 1.0 or True) and
    everything else is considered non scores.
    
    Return tar, non
    """
    tt = labels==1
    tar = scores[tt]
    non = scores[np.logical_not(tt)]
    return tar, non    

def probit(p):
    return sqrt2*special.erfinv(2.0*p - 1.0)
