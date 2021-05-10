
import numpy as np
from scipy.special import erf, erfinv, expit

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
    return sqrt2*erfinv(2.0*p - 1.0)

def probitinv(x):
    return (1.0 + erf(x/sqrt2) ) / 2.0


def eer_2_dprime(eer):
    return -2*probit(eer)

def cs_sigmoid(x):
    """numerically stable and complex-step-friendly version of sigmoid"""
    if not np.iscomplexobj(x): return expit(x)
    rx = np.real(x)
    p, q = expit(rx), expit(-rx)
    return p + 1.0j*p*q*np.imag(x)

def sigmoid(x,deriv=False):
    p = cs_sigmoid(x)
    if not deriv: return p
    
    q = cs_sigmoid(-x)
    def back(dp): return dp*p*q
    return p, back
    
    

def cs_softplus(x):
    """numerically stable and complex-step-friendly version of: 
       
       softplus = log( 1 + exp(x) )
    """
    if not np.iscomplexobj(x): 
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    #return np.log( 1 + np.exp(x) )
    rx = np.real(x)
    y = cs_softplus(rx)
    return y + 1.0j*expit(rx)*np.imag(x)


def softplus(x, deriv=False):
    y = cs_softplus(x)
    if not deriv: return y
    
    dydx = cs_sigmoid(x)
    def back(dy): return dy*dydx
    return y, back 
