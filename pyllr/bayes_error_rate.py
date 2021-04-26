"""
The algorithms used here can be found in:
    
  
Niko Brummer and Edward de Villiers,    
"The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF"",
2013, 
https://arxiv.org/abs/1304.2865    
"""

import numpy as np
from scipy import stats
from scipy.special import expit as sigmoid

from .pav_rocch import PAV, ROCCH

def ordinalrank(x):
    return stats.rankdata(x, method='ordinal')

def fast_Bayes_error_rate(scores, labels, prior_log_odds, return_der_Pmiss_Pfa = False):
    """
    Returns the Bayes error-rate at one or more operating points, when the 
    given scores are treated as log-likelihood-ratios and used to make 
    binary decisions at the theoretically optimal Bayes threshold.
    
    Abbreviate prior-log-odds as plo, and let p = sigmoid(plo) be the prior
    probability for class 1 (and 1-p the prior for class 0). Then Bayes-error-rate 
    is:
        
        ber(plo) = p Pmiss(-plo) + (1-p) Pfa(-plo)
        
    Here the miss-rate, Pmiss(-plo), is the fraction of class 1 trials, whose 
    scores are below the Bayes threshold of -plo. The false-alarm rate, Pfa(-plo)  
    is the fraction of class 0 trials, whose scores are above the Bayes threshold 
    of -plo.   
    
    The algortihm used here is typically faster than a naive implementation
    where each theshold is applied separately to all the scores. This algorithm
    jointly sorts the scores and thresholds and computes the error-rates
    from the resulting sorted ranks.
    
    """    
    plo = prior_log_odds
    assert (np.diff(plo) >=0 ).all()  # plo must be sorted
    thr = -plo                        # Bayes threshold

    tar = scores[labels==1]
    non = scores[labels==0]

    D = len(plo)                      # number of operating points
    T = len(tar)
    N = len(non)
    assert N + T > 0 < D

    Ddownto1 = np.arange(D,0,-1)

    rk = ordinalrank(np.concatenate((thr,tar)))
    rkD = rk[:D]
    Pmiss = ( rkD - Ddownto1 ) /  T

    rk = ordinalrank(np.concatenate((thr,non)))
    rkD = rk[:D]
    Pfa = ( N - rkD + Ddownto1 ) / N

    Ptar = sigmoid(plo)
    Pnon = sigmoid(thr)

    # Bayes error-rate
    ber = Ptar * Pmiss + Pnon * Pfa

    # default error-rate, using prior only
    der = np.minimum(Ptar,Pnon)

    return (ber, der, Pmiss, Pfa) if return_der_Pmiss_Pfa else ber


def default_error_rate(prior_log_odds):
    """
    Let the prior probability for class 1 be:

        p = sigmoid(prior-log-odds)
          = 1 - sigmnoid(- prior-log-odds)

    where 1-p is the prior probabiity for class 0. The default Bayes error-rate
    is min(p, 1-p), which is he error-rate that would obtain if bianry decisions
    were made using the prior (p) alone. 
    
    The default error-rate provides a useful reference against which to compare 
    the Bayes error-rate of a binary classifier.
    
          
    """
    
    Ptar = sigmoid(prior_log_odds)
    Pnon = sigmoid(-prior_log_odds)
    return np.minimum(Ptar,Pnon)
    


def Bayes_error_rate_analysis(scores,labels,prior_log_odds):
    """
    This is a convenience method to compute the optimal, actual and default
    Bayes error-rates over a given range of operating points
    
    """
    plo = prior_log_odds
    pav = ROCCH(PAV(scores,labels)).Bayes_error_rate(plo)
    ber, der, Pmiss, Pfa = fast_Bayes_error_rate(scores,labels,plo)
    return pav, ber, der


     
        





    
    
