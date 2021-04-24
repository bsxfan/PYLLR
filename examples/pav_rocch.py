"""
This script illustrates the use of the PAV and ROCCH algorithms.
"""

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

from pyllr.pav_rocch import PAV, ROCCH


if __name__ == "__main__":
    
    n = 1000
    non = randn(n)
    tar = 2 + 1.5*randn(n)
    
    scores = np.concatenate((tar,non)).astype(np.float32)
    labels = np.zeros_like(scores)
    labels[:len(tar)] = 1.0
    
    pav = PAV(scores,labels)
    rocch = ROCCH(pav)
    
    fig, ax = plt.subplots(2, 2)

    sc, llr = pav.scores_vs_llrs()
    ax[0,0].plot(sc,llr)
    ax[0,0].grid()
    ax[0,0].set_title("PAV: score --> log LR")
    
    pmiss,pfa = rocch.Pmiss_Pfa()
    ax[0,1].plot(pfa,pmiss,label='rocch')
    ax[0,1].plot(np.array([0,1]),np.array([0,1]),label="Pmiss = Pfa")
    ax[0,1].grid()
    ax[0,1].set_title("ROC convex hull")
    ax[0,1].legend(loc='best', frameon=False)

    plo = np.linspace(-5,5,100)
    ax[1,0].plot(sigmoid(plo),rocch.Bayes_error_rate(plo),label='minDCF')
    ax[1,0].grid()
    ax[1,0].legend(loc='best', frameon=False)
    ax[1,0].set_xlabel("P(target)")
    
    ber, pmiss, pfa = rocch.Bayes_error_rate(plo,True)
    ax[1,1].plot(sigmoid(plo),ber,label='minDCF')
    ax[1,1].plot(sigmoid(plo),pmiss,label='Pmiss')
    ax[1,1].plot(sigmoid(plo),pfa,label='Pfa')
    ax[1,1].legend(loc='best', frameon=False)
    ax[1,1].grid()
    
    print("EER = ",rocch.EER())

    plt.show()
    
    
    
    
    
    
    
    