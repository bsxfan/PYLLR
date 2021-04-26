import numpy as np
from scipy.special import logit

from pyllr.utils import softplus

log2 = np.log(2)

def cross_entropy(tar,non,Ptar=0.5,deriv=False):
    logitprior = logit(Ptar)
    if not deriv:
        t = np.mean(softplus(-tar-logitprior))
        n = np.mean(softplus(non+logitprior))
        return ( Ptar*t  + (1-Ptar)*n ) / log2
    
    t, back1 = softplus(-tar-logitprior,deriv=True)
    n, back2 = softplus(non+logitprior,deriv=True)
    k1 = Ptar/(len(t)*log2)
    k2 = (1-Ptar)/(len(n)*log2)
    y = k1*t.sum()  + k2*n.sum()
    def back(dy):
        dtar = back1(-dy*k1)
        dnon = back2(dy*k2)
        return dtar, dnon
    return y, back
    

def cllr(tar,non,deriv=False):
    return cross_entropy(tar,non,Ptar=0.5,deriv=deriv)

    
def min_cllr(pav):
    llrs, tar_counts, non_counts = pav.llrs()
    tnz = tar_counts > 0  # avoid 0 * inf
    nnz = non_counts > 0
    T = tar_counts.sum()    
    N = non_counts.sum()
    tar_costs = softplus(-llrs)    
    non_costs = softplus(llrs)
    return ( (tar_costs[tnz] @ tar_counts[tnz]) / T \
            + (non_costs[nnz] @ non_counts[nnz]) / N ) / (2*log2)


    



    