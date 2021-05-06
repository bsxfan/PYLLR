"""
Module lib.calibration.pav_rocch

This module provides a fast (for Python) Pool Adjacent Violators PAV algorithm
implementation. The basic algorithm is invoked in sklearn.isotonic, although we 
peeled off a layer of pre- and post-processing that we don't need.

The PAV algorithm provides an optimal score-to-posterior, or score-to-log-
likelihood-ratio transformation, where the scores are possibly uncalibrated 
binary classifier outputs. The PAV can also be used to construct the ROC convex 
hull (ROCCH), which provides further score analysis tools. 


"""
import numpy as np
from sklearn._isotonic import _inplace_contiguous_isotonic_regression as fastpav


from scipy.special import expit as sigmoid
from scipy.special import logit
from scipy.optimize import minimize_scalar


class PAV:
    """
    The constructor PAV(scores,labels) invokes the PAV algorithm and stores 
    the result in a convenient form in the newly created object. There are methods 
    to recover the score to log-likelihood-ratio transform and the ROC convex 
    hull.
    """
    def __init__(self,scores,labels):
        """
        Constructor for PAV.
        
            scores: (n,) array of real-valued binary classifier 
                    scores, where more positive scores support class 1, and 
                    more negative scores support class 0. The scores can be 
                    in the form of (possibly uncalibrated) probabilities,
                    likelihood-ratios, or log-likelihood-ratios, all of which
                    satisfy the above criteria for being binary classifier 
                    scores.
            labels: (n,) array of 0/1 labels, identifying the true class 
                    associated with each score. There has to be at least one
                    example of each class.
        """
       
        self.T = T = labels.sum()
        self.N = N = len(labels) - self.T
        assert T > 0 < N
        
        ii = np.lexsort((-labels,scores))  # when scores are equal, those with 
                                           # label 1 are considered inferior
        weights = np.ones_like(scores, dtype=np.float64)
        y = np.empty_like(scores, dtype=np.float64)
        y[:] = labels[ii]
        fastpav(y,weights)

        p, ff, counts = np.unique(y, return_index = True, return_counts = True)
        self.nbins = len(p)
        self.p = p
        self.counts = counts
        self.scores = np.empty((self.nbins,2))
        self.scores[:,0] = scores[ii[ff]]            # bin low scores
        self.scores[:,1] = scores[ii[ff+counts-1]]   # bin high scores
        
        self.targets = np.rint(counts * p).astype(int)    # number of target scores in each PAV block    
        self.nontars = counts - self.targets              # number of non-tar scores in each PAV block
        assert self.targets.sum() == T
        assert self.nontars.sum() == N
        
    def rocch(self, asmatrix = False):
        """
        This returns the convex hull of the ROC (receiver operating curve)
        associated with the scores and labels that were used to construct this 
        PAV object. The convex hull is returned as a list of vertices in the 
        (Pmiss, Pfa) ROC plane. The Pmiss values are non-decreasing, while the 
        Pfa values are non-increasing. The first vertex is at (0,1) and the 
        last at (1,0). There are always at least two vertices. The vertices 
        describe a convex, piece-wise linear curve. 
        
        The result can be retuned as a (2,n-vertices) matrix, or Pmiss and 
        Pfa can be returned separately, according to the asmatrix flag.
        
        The user does not typically have to invoke this call. Instead the user 
        would do my_rocch = ROCCH(PAV(scores,labels)) and then call some methods
        on my_rocch.
        """
        nbins = self.nbins
        PmissPfa = np.empty((2,nbins+1)) 
        pmiss = PmissPfa[0,:]
        pfa = PmissPfa[1,:]
        pmiss[0] = 0.0
        np.cumsum(self.targets, out = pmiss[1:])
        pmiss /= self.T

        pfa[0] = 0.0
        np.cumsum(self.nontars, out = pfa[1:])
        pfa -= self.N
        pfa /= (-self.N)
        return PmissPfa if asmatrix else (pmiss, pfa)
    
    def llrs(self):
        """
        Returns three vectors: 
            - The llrs for each PAV bin.
            - The number of targets (class 1) in each bin
            - The number of non-targets (class 0) in each bin
        """
        llr = logit(self.p) - np.log(self.T / self.N)
        return llr, self.targets, self.nontars
    
    
    def scores_vs_llrs(self):
        """
        Returns score and llr points, convenient for plotting purposes. A score 
        vector and an llr vector are returned, each with 2*nbins elements, 
        where nbins is the number of bins in this PAV solution. The scores 
        vector alternates the minimum and maximum score in each bin. There is 
        only one llr value associated with each bin, but those values are 
        duplicated, to correspond to the scores. The resulting plot of 
        scores vs llrs is steppy, with exactly horizontal and vertical line 
        segments.
        
        The initial and final llr bins may be -inf and +inf.
        
        """
        p = self.p
        LLRs = np.empty_like(self.scores)
        llr = LLRs[:,0]
        llr[:] = logit(p)
        llr -= np.log(self.T / self.N)   
        LLRs[:,1] = llr 
        return self.scores.ravel(), LLRs.ravel()


class ROCCH:
    """
    An ROCCH object contains the convex hull of the ROC (receiver operating 
    curve) associated with a set of binary classifier scores and labels. This 
    object can be constructed from a PAV object.
    
    """
    def __init__(self,pava):
        """
        ROCCH(PAV(scores,labels)) constructs an object of this class.
        """    
        self.PmissPfa = pava.rocch(True)
        
        
        
    def AUC(self):
        """
        AUC = Area Under Curve (i.e. the ROCCH curve). This can be interpreted
        as an error-rate. It is the probability that a randomly selected 
        target score exceeds a randomly selected non-target score.
        """
        pmiss = self.PmissPfa[0,:]     # increasing
        pfa = self.PmissPfa[1,:]       # decreasing
        n = len(pmiss)-1               # number of segments
        sum = 0.0
        for i in range(n):
            delta_x = pfa[i] - pfa[i+1]
            avg_y = (pmiss[i] + pmiss[i+1]) / 2
            sum += delta_x * avg_y
        return sum    
            

    def Pmiss_Pfa(self):
        """
        Returns the set of ROCCH vertices as two (n-vertices,) arrays, for
        each of Pmiss and Pfa.
        """
        return self.PmissPfa[0,:], self.PmissPfa[1,:]

    def Bayes_error_rate(self,prior_log_odds,return_Pmiss_Pfa = False):
        """
        Returns the optimal Bayes error-rate at one or more operating points
        of a binary classifier. The accuracy of the binary classifier is 
        empricially represented by the scores and labels that were used to
        construct this ROCCH object. The `prior' is a synthetic prior 
        probability for class 1, but is passed to this method in log-odds form,
        i.e. prior_log_odds = log(prior) - log(1-prior). One or more prior values
        can be passed, where each value is referred to as an operating point.
        For each operating point at prior = p, the returned optimal 
        Bayes-error-rate is:
            
            BER(p) = min_t p * Pmiss(t) + (1-p) * Pfa(t)
            
        where t is the decision threshold applied to the scores that were used 
        to construct this ROCCH. 
        
        The algorithm used here is typically faster than a naive implementation
        that operates directly on the scores, because the minimization can be 
        done only over the ROCCH vertices, of which are typically far fewer 
        than the original scores.
            
            
        """
        sc = np.isscalar(prior_log_odds)
        m = 1 if sc else len(prior_log_odds)
        PP = np.empty((2,m))
        PP[0,:] = sigmoid(prior_log_odds)
        PP[1,:] = sigmoid(-prior_log_odds)
        E = PP.T @ self.PmissPfa              # (m,nbins+1)
        if not return_Pmiss_Pfa:
            ber = E.min(axis=1)
            return ber.item() if sc else ber
        jj = E.argmin(axis=1)
        ber = E[range(len(jj)),jj]
        Pmiss = self.PmissPfa[0,jj]
        Pfa = self.PmissPfa[1,jj]
        if sc: ber, Pmiss, Pfa = ber.item(), Pmiss.item(), Pfa.item() 
        return ber, Pmiss, Pfa
        
    def EER(self):
        """Returns the value of the equal error-rate (EER), which is equal to 
           the Pmiss and Pfa values on the ROCCH curve, where they are equal.
           The EER is equal to the maximum of the optimal Bayes-error-rate of 
           this curve, when maximized w.r.t. the operating point. 
        
        """
        f = lambda x: -self.Bayes_error_rate(x)
        res = minimize_scalar(f)
        return -f(res.x)
        



