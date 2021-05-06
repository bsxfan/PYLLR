import numpy as np
from pyllr.pav_rocch import PAV, ROCCH
from pyllr.cllr import cllr, min_cllr
from pyllr.utils import tarnon_2_scoreslabels, scoreslabels_2_tarnon, probit, probitinv

def scoreslabels_2_eer_cllr_mincllr(scores, labels):
    """
    Given labelled scores, computes the binary classifier evaluation 
    objectives: EER, Cllr and minCllr.
    
    The scores are assumed to be in log-likelihood-ratio (llr) form: 
        
        LLR = log P(data | target) - log P(data | non-target)
        
    If the scores or in some other form, EER and minCllr may still be 
    meaningful, but Cllr, which calibration sensitive, may be
    meaningless.     
    
    If the scores are well-calibrated, Cllr and minCllr will be close.
        
    
    Inputs:
        scores: (n,) an array of binary classifier scores in llr form
        labels: (n,) 0/1 class labels, where target = 1 and non-target = 0
        
    Outputs:
        EER: float. Equal-Error-Rate. This is the ROCCH-EER. 
        Cllr: float. Prior-weighted cross-entropy at prior = 0.5. Scaled such 
                     that Cllr = 1 when all llr scores are 0. 
        minCllr: float. The value that Cllr attains when the optimal score
                        calibration function, subject to monotonicity is found
                        usign the given labels. 
    
    """
    pav = PAV(scores,labels)
    eer = ROCCH(pav).EER()
    Cllr = cllr(*scoreslabels_2_tarnon(scores, labels))
    minCllr = min_cllr(pav)
    return eer, Cllr, minCllr


def tarnon_2_eer_cllr_mincllr(tar, non):
    """
    Given labelled scores, computes the binary classifier evaluation 
    objectives: EER, Cllr and minCllr.
    
    The scores are assumed to be in log-likelihood-ratio (llr) form: 
        
        LLR = log P(data | target) - log P(data | non-target)
        
    If the scores or in some other form, EER and minCllr may still be 
    meaningful, but Cllr, which calibration sensitive, may be
    meaningless.     
    
    If the scores are well-calibrated, Cllr and minCllr will be close.
        
   
    Inputs:
        tar: (m,) an array of binary classifier scores in llr form. All these
             scores are assumed to be of class 1 (targets).
        non: (n,) an array of binary classifier scores in llr form. All these
             scores are assumed to be of class 0 (non-targets).
        
    Outputs:
        EER: float. Equal-Error-Rate. This is the ROCCH-EER. 
        Cllr: float. Prior-weighted cross-entropy at prior = 0.5. Scaled such 
                     that Cllr = 1 when all llr scores are 0. 
        minCllr: float. The value that Cllr attains when the optimal score
                        calibration function, subject to monotonicity is found
                        usign the given labels. 
    
    """
    
    
    scores, labels =  tarnon_2_scoreslabels(tar, non)
    pav = PAV(scores,labels)
    eer = ROCCH(pav).EER()
    Cllr = cllr(tar, non)
    minCllr = min_cllr(pav)
    return eer, Cllr, minCllr

def tarnon_2_eer(tar,non):
    """
    Given labelled scores, computes the binary classifier evaluation 
    objective: Equal-Error-Rate (EER). The scores may be in uncalibrated form.
    The EER is not calibration-sensitive.

    Same as scoreslabels_2_eer_cllr_mincllr(), but with a different input 
    format for the input scores. See also the documentation for that function.
    
    Inputs:
        tar: (m,) an array of binary classifier scores. All these
             scores are assumed to be of class 1 (targets).
        non: (n,) an array of binary classifier scores. All these
             scores are assumed to be of class 0 (non-targets).
        
    Output:
        EER: float. Equal-Error-Rate. This is the ROCCH-EER. 
    
    """

    pav = PAV(*tarnon_2_scoreslabels(tar,non))
    return ROCCH(pav).EER()


    
def scoreslabels_2_eer(scores,labels):
    """
    Given labelled scores, computes the binary classifier evaluation 
    objective: Equal-Error-Rate (EER). The scores may be in uncalibrated form.
    The EER is not calibration-sensitive.

    Same as scoreslabels_2_eer_cllr_mincllr(), but with a different input 
    format for the input scores. See also the documentation for that function.
    
    Inputs:
        tar: (m,) an array of binary classifier score. All these
             scores are assumed to be of class 1 (targets).
        non: (n,) an array of binary classifier scores. All these
             scores are assumed to be of class 0 (non-targets).
        
    Output:
        EER: float. Equal-Error-Rate. This is the ROCCH-EER. 
    
    """
    pav = PAV(scores,labels)
    return ROCCH(pav).EER()




def tarnon_2_auc(tar,non):
    pav = PAV(*tarnon_2_scoreslabels(tar,non))
    return ROCCH(pav).AUC()

def tarnon_2_eer_auc(tar,non):
    pav = PAV(*tarnon_2_scoreslabels(tar,non))
    rocch = ROCCH(pav)
    return rocch.EER(), rocch.AUC()

def eer_2_auc_approx(eer):
    """
    The relationship between EER and AUC is not an exact function, because 
    AUC depends on details of the score distributions, other than the EER.
    This function gives a good approximation to AUC when the scores are 
    approximately Gaussian and when targets and non-targets have approximately
    equal variances.
    """
    return probitinv(np.sqrt(2)*probit(eer))


