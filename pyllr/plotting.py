"""
Module lib.calibration.plotting

This module provides some plots to graphically analyze the calibration of the outputs
of probabilistic binary classifiers.

"""
import numpy as np

import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid

from .pav_rocch import PAV, ROCCH
from .bayes_error_rate import default_error_rate, fast_Bayes_error_rate

from .utils import probit

class BayesErrorPlot:
    """
    This plot graphically analyzes the calibration of the outputs
    of probabilistic binary classifiers. The goodness of a classifier is analyzed 
    via the scores provided by the classifier on a supervised database of trials.
    This plot can be used to analyze a single classifier, or to compare two or 
    more different ones.
    
    The evaluation metric is the Bayes error-rate that results when attempting 
    to make binary classifcation decisions, when thresholding the scores at the
    theoretically optimal Bayes decision threshold.
    
    The plot displays Bayes error-rate on the vertical axis against 
    the prior log odds (log-ratio between the prior class probabilities), on the 
    horizontal axis. The plot is constructed by initializing with the range 
    of operating points. Thereafter methods can be called to add the graphs for 
    one or more classifiers.
    
    """
    
    def __init__(self,axes,prior_log_odds,title=""):
        """
        Initialize the plot, where:

            axes: matplotlib Axes on which to plot

            prior_log_odds: a sorted (n,) array of operating points to be used
                            on the x-axis. Too many values will be slow. Too few 
                            may miss details. You could try for example: 
                                prior_log_odds = np.linspace(-5,5,200)
                            Keep in mind that - prior_log_odds is the 
                            decision threshold against which the scores (assumed 
                            to be log-likelihood-ratios) are compared to make 
                            decisions.
                            
            title: string, optional  

              
            This initialization automatically adds the graph of the default 
            Bayes error-rate as a reference. The default corresponds to making
            binary Bayes classifications usign the class prior alone. 
            This error-rate is just min(p,1-p), where p = sigmoid(prior_log_odds),
            where p is the prior for class 1.                                



        """
        self.axes = ax = axes
        self.plo = plo = prior_log_odds
        ax.semilogy(plo,default_error_rate(plo),'k-',label=r'$\min(\pi,1-\pi)$')
        if len(title)>0: ax.set_title(title)
        #ax.set_xlabel(r'$\textrm{logit} $P_{tar}=-\textrm{threshold}$')
        ax.set_xlabel(r'$\log\frac{\pi}{1-\pi}$')
        ax.set_ylabel("Bayes error-rate")
        ax.grid()



    def add_pav(self,scores,labels,plotformat='k--',plotlabel="PAV-optimal",**kwargs):
        """
        Add a graph representing the empirical Bayes error-rate of a classifier,
        whose accuracy is represented by a set of scores obtained on a supervised
        evaluation (or valuation, or test) data set. This method automatically
        adjusts the decision threshold to the one that would give the minimum 
        error-rate, where the evaluation labels themselves are used in the 
        setting of the threshold. This graph answers the question of what could
        the performance have been (on this data set) if the scores had been 
        calibrated such that the theoretical Bayes threshold is also the emprically
        optimal one (at every operating point). The threshold minimization is 
        done effectively using the PAV algorithm.
        
        The maximum of this plot is the equal-error-rate (EER).
        
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
                    
            plotformat: string (optional, default 'k--') to pass formatting 
                        instructions (colour, line style) to the plotting 
                        algorithm. Pass the empty string to ignore, for example 
                        if you would rather pass keyword arguments to the plotter.
           
            plotlabel: string (optional), the legend entry for this graph
                       
            **kwargs: (optional) additional named plot formatting parameters
            
            
            This method returns the PAV and ROCCH objects that are used 
            internally, so that they may be re-used, for other purposes, such
            as computing the EER, or also making a TransformPlot.  
                    
        """
        ax, plo = self.axes, self.plo
        self.pav = pav = PAV(scores,labels)
        self.rocch = rocch = ROCCH(pav) 
        ber = rocch.Bayes_error_rate(plo)
        ax.semilogy(plo,ber,plotformat,label=plotlabel,**kwargs)
        return pav, rocch
    
        
    def add(self,scores,labels,plotformat="",plotlabel="actual",**kwargs):    
        """
        Add a graph representing the empirical Bayes error-rate of a classifier,
        whose accuracy is represented by a set of scores obtained on a supervised
        evaluation (or valuation, or test) data set. This graph answers the 
        question of what is the actual performance (on this data set) when the 
        scores are interpreted as-is as log likelihood-ratios and are thresholded
        to make bianry decisions at the theoretical Bayes threshold (at every 
        operating point). 
        
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
                    
            plotformat: string (optional, default "") to pass formatting 
                        instructions (colour, line style) to the plotting 
                        algorithm. Omit or pass the empty string to ignore, for 
                        example if you would rather pass keyword arguments to 
                        the plotter.
           
            plotlabel: string, the legend entry for this graph
                       
            **kwargs: (optional) additional plot formatting keyword parameters
                    
        """
        ax, plo = self.axes, self.plo
        ber = fast_Bayes_error_rate(scores, labels, plo) 
        ax.semilogy(plo,ber,plotformat,label=plotlabel,**kwargs)
        
        
    def legend(self,**kwargs):
        """
        After adding all graph components, call this method to make the legend. 
        """
        self.axes.legend(**kwargs)


class TransformPlot:
    """
    This plot displays the score to log-likelihood-ratio functions of one
    or more calibration transformations.
    
    Methods: 
        
        add(scores, logLRs, ...) adds a graph for a calibrator represented 
                                 empirically by its inputs and outputs. 
    
        add_pav(pav, ...) adds a graph to represent the optimal transform
                          represented by pav = PAV(scores,labels)
    
    """
    
    
    def __init__(self,axes,title=""):
        """
        Initialise the polot on the given axes.
        
            axes: matplotlib Axes
            title: string
        """
        self.axes = ax = axes
        if len(title)>0: ax.set_title(title)
        ax.set_xlabel('raw score')
        ax.set_ylabel("log LR")
        ax.grid()

        
    def add(self,scores,llrs,plotformat="",plotlabel="",sortscores=True,**kwargs):  
        """
        Adds a graph for a calibrator represented empirically by its inputs (scores)
        and outputs (log LRs). 
        
        Unless sortscores = False, the scores will be sorted to facilitate 
        line plotting.
        
            scores: (n,) array of binary classifier scores. The inputs of the 
                    calibrator.
            
            llrs: (n,) array of log P(score | class 1) - log P(score | class 0)
                  The outputs of the calibrator that you want to display.
            
            plotformat: string to specify colour and line style
            
            sortscores: Bool (optional, default = True). Set this to False
                        only if the scores are already sorted.
                        
            **kwargs: (optional) additional keyword arguments to the plotter             
        
        
        """
        limit = 5000
        n = len(scores)
        if n > limit:
            ii = np.random.choice(n,limit,replace=False)
            scores = scores[ii]
            llrs = llrs[ii]
        
        ax = self.axes
        if sortscores:
            ii = np.argsort(scores)
            scores, llrs = scores[ii], llrs[ii]
        ax.plot(scores,llrs,plotformat,label=plotlabel,**kwargs)
        
        
    def add_pav(self,pav,plotformat="k",plotlabel="PAV",**kwargs):  
        """
        Adds a graph to represent the optimal score to log-likelihood-ratio 
        transform represented by pav = PAV(scores,labels). A graph of the identity 
        transform is added automatically.
        
        Be aware that the optimal PAV transform often has logLR
        values of -inf on the left and +inf on the right. These are not plotted!
        The left -inf values occur for all class 0 scores < min(class 1 score). 
        The right +inf values occur for all class 1 scores > max(class 0 scores). 
        
        """
        scores, llrs = pav.scores_vs_llrs()
        xmin, xmax = scores[0], scores[-1]
        self.add([xmin,xmax],[xmin,xmax],'k--',"identity",False)
        self.add(scores,llrs,plotformat,plotlabel,False,**kwargs)
        
        
    def legend(self,**kwargs):
        self.axes.legend(**kwargs)
        
        
        
        
class DETPlot:
    """
    """
    
    
    def __init__(self,axes,title=""):
        """
        Initialise the plot on the given axes.
        
            axes: matplotlib Axes
            title: string
        """
        self.axes = ax = axes
        if len(title)>0: ax.set_title(title)
        ax.set_xlabel('Pfa[%]')
        ax.set_ylabel("Pmiss[%]")
        
        # pfa_limits = probit(np.array([3e-6,5e-1]))
        # pmiss_limits = probit(np.array([3e-4,9e-1]))
        xticks = np.array([1e-5,1e-4,1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,4e-1])
        xticklabels = ['0.001',' 0.01','  0.1','  0.2','  0.5','    1','    2','    5','   10','   20','   40']
        yticks = np.array([1e-3,2e-3,5e-3,1e-2,2e-2,5e-2,1e-1,2e-1,4e-1,8e-1])
        yticklabels = ['0.1','0.2','0.5',' 1 ',' 2 ',' 5 ',' 10',' 20',' 40',' 80']
    
        #ax.set_xlim(*pfa_limits)
        #ax.set_ylim(*pmiss_limits)
        #ax.set_box_aspect(1)
        ax.set_aspect("equal")
        
        
        ax.set_xticks(probit(xticks))
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(probit(yticks))
        ax.set_yticklabels(yticklabels)
        ax.grid()

        
    def add(self,rocch,plotformat="",plotlabel="",**kwargs):  
        pmiss, pfa = rocch.Pmiss_Pfa()
        self.axes.plot(probit(pfa),probit(pmiss),plotformat,label=plotlabel,**kwargs)
        
    def legend(self,**kwargs):
        self.axes.legend(**kwargs)
        
        
def tarnon_hist(tar,non,axes,title=None,tbins=None,nbins=None):
    if title is None: title = "score histogram"
    if tbins is None: tbins = max(20,min(200,int(len(tar)/20)))    
    if nbins is None: nbins = max(20,min(200,int(len(non)/20)))    
    axes.hist(non,bins=nbins,density=True)
    axes.hist(tar,bins=tbins,density=True,alpha=0.8)
    axes.set_title(title)
    axes.legend(["non","tar"])
