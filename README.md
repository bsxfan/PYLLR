# PYLLR
Python toolkit for likelihood-ratio calibration of binary classifiers 

The emphasis is on binary classifiers (for example speaker verification), where the output of the classifier is in the form of a well-calibrated log-likelihood-ratio (LLR). The tools include:
- PAV and ROCCH score analysis. 
- DET curves and EER
- DCF and minDCF
- Bayes error-rate plots
- Cllr
- Simple linear fusion and calibration (generative Gaussian and logistic regression) 

Most of the algorithms in PYLLR are Python translations of the older MATLAB [BOSARIS Tookit](https://sites.google.com/site/bosaristoolkit/). Descriptions of the algorithms are available in:

> Niko Brümmer and Edward de Villiers, [The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF](https://arxiv.org/abs/1304.2865), 2013.

For now, the Python version of this toolkit is available [here](https://github.com/luferrer/DCA-PLDA/blob/611d4a1e7fa1b106c038104d260fe0556f504b47/dca_plda/calibration.py#L1) as part of the [DCA-PLDA](https://github.com/luferrer/DCA-PLDA) repository. We plan to make an expanded version of the Python toolkit available in this repository. 
