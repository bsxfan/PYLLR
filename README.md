# PYLLR
Python toolkit for likelihood-ratio calibration of binary classifiers 

The emphasis is on binary classifiers (for example speaker verification), where the output of the classifier is in the form of a well-calibrated log-likelihood-ratio (LLR). The tools include:
- PAV and ROCCH score analysis. 
- DET curves and EER
- DCF and minDCF
- Bayes error-rate plots
- Cllr
- Fusion and calibration are not presently implemented, but we will do so soon.

Most of the algorithms in PYLLR are Python translations of the older MATLAB [BOSARIS Tookit](https://sites.google.com/site/bosaristoolkit/). Descriptions of the algorithms are available in:

> Niko Brümmer and Edward de Villiers, [The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF](https://arxiv.org/abs/1304.2865), 2013.

## Out of a hundred trials, how many errors does your speaker verifier make?
We have incluced in the examples directory, some code that reproduces the plots in our paper:

> Niko Brümmer, Luciana Ferrer and Albert Swart, "Out of a hundred trials, how many errors does your speaker verifier make?", 2011, https://arxiv.org/abs/2104.00732. 

For instructions, go to the [readme](https://github.com/bsxfan/PYLLR/blob/main/examples/interspeech2021/README.txt).
