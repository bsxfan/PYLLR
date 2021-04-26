
This directory contains code to reproduce all figures for the InterSpeech 2021 paper

Niko Brümmer, Luciana Ferrer and Albert Swart, "Out of a hundred trials, how many errors does your speaker verifier make?", Interspeech Brno, 2021.

A preprint of the paper can be found on [arXiv](https://arxiv.org/abs/2104.00732)

To reproduce the figures in the paper, follow these steps:

1. Create and activate a conda environment
 * conda create --name pyllr python=3
 * conda activate pyllr

2. Install PYLLR
 * git clone git@github.com:bsxfan/PYLLR.git
 * cd PYLLR
 * pip install -r requirements.txt
 * pip install -e .

3. Download the scores from [here](https://bit.ly/3dSd8oh) into the examples/insterspeech2021 subdirectory.  The directory structure should be
 -- insterspeech2021
  |-- final_plots.py
  |-- scores
  |   |- dc_plda
  |   |- plda 

4.  Run the `final_plots.py` script to reproduce the figures in pdf format.

