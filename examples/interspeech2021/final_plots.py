import numpy as np

import matplotlib.pyplot as plt

from pyllr.plotting import  BayesErrorPlot
from pyllr.quick_eval import scoreslabels_2_eer
from pyllr.utils import tarnon_2_scoreslabels



def make_berplot(sys):

    fig, ax = plt.subplots()
    plo = np.linspace(-15,10,200)
    ber = BayesErrorPlot(ax,plo,sys.upper().replace("_","-"))

    datasets = ['voxceleb2', 'sitw.eval', 'heldout_fvcaus']
    plotlabels = ['VoxCeleb2', 'SITW', 'FVCAus']
    plotcolors = 'rgb'

    for dset, plotlabel, plotcolor in zip(datasets, plotlabels, plotcolors):

        data = np.load(f"scores/{sys}/{dset}.npz")
        tar, non = data["tar"], data["non"]
        tar, non = tar.astype('float'), non.astype('float')

        scores, labels = tarnon_2_scoreslabels(tar,non)
        eer = scoreslabels_2_eer(scores,labels)

        ber.add(scores,labels,f"{plotcolor}-",plotlabel=plotlabel)
        ax.plot([plo[0],plo[-1]],[eer,eer],f"{plotcolor}:",label=f"{plotlabel} EER")


    ber.legend()
    plt.savefig(f"{sys}_ber.pdf")


# Figure 1.  Plot the Trapezium Bound

data = np.load("scores/dca_plda/voxceleb2.npz")
tar, non = data["tar"], data["non"]
tar, non = tar.astype('float'), non.astype('float')

scores, labels = tarnon_2_scoreslabels(tar,non)
plo = np.linspace(-15,15,201)
fig, ax = plt.subplots()
ber = BayesErrorPlot(ax,plo,'Trapezium bound')
ber.add_pav(scores,labels,"g",plotlabel=r"$\hat P_e$")
eer = ber.rocch.EER()
ax.plot([plo[0],plo[-1]],[eer,eer],"r",label="EER")
ber.legend()
plt.savefig("trapezium.pdf")

# Figure 2.  Bayes Error Rate plot for the PLDA system
make_berplot("plda")

# Figure 3.  Bayes Error Rate plot for the DCA-PLDA system
make_berplot("dca_plda")



plt.show()
