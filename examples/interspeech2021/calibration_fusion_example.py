import numpy as np

import matplotlib.pyplot as plt

from pyllr.plotting import  BayesErrorPlot
from pyllr.quick_eval import scoreslabels_2_eer
from pyllr.utils import tarnon_2_scoreslabels

from pyllr.calibration import train_calibration

def load_scores(fname):
    scores = np.load(fname)
    tar, non = scores["tar"], scores["non"]
    tar, non = tar.astype('float'), non.astype('float')
    return tarnon_2_scoreslabels(tar,non)


if __name__ == "__main__":

    datasets = ['voxceleb2', 'sitw.eval', 'heldout_fvcaus']

    dset = datasets[0]

    scores1, labels = load_scores(f"scores/plda/{dset}.npz")
    scores2, labels2 = load_scores(f"scores/dca_plda/{dset}.npz")
    assert np.all(labels==labels2), "Scores are not aligned"
    scores = np.vstack((scores1,scores2)).T

    ptar=0.5
    cal,cal_params = train_calibration(scores1,labels,ptar=ptar)
    scores1c = cal(scores1)
    cal,cal_params = train_calibration(scores2,labels,ptar=ptar)
    scores2c = cal(scores2)
    cal,cal_params = train_calibration(scores,labels,ptar=ptar)
    fusion = cal(scores)


    fig, ax = plt.subplots()
    plo = np.linspace(-15,10,200)
    ber = BayesErrorPlot(ax,plo,dset)

    ber.add(scores1,labels,f"b--",plotlabel="PLDA")
    ber.add(scores1c,labels,f"b",plotlabel="PLDA")
    ber.add(scores2,labels,f"g--",plotlabel="DCA PLDA")
    ber.add(scores2c,labels,f"g",plotlabel="DCA PLDA")
    ber.add(fusion,labels,f"r",plotlabel="Fusion")
    # ax.plot([plo[0],plo[-1]],[eer,eer],f"{plotcolor}:",label=f"{plotlabel} EER")

    ber.legend()

    plt.show()

