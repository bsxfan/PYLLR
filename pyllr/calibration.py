"""
Module pyllr.calibration

"""
import numpy as np

from scipy.special import expit, logit
from scipy.optimize import minimize



log2 = np.log(2)

def sigmoid(x,deriv=False):
    p = expit(x)
    if not deriv: return p

    q = expit(-x)
    def back(dp): return dp*p*q
    return p, back

def softplus(x,deriv=False):
    # softplus = log(1+exp(x)) = -log(sigmoid(-x)) = -log(1-sigmoid(x))
    y = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    if not deriv: return y

    dydx = sigmoid(x)
    def back(dy): return dy*dydx
    return y, back


def affine_transform(x,params,deriv=False):
    def back(dy):
        da = dy@x
        db = dy.sum()
        return np.append(da,db)

    a,b = params[:-1], params[-1]
    y = x if np.ndim(x) == 2 else x[:,None]
    y = y@a + b
    return (y,back) if deriv else y


def binary_logregr_obj(params,scores,labels,ptar,transf=affine_transform):
    sign = np.where(labels,-1,1)
    logitprior = logit(ptar)
    wtar = ptar/(np.sum(labels)*log2)
    wnon = (1-ptar)/(np.sum(1-labels)*log2)

    # forward
    x,back_transf = transf(scores,params,deriv=True)
    logits = sign*(logitprior + x)
    y = softplus(logits)
    xent = np.where(labels,wtar,wnon)@y

    # backprop
    dydx = sigmoid(logits)
    dllh = dydx*np.where(labels,-wtar,wnon)
    grad = back_transf(dllh)
    return xent,grad

def train_calibration(scores,labels,ptar):
    nsys = 1 if scores.ndim == 1 else scores.shape[1]

    obj = lambda p : binary_logregr_obj(p,scores,labels,ptar)
    params0 = np.append(np.ones(nsys,dtype=np.float64)/nsys,0.0)

    res = minimize(obj,params0,method="L-BFGS-B", jac=True, options={'disp':True})
    params = res.x

    cal = lambda s : affine_transform(np.atleast_1d(s),params,deriv=False)
    return cal,params

