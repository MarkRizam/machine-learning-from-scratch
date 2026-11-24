"""Educational regression, binary classification and clustering metrics."""
import numpy as np


def _pair(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.ndim != 1 or not len(y) or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Expected matching nonempty finite vectors')
    return y, p


def mae(y, p):
    y,p = _pair(y,p)
    return float(np.mean(np.abs(y-p)))


def mse(y,p):
    y,p = _pair(y,p)
    return float(np.mean((y-p)**2))


def rmse(y,p):
    return float(np.sqrt(mse(y,p)))


def r2(y,p):
    y,p = _pair(y,p)
    if len(y)<2: raise ValueError('R2 requires at least two samples')
    sse, sst = np.sum((y-p)**2), np.sum((y-y.mean())**2)
    return float(1-sse/sst) if sst else float(sse==0)


def accuracy(y,p):
    y,p = _pair(y,p)
    return float(np.mean(y==p))


def confusion_matrix(y,p):
    y,p = _pair(y,p)
    if not np.isin(y,[0,1]).all() or not np.isin(p,[0,1]).all():
        raise ValueError('Binary labels only')
    return np.array([[np.sum((y==a)&(p==b)) for b in [0,1]] for a in [0,1]])


def precision(y,p):
    tn,fp,fn,tp = confusion_matrix(y,p).ravel()
    return float(tp/(tp+fp)) if tp+fp else 0.0


def recall(y,p):
    tn,fp,fn,tp = confusion_matrix(y,p).ravel()
    return float(tp/(tp+fn)) if tp+fn else 0.0


def f1(y,p):
    tn,fp,fn,tp = confusion_matrix(y,p).ravel()
    return float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0


def log_loss(y,p):
    y,p = _pair(y,p)
    if not np.isin(y,[0,1]).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected binary labels and probabilities')
    eps=np.finfo(float).eps
    p=np.clip(p,eps,1-eps)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def inertia(x,centers,labels):
    x,centers,labels=np.asarray(x,dtype=float),np.asarray(centers,dtype=float),np.asarray(labels)
    if x.ndim!=2 or centers.ndim!=2 or not all(x.shape) or not all(centers.shape) or x.shape[1]!=centers.shape[1] or not np.isfinite(x).all() or not np.isfinite(centers).all() or labels.shape!=(len(x),) or labels.dtype.kind not in 'iu' or np.any((labels<0)|(labels>=len(centers))):
        raise ValueError('Invalid clustering inputs')
    return float(np.sum((x-centers[labels])**2))
