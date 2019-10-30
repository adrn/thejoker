# Third-party
from numpy import log, pi
from scipy.special import loggamma

__all__ = ['beta_logpdf']


def beta_logpdf(x, a, b):
    denom = loggamma(a) + loggamma(b)
    res = loggamma(a+b) + (a-1)*log(x) + (b-1)*log(1-x) - denom
    return res.real


def norm_logpdf(x, mu, sig):
    return -0.5 * (((x-mu) / sig)**2 + log(2*pi*sig**2))
