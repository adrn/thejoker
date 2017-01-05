# Third-party
import numpy as np
from scipy.stats import beta, norm

# Project
from ..celestialmechanics import rv_from_elements

__all__ = ['pack_mcmc', 'unpack_mcmc', 'ln_likelihood', 'ln_prior']

# TODO: so many hacks here with jitter

def pack_mcmc(p, fixed_jitter=False):
    if fixed_jitter:
        (P, ecc, omega, phi0, K, v0) = p
        s = 0. * P
    else:
        (P, ecc, omega, phi0, s, K, v0) = p

    return np.vstack((np.log(P),
                      np.sqrt(K) * np.cos(phi0), np.sqrt(K) * np.sin(phi0),
                      np.sqrt(ecc) * np.cos(omega), np.sqrt(ecc) * np.sin(omega),
                      v0,
                      2*np.log(s)))

def unpack_mcmc(p):
    (ln_P,
     sqrtK_cos_phi0, sqrtK_sin_phi0,
     sqrte_cos_pomega, sqrte_sin_pomega,
     _v0,
     log_s2) = p

    return np.vstack((np.exp(ln_P),
                      sqrte_cos_pomega**2 + sqrte_sin_pomega**2,
                      np.arctan2(sqrte_sin_pomega, sqrte_cos_pomega),
                      np.arctan2(sqrtK_sin_phi0, sqrtK_cos_phi0),
                      np.sqrt(np.exp(log_s2)),
                      (sqrtK_cos_phi0**2 + sqrtK_sin_phi0**2),
                      _v0))

def ln_likelihood(p, data):
    P, ecc, omega, phi0, s, K, v0 = unpack_mcmc(p)
    model_rv = rv_from_elements(data._t, P, K, ecc, omega, phi0) + v0
    return -0.5 * (model_rv - data._rv)**2 / (1/data._ivar + s**2)

def ln_prior(p, fixed_jitter=False):
    P, ecc, omega, phi0, s, K, v0 = unpack_mcmc(p)

    lnp = 0.

    if ecc < 0 or ecc > 1:
        return -np.inf

    lnp += beta.logpdf(ecc, 0.867, 3.03) # Kipping et al. 2013

    # TODO: do we need P_min, P_max here?

    if not fixed_jitter:
        # DFM's idea: wide, Gaussian prior in log(s^2)
        lnp += norm.logpdf(2*np.log(s), 0, 4)

    return lnp

def ln_posterior(p, data, fixed_jitter=False):
    if fixed_jitter:
        p = list(p) + [0.]

    lnp = ln_prior(p)
    if np.isinf(lnp):
        return lnp

    lnl = ln_likelihood(p, data)
    lnprob = lnp + lnl.sum()

    if np.isnan(lnprob):
        return -np.inf

    return lnprob
