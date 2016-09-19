# Third-party
import numpy as np
from scipy.stats import beta, norm

# Project
from ..celestialmechanics import rv_from_elements

__all__ = ['pack_mcmc', 'unpack_mcmc', 'ln_likelihood', 'ln_prior']

def pack_mcmc(p):
    (P, asini, ecc, omega, phi0, v0, s2) = p

    return np.vstack((np.log(P),
                      np.sqrt(asini) * np.cos(phi0), np.sqrt(asini) * np.sin(phi0),
                      np.sqrt(ecc) * np.cos(omega), np.sqrt(ecc) * np.sin(omega),
                      v0,
                      np.log(s2)))

def unpack_mcmc(p):
    (ln_P,
     sqrtasini_cos_phi0, sqrtasini_sin_phi0,
     sqrte_cos_pomega, sqrte_sin_pomega,
     _v0,
     log_s2) = p

    return np.vstack((np.exp(ln_P),
                      (sqrtasini_cos_phi0**2 + sqrtasini_sin_phi0**2),
                      sqrte_cos_pomega**2 + sqrte_sin_pomega**2,
                      np.arctan2(sqrte_sin_pomega, sqrte_cos_pomega),
                      np.arctan2(sqrtasini_sin_phi0, sqrtasini_cos_phi0),
                      _v0,
                      log_s2))

def ln_likelihood(p, data):
    P, asini, ecc, omega, phi0, v0, log_s2 = unpack_mcmc(p)
    model_rv = rv_from_elements(data._t, P, asini, ecc, omega, phi0, v0)

    s2 = np.exp(log_s2)
    return -0.5 * (model_rv - data._rv)**2 / (1/data._ivar + s2)

def ln_prior(p):
    P, asini, ecc, omega, phi0, v0, log_s2 = unpack_mcmc(p)

    lnp = 0.

    if ecc < 0 or ecc > 1:
        return -np.inf

    lnp += beta.logpdf(ecc, 0.867, 3.03) # Kipping et al. 2013

    # TODO: do we need P_min, P_max here?

    # DFM's idea: wide, Gaussian prior in log(s^2)
    lnp += norm.logpdf(log_s2, 0, 1)

    return lnp

def ln_posterior(p, data):
    lnp = ln_prior(p)
    if np.isinf(lnp):
        return lnp

    lnl = ln_likelihood(p, data)
    lnprob = lnp + lnl.sum()

    if np.isnan(lnprob):
        return -np.inf

    return lnprob
