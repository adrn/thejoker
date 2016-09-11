# Third-party
import numpy as np
from scipy.stats import beta

# Project
from ..celestialmechanics import rv_from_elements

__all__ = ['pack_mcmc', 'unpack_mcmc', 'ln_likelihood', 'ln_prior']

def pack_mcmc(p):
    (P, asini, ecc, omega, phi0, v0) = p

    return np.vstack((np.log(P),
                      np.sqrt(asini) * np.cos(phi0), np.sqrt(asini) * np.sin(phi0),
                      np.sqrt(ecc) * np.cos(omega), np.sqrt(ecc) * np.sin(omega),
                      v0))

def unpack_mcmc(p):
    (ln_P,
     sqrtasini_cos_phi0, sqrtasini_sin_phi0,
     sqrte_cos_pomega, sqrte_sin_pomega,
     _v0) = p

    return np.vstack((np.exp(ln_P),
                      (sqrtasini_cos_phi0**2 + sqrtasini_sin_phi0**2),
                      sqrte_cos_pomega**2 + sqrte_sin_pomega**2,
                      np.arctan2(sqrte_sin_pomega, sqrte_cos_pomega),
                      np.arctan2(sqrtasini_sin_phi0, sqrtasini_cos_phi0),
                      _v0))

def ln_likelihood(p, data):
    P, asini, ecc, omega, phi0, v0 = unpack_mcmc(p)
    model_rv = rv_from_elements(data._t, P, asini, ecc, omega, phi0, v0)
    return -0.5 * (model_rv - data._rv)**2 * data._ivar

def ln_prior(p):
    P, asini, ecc, omega, phi0, v0 = unpack_mcmc(p)

    lnp = 0.

    if ecc < 0 or ecc > 1:
        return -np.inf

    lnp += beta.logpdf(ecc, 0.867, 3.03) # Kipping et al. 2013

    # TODO: do we need P_min, P_max here?

    return lnp

def ln_posterior(p, data):
    lnp = ln_prior(p)
    if np.isinf(lnp):
        return lnp

    lnl = ln_likelihood(p, data)

    return lnp + lnl.sum()

