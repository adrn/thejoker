# Third-party
import numpy as np
from scipy.stats import beta, norm

# Project
from ..celestialmechanics import rv_from_elements
from .utils import get_ivar

__all__ = ['to_mcmc_params', 'from_mcmc_params', 'ln_likelihood', 'ln_prior']

log_2pi = np.log(2*np.pi)

def to_mcmc_params(p):
    r"""
    MCMC internal function.

    Transform from linear orbital parameter values to standard
    variables for MCMC sampling:

    .. math::

        \ln P \\
        \sqrt{K}\,\cos\phi_0, \sqrt{K}\,\sin\phi_0 \\
        \sqrt{e}\,\cos\omega, \sqrt{e}\,\sin\omega \\
        \ln s \\
        v_0,..., v_n

    Parameters
    ----------
    p : iterable
        A packed parameter vector containing the orbital parameters
        and long-term velocity trend parameters.

    """
    P, phi0, ecc, omega, s, K, *v_terms = p
    return np.vstack([np.log(P),
                      np.sqrt(K) * np.cos(phi0), np.sqrt(K) * np.sin(phi0),
                      np.sqrt(ecc) * np.cos(omega), np.sqrt(ecc) * np.sin(omega),
                      2*np.log(s)] + list(v_terms))

def from_mcmc_params(p):
    """
    MCMC internal function.

    Transform from the standard MCMC parameters to the linear
    values of the orbital parameters.

    Parameters
    ----------
    p : iterable
        A packed parameter vector containing the MCMC-transforemd
        versions of the orbital parameters and long-term velocity
        trend parameters.

    """
    (ln_P,
     sqrtK_cos_phi0, sqrtK_sin_phi0,
     sqrte_cos_omega, sqrte_sin_omega,
     log_s2, *v_terms) = p

    return np.vstack([np.exp(ln_P),
                      np.arctan2(sqrtK_sin_phi0, sqrtK_cos_phi0) % (2*np.pi),
                      sqrte_cos_omega**2 + sqrte_sin_omega**2,
                      np.arctan2(sqrte_sin_omega, sqrte_cos_omega) % (2*np.pi),
                      np.sqrt(np.exp(log_s2)),
                      (sqrtK_cos_phi0**2 + sqrtK_sin_phi0**2)] + v_terms)

def ln_likelihood(p, joker_params, data):
    P, phi0, ecc, omega, s, K, *v_terms = p

    # a little repeated code here...
    A = np.vander(data._t_bmjd, N=len(v_terms), increasing=True)
    trend = np.sum([A[:,i]**i * v_terms[i] for i in range(A.shape[1])], axis=0)

    model_rv = rv_from_elements(data._t_bmjd, P, K, ecc, omega, phi0) + trend
    ivar = get_ivar(data, s)

    return 0.5 * (-(model_rv - data.rv.value)**2 * ivar - log_2pi + np.log(ivar))

def ln_prior(p, joker_params):
    P, phi0, ecc, omega, s, K, *v_terms = p

    lnp = 0.

    # TODO: more repeated code here and hard-coded priors
    if ecc < 0 or ecc > 1:
        return -np.inf

    lnp += beta.logpdf(ecc, 0.867, 3.03) # Kipping et al. 2013

    # TODO: do we need P_min, P_max here?

    if not joker_params._fixed_jitter:
        # DFM's idea: wide, Gaussian prior in log(s^2)
        lnp += norm.logpdf(np.log(s), ) # TODO: put in hyper-parameters

    return lnp

def ln_posterior(mcmc_p, joker_params, data):
    if joker_params._fixed_jitter:
        mcmc_p = list(mcmc_p)
        mcmc_p.insert(5, -np.inf) # HACK: whoa, major hackage!

    p = from_mcmc_params(mcmc_p)

    lnp = ln_prior(p, joker_params)
    if np.isinf(lnp):
        return lnp

    lnl = ln_likelihood(p, joker_params, data)
    lnprob = lnp + lnl.sum()

    if np.isnan(lnprob):
        return -np.inf

    return lnprob
