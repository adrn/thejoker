# Third-party
import astropy.units as u
import numpy as np
from scipy.stats import beta, norm

# Project
from ..celestialmechanics import rv_from_elements
from .utils import get_ivar

__all__ = ['to_mcmc_params', 'from_mcmc_params',
           'pack_samples', 'pack_samples_mcmc', 'unpack_samples', 'unpack_samples_mcmc',
           'ln_likelihood', 'ln_prior', 'ln_posterior']

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

def pack_samples(samples, params, data):
    """
    """
    if 'jitter' in samples:
        jitter = samples['jitter'].to(data.rv.unit).value
    else:
        jitter = np.zeros_like(samples['P'].value)

    arr = [samples['P'].to(u.day).value,
           samples['phi0'].to(u.radian).value,
           np.asarray(samples['ecc']),
           samples['omega'].to(u.radian).value,
           jitter,
           samples['K'].to(data.rv.unit).value]
    arr = arr + [samples['v{}'.format(i)].to(data.rv.unit/u.day**i).value
                 for i in range(params.trend.n_terms)]
    return np.array(arr).T

def pack_samples_mcmc(samples, params, data):
    """
    """
    samples_vec = pack_samples(samples, params, data)
    samples_mcmc = to_mcmc_params(samples_vec.T)

    if params._fixed_jitter:
        samples_mcmc = np.delete(samples_mcmc, 5, axis=0)

    return np.array(samples_mcmc).T

def unpack_samples(samples_arr, params, data):
    samples = dict()
    samples['P'] = samples_arr.T[0] * u.day
    samples['phi0'] = samples_arr.T[1] * u.radian
    samples['ecc'] = samples_arr.T[2] * u.one
    samples['omega'] = samples_arr.T[3] * u.radian

    if not params._fixed_jitter:
        samples['jitter'] = samples_arr.T[4] * data.rv.unit
        shift = 1
    else:
        samples['jitter'] = np.zeros_like(samples_arr.T[0]) * data.rv.unit
        shift = 0

    samples['K'] = samples_arr.T[4+shift] * data.rv.unit

    for i in range(params.trend.n_terms):
        samples['v{}'.format(i)] = samples_arr.T[5+shift+i] * data.rv.unit/u.day**i

    return samples

def unpack_samples_mcmc(samples_arr, params, data):
    samples_arr = from_mcmc_params(samples_arr.T).T
    return unpack_samples(samples_arr, params, data)

def ln_likelihood(p, joker_params, data):
    P, phi0, ecc, omega, s, K, *v_terms = p

    # a little repeated code here...

    # phi0 now is implicitly relative to data.t_offset, not mjd=0
    t = data._t_bmjd
    zdot = rv_from_elements(times=t, P=P, K=1., e=ecc,
                            omega=omega, phi0=phi0,
                            anomaly_tol=joker_params.anomaly_tol)

    # TODO: right now, we only support a single, global velocity trend!
    A1 = np.vander(t, N=joker_params.trend.n_terms, increasing=True)
    A = np.hstack((zdot[:,None], A1))
    p = np.array([K] + v_terms)
    ivar = get_ivar(data, s)

    dy = A.dot(p) - data.rv.value

    return 0.5 * (-dy**2 * ivar - log_2pi + np.log(ivar))

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
        # lnp += norm.logpdf(np.log(s), ) # TODO: put in hyper-parameters
        # TODO:
        pass

    return lnp

def ln_posterior(mcmc_p, joker_params, data):
    if joker_params._fixed_jitter:
        mcmc_p = list(mcmc_p)
        mcmc_p.insert(5, -np.inf) # HACK: whoa, major hackage!

    p = from_mcmc_params(mcmc_p).reshape(len(mcmc_p))

    lnp = ln_prior(p, joker_params)
    if np.isinf(lnp):
        return lnp

    lnl = ln_likelihood(p, joker_params, data)
    lnprob = lnp + lnl.sum()

    if np.isnan(lnprob):
        return -np.inf

    return lnprob
