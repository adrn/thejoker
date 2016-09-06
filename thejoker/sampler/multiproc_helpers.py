# Third-party
from astropy import log as logger
import numpy as np

# Project
from thejoker.sampler import tensor_vector_scalar, marginal_ln_likelihood

__all__ = ['marginal_ll_worker', 'get_good_samples', 'samples_to_orbital_params',
           'samples_to_orbital_params_worker']

def marginal_ll_worker(task):
    nl_p_chunk, data = task
    n_chunk = len(nl_p_chunk)

    ll = np.zeros(n_chunk)
    for i in range(n_chunk):
        try:
            ATA,p,chi2 = tensor_vector_scalar(nl_p_chunk[i], data)
            ll[i] = marginal_ln_likelihood(ATA, chi2)
        except:
            ll[i] = np.nan

    return ll

def get_good_samples(nonlinear_p, data, pool, chunk_size):
    n_total = len(nonlinear_p)
    tasks = [[nonlinear_p[i*chunk_size:(i+1)*chunk_size], data]
             for i in range(n_total//chunk_size+1)]

    results = pool.map(marginal_ll_worker, tasks)
    marg_ll = np.concatenate(results)

    assert len(marg_ll) == len(nonlinear_p)

    uu = np.random.uniform(size=len(nonlinear_p))
    good_samples_bool = uu < np.exp(marg_ll - marg_ll.max())
    good_samples = nonlinear_p[np.where(good_samples_bool)]
    n_good = len(good_samples)
    logger.info("{} good samples".format(n_good))

    return good_samples

def samples_to_orbital_params_worker(task):
    nl_p_chunk, data = task
    n_chunk = len(nl_p_chunk)

    pars = np.zeros((n_chunk, 6))
    for i in range(n_chunk):
        nl_p = nl_p_chunk[i]
        P, phi0, ecc, omega = nl_p
        ATA,p,_ = tensor_vector_scalar(nl_p, data)

        cov = np.linalg.inv(ATA)
        v0,asini = np.random.multivariate_normal(p, cov)

        if asini < 0:
            # logger.warning("Swapping asini")
            asini = np.abs(asini)
            omega += np.pi

        pars[i] = [P, asini, ecc, omega, phi0, v0]

    return pars

def samples_to_orbital_params(nonlinear_p, data, pool, chunk_size):
    n_total = len(nonlinear_p)
    tasks = [[nonlinear_p[i*chunk_size:(i+1)*chunk_size], data]
             for i in range(n_total//chunk_size+1)]
    orbit_pars = pool.map(samples_to_orbital_params_worker, tasks)
    orbit_pars = np.concatenate(orbit_pars)
    return orbit_pars.reshape(-1, orbit_pars.shape[-1])
