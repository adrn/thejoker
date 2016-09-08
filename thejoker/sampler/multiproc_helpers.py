# Third-party
from astropy import log as logger
import h5py
import numpy as np

# Project
from thejoker.sampler import tensor_vector_scalar, marginal_ln_likelihood

__all__ = ['get_good_samples', 'samples_to_orbital_params']

def _marginal_ll_worker(task):
    """
    Compute the marginal log-likelihood integrated over the linear parameters
    v0 and asini (or velocity semi-amplitude, K). This is meant to be
    ``map``ped using one of the ``Pool`` classes by the functions below and
    is not supposed to be in the public API.

    Parameters
    ----------
    task : iterable
        A list containing the indices of samples to be operated on, the filename
        containing the prior samples, and the data.

    Returns
    -------
    ll : `numpy.ndarray`
        Array of log-likelihood values.
    """
    start_stop, filename, data = task

    # read a chunk of the prior samples
    with h5py.File(filename, 'r') as f:
        chunk = np.array(f['samples'][start_stop[0]:start_stop[1]])

    n_chunk = len(chunk)

    ll = np.zeros(n_chunk)
    for i in range(n_chunk):
        try:
            ATA,p,chi2 = tensor_vector_scalar(chunk[i], data)
            ll[i] = marginal_ln_likelihood(ATA, chi2)
        except:
            ll[i] = np.nan

    return ll

def get_good_samples(n_samples, filename, data, pool):
    """
    Return the indices of 'good' samples by computing the log-likelihood
    for ``n_samples`` prior samples and doing rejection sampling.

    For speed when parallelizing, this accepts a filename for an HDF5
    that contains the prior samples, splits up the samples based on the
    number of processes / MPI workers, and only distributes the indices
    for each worker to read.

    Parameters
    ----------
    n_samples : int
        The number of prior samples to use.
    filename : str
        Path to an HDF5 file comtaining the prior samples.
    data : `thejoker.data.RVData`
        An instance of ``RVData`` with the data we're modeling.
    pool : `thejoker.pool.GenericPool` or subclass
        An instance of a processing pool - must have a ``.map()`` method.

    Returns
    -------
    samples_idx : `numpy.ndarray`
        An array of integers for the prior samples that pass
        rejection sampling.

    """
    if pool.size > 0:
        # try chunking by the pool size
        chunk_size = n_samples // pool.size
        if chunk_size == 0:
            chunk_size = 1
    else:
        chunk_size = 1

    tasks = [[(i*chunk_size, (i+1)*chunk_size), filename, data]
             for i in range(n_samples//chunk_size+1)]

    results = pool.map(_marginal_ll_worker, tasks)
    marg_ll = np.concatenate(results)

    assert len(marg_ll) == n_samples

    uu = np.random.uniform(size=n_samples)
    good_samples_bool = uu < np.exp(marg_ll - marg_ll.max())
    good_samples_idx, = np.where(good_samples_bool)
    n_good = good_samples_bool.sum()
    logger.info("{} good samples after rejection sampling".format(n_good))

    return good_samples_idx

def _orbital_params_worker(task):
    """


    This is meant to be
        ``map``ped using one of the ``Pool`` classes by the functions below and
        is not supposed to be in the public API.
    """

    idx, filename, data, global_seed = task
    n_chunk = len(idx)

    if global_seed is not None:
        seed = int("{}{}".format(global_seed, idx[0]))
    else:
        seed = idx[0]
    np.random.seed(seed) # TODO: is this good enough?
    logger.debug("worker with chunk {} has seed {}".format(idx[0], seed))

    pars = np.zeros((n_chunk, 6))
    with h5py.File(filename, 'r') as f:
        for j,i in enumerate(idx): # these are the integer locations of the 'good' samples!
            nonlinear_p = f['samples'][i]
            P, phi0, ecc, omega = nonlinear_p
            ATA,p,_ = tensor_vector_scalar(nonlinear_p, data)

            cov = np.linalg.inv(ATA)
            v0,asini = np.random.multivariate_normal(p, cov)

            if asini < 0:
                # logger.warning("Swapping asini")
                asini = np.abs(asini)
                omega += np.pi
                omega = omega % (2*np.pi) # HACK: I think this is safe

            pars[j] = [P, asini, ecc, omega, phi0, v0]

    return pars

def samples_to_orbital_params(good_samples_idx, filename, data, pool, global_seed=None):
    """
    Generate the full set of orbital parameters for the 'good'
    samples that pass rejection sampling.

    For speed when parallelizing, this accepts a filename for an HDF5
    that contains the prior samples, splits up the samples based on the
    number of processes / MPI workers, and only distributes the indices
    for each worker to read.

    Parameters
    ----------
    good_samples_idx : array_like
        The array of indices for the 'good' samples in the prior
        samples cache file.
    filename : str
        Path to an HDF5 file comtaining the prior samples.
    data : `thejoker.data.RVData`
        An instance of ``RVData`` with the data we're modeling.
    pool : `thejoker.pool.GenericPool` or subclass
        An instance of a processing pool - must have a ``.map()`` method.
    global_seed : int (optional)
        The global level random number seed.

    """

    n_samples = len(good_samples_idx)

    if pool.size > 0:
        # try chunking by the pool size
        chunk_size = n_samples // pool.size
        if chunk_size == 0:
            chunk_size = 1
    else:
        chunk_size = 1

    # if chunk doesn't divide evenly into pool, the last chunk will be the remainder
    if n_samples%chunk_size:
        plus = 1
    else:
        plus = 0

    tasks = [[good_samples_idx[i*chunk_size:(i+1)*chunk_size], filename, data, global_seed]
             for i in range(n_samples//chunk_size+plus)]

    orbit_pars = pool.map(_orbital_params_worker, tasks)
    orbit_pars = np.concatenate(orbit_pars)

    return orbit_pars.reshape(-1, orbit_pars.shape[-1])
