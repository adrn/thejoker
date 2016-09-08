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

def get_good_samples(n_samples, filename, data, pool, chunk_size):
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
    chunk_size : int
        The chunk size to split the prior samples in to based on the
        number of processes / workers.

    Returns
    -------
    samples_idx : `numpy.ndarray`
        An array of integers for the prior samples that pass
        rejection sampling.

    """
    tasks = [[(i*chunk_size, (i+1)*chunk_size), filename, data]
             for i in range(n_samples//chunk_size+1)]

    results = pool.map(_marginal_ll_worker, tasks)
    marg_ll = np.concatenate(results)

    assert len(marg_ll) == n_samples

    uu = np.random.uniform(size=n_samples)
    good_samples_bool = uu < np.exp(marg_ll - marg_ll.max())
    good_samples_idx = np.where(good_samples_bool)
    n_good = good_samples_bool.sum()
    logger.info("{} good samples after rejection sampling".format(n_good))

    return good_samples_idx

def _orbital_params_worker(task):
    """

    This is meant to be
        ``map``ped using one of the ``Pool`` classes by the functions below and
        is not supposed to be in the public API.
    """

    # TODO: here I need to appropriately set the random number generator seed
    # np.random.seed(??)

    idx, filename, data = task
    n_chunk = len(idx)

    # TODO: gotta think about how to do this -- now we need to chunk on the GOOD
    #   indices from the prior samples file -- now `idx` is an array of indices
    #   to read from the prior samples file?

    pars = np.zeros((n_chunk, 6))
    with h5py.File(filename, 'r') as f:
        for i in idx: # these are the integer locations of the 'good' samples!
            nonlinear_p = f['samples'][i][:]
            P, phi0, ecc, omega = nonlinear_p
            ATA,p,_ = tensor_vector_scalar(nonlinear_p, data)

            cov = np.linalg.inv(ATA)
            v0,asini = np.random.multivariate_normal(p, cov)

            if asini < 0:
                # logger.warning("Swapping asini")
                asini = np.abs(asini)
                omega += np.pi
                omega % (2*np.pi) # HACK: I think this is safe

            pars[i] = [P, asini, ecc, omega, phi0, v0]

    return pars

def samples_to_orbital_params(good_samples_idx, filename, data, pool, chunk_size):
    """

    For speed when parallelizing, this accepts a filename for an HDF5
    that contains the prior samples, splits up the samples based on the
    number of processes / MPI workers, and only distributes the indices
    for each worker to read.
    """

    n_total = len(nonlinear_p)
    tasks = [[nonlinear_p[i*chunk_size:(i+1)*chunk_size], data]
             for i in range(n_total//chunk_size+1)]
    orbit_pars = pool.map(samples_to_orbital_params_worker, tasks)
    orbit_pars = np.concatenate(orbit_pars)
    return orbit_pars.reshape(-1, orbit_pars.shape[-1])
