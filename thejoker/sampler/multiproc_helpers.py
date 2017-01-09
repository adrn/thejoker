# Third-party
import h5py
import numpy as np

# Project
from ..log import log
from .utils import get_ivar
from .likelihood import design_matrix, tensor_vector_scalar, marginal_ln_likelihood

__all__ = ['get_good_sample_indices', 'sample_indices_to_full_samples']

def _marginal_ll_worker(task):
    """
    Compute the marginal log-likelihood, i.e. the likelihood integrated over
    the linear parameters. This is meant to be ``map``ped using a processing
    pool` within the functions below and is not supposed to be in the
    public API.

    Parameters
    ----------
    task : iterable
        An array containing the indices of samples to be operated on, the
        filename containing the prior samples, and the data.

    Returns
    -------
    ll : `numpy.ndarray`
        Array of log-likelihood values.

    """
    start_stop, prior_cache_file, data, jparams = task

    # read a chunk of the prior samples
    with h5py.File(prior_cache_file, 'r') as f:
        chunk = np.array(f['samples'][start_stop[0]:start_stop[1]])

    n_chunk = len(chunk)

    ll = np.zeros(n_chunk)
    for i in range(n_chunk):
        try:
            ll[i] = marginal_ln_likelihood(chunk[i], data, jparams)
        except Exception as e:
            log.error(e)
            ll[i] = np.nan

    return ll

def get_good_sample_indices(n_prior_samples, prior_cache_file, data, joker_params, pool):
    """
    Return the indices of 'good' samples by computing the log-likelihood
    for ``n_prior_samples`` prior samples and doing rejection sampling.

    For speed when parallelizing, this accepts a filename for an HDF5
    that contains the prior samples, splits up the samples based on the
    number of processes / MPI workers, and only distributes the indices
    for each worker to read. This limits the amount of data that needs
    to be passed around.

    Parameters
    ----------
    n_prior_samples : int
        The number of prior samples to use.
    prior_cache_file : str
        Path to an HDF5 file containing the prior samples.
    data : `~thejoker.data.RVData`
        An instance of ``RVData`` with the data we're modeling.
    joker_params : `~thejoker.sampler.params.JokerParams`
        A specification of the parameters to use.
    pool : `~schwimmbad.pool.BasePool` or subclass
        An instance of a processing pool - must have a ``.map()`` method.

    Returns
    -------
    samples_idx : `numpy.ndarray`
        An array of integers for the prior samples that pass
        rejection sampling.

    TODO
    ----
    - The structure of this function is ok for most cluster-like machines
        up to ~2**28 samples or so. Then it becomes an issue that I keep
        the likelihood values in memory. We may want to be able to cache
        the likelihood values instead?

    """
    if pool.size > 0:
        # try chunking by the pool size
        chunk_size = n_prior_samples // pool.size
        if chunk_size == 0:
            chunk_size = 1
    else:
        chunk_size = 1

    tasks = [[(i*chunk_size, (i+1)*chunk_size), prior_cache_file, data, joker_params]
             for i in range(n_prior_samples//chunk_size+1)]

    results = [r for r in pool.map(_marginal_ll_worker, tasks)]
    marg_ll = np.concatenate(results)

    assert len(marg_ll) == n_prior_samples

    # rejection sample using the marginal likelihood
    uu = np.random.uniform(size=n_prior_samples)
    good_samples_bool = uu < np.exp(marg_ll - marg_ll.max())
    good_samples_idx, = np.where(good_samples_bool)

    return good_samples_idx

# ----------------------------------------------------------------------------

def _sample_vector_worker(task):
    """
    This is meant to be
        ``map``ped using one of the ``Pool`` classes by the functions below and
        is not supposed to be in the public API.
    """

    idx, prior_cache_file, data, joker_params, global_seed, chunk_index = task
    n_chunk = len(idx)

    if global_seed is not None:
        seed = int("{}{}".format(global_seed, chunk_index))
        rnd = np.random.RandomState(seed)
        log.debug("worker with chunk {} has seed {}".format(idx[0], seed))

    else:
        rnd = np.random.RandomState()
        log.debug("worker with chunk {} not seeded")

    pars = np.zeros((n_chunk, joker_params.num_params))
    with h5py.File(prior_cache_file, 'r') as f:
        for j,i in enumerate(idx): # these are the integer locations of the 'good' samples!
            nonlinear_p = f['samples'][i]
            P, phi0, ecc, omega, s = nonlinear_p

            ivar = get_ivar(data, s)
            A = design_matrix(nonlinear_p, data, joker_params)
            ATA,p,_ = tensor_vector_scalar(A, ivar, data.rv.value)

            cov = np.linalg.inv(ATA)
            K, *v_terms = rnd.multivariate_normal(p, cov)

            if K < 0:
                # log.warning("Swapping K")
                K = np.abs(K)
                omega += np.pi
                omega = omega % (2*np.pi) # HACK: I think this is safe

            pars[j] = [P, ecc, omega, phi0, s, K] + v_terms

    return pars

def sample_indices_to_full_samples(good_samples_idx, prior_cache_file, data, joker_params,
                                   pool, global_seed=None):
    """
    Generate the full set of parameter values (linear + non-linear) for
    the nonlinear parameter prior samples that pass the rejection sampling.

    For speed when parallelizing, this accepts a filename for an HDF5
    that contains the prior samples, splits up the samples based on the
    number of processes / MPI workers, and only distributes the indices
    for each worker to read.

    Parameters
    ----------
    good_samples_idx : array_like
        The array of indices for the 'good' samples in the prior
        samples cache file.
    prior_cache_file : str
        Path to an HDF5 file containing the prior samples.
    data : `thejoker.data.RVData`
        An instance of ``RVData`` with the data we're modeling.
    joker_params : `~thejoker.sampler.params.JokerParams`
        A specification of the parameters to use.
    pool : `~schwimmbad.pool.BasePool` or subclass
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
    if n_samples % chunk_size:
        plus = 1
    else:
        plus = 0

    tasks = [[good_samples_idx[i*chunk_size:(i+1)*chunk_size], prior_cache_file,
              data, joker_params, global_seed, i]
             for i in range(n_samples//chunk_size+plus)]

    samples = [r for r in pool.map(_sample_vector_worker, tasks)]
    samples = np.concatenate(samples)

    return samples.reshape(-1, samples.shape[-1])
