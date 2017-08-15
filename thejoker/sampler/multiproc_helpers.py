# Third-party
import h5py
import numpy as np

# Project
from ..log import log
from .utils import get_ivar
from .likelihood import (design_matrix, tensor_vector_scalar,
                         marginal_ln_likelihood)
from .fast_likelihood import batch_marginal_ln_likelihood

__all__ = ['compute_likelihoods', 'get_good_sample_indices',
           'sample_indices_to_full_samples']

def chunk_tasks(N, pool, arr=None, args=None, start_idx=0):
    if args is None:
        args = []

    args = list(args)

    tasks = []
    if pool.size > 0 and N > pool.size:
        # chunk by the pool size
        base_chunk_size = N // pool.size
        rmdr = N % pool.size

        i1 = start_idx
        for i in range(pool.size):
            i2 = i1 + base_chunk_size
            if i < rmdr:
                i2 += 1

            if arr is None: # store indices
                tasks.append([(i1, i2), i1] + args)

            else: # store sliced array
                tasks.append([arr[i1:i2], i1] + args)

            i1 = i2

    else:
        if arr is None: # store indices
            tasks.append([(start_idx,N+start_idx), start_idx] + args)

        else: # store sliced array
            tasks.append([arr[start_idx:N+start_idx], start_idx] + args)

    return tasks

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
    start_stop, chunk_index, prior_cache_file, data, jparams = task

    # read a chunk of the prior samples
    with h5py.File(prior_cache_file, 'r') as f:
        chunk = np.array(f['samples'][start_stop[0]:start_stop[1]])

    chunk = chunk.astype(np.float64)

    # memoryview is returned
    ll = batch_marginal_ln_likelihood(chunk, data, jparams)
    return np.array(ll)

def compute_likelihoods(n_prior_samples, prior_cache_file, start_idx, data,
                        joker_params, pool):
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
    start_idx : int
        Index to start reading prior samples from in the prior cache file.
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
    args = [prior_cache_file, data, joker_params]
    tasks = chunk_tasks(n_prior_samples, pool=pool, args=args,
                        start_idx=start_idx)

    results = [r for r in pool.map(_marginal_ll_worker, tasks)]
    marg_ll = np.concatenate(results)

    assert len(marg_ll) == n_prior_samples

    return marg_ll

def get_good_sample_indices(marg_ll, seed=None):
    """Return the indices of 'good' samples from pre-computed values of the
    log-likelihood.

    Parameters
    ----------
    marg_ll : array_like
        Array of marginal log-likelihood values.
    seed : int (optional)
        Random number seed for uniform samples to use in rejection sampling.

    Returns
    -------
    samples_idx : `numpy.ndarray`
        An array of integers for the prior samples that pass
        rejection sampling.

    """

    # rejection sample using the marginal likelihood
    rnd = np.random.RandomState(seed)
    uu = rnd.uniform(size=len(marg_ll))
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

    (idx, chunk_index, prior_cache_file, data, joker_params, global_seed,
     return_logprobs) = task
    n_chunk = len(idx)

    if global_seed is not None:
        seed = global_seed + chunk_index
        rnd = np.random.RandomState(seed)
        log.debug("worker with chunk {} has seed {}".format(idx[0], seed))

    else:
        rnd = np.random.RandomState()
        log.debug("worker with chunk {} not seeded".format(idx[0]))

    pars = np.zeros((n_chunk, joker_params.num_params + 2*int(return_logprobs)))
    with h5py.File(prior_cache_file, 'r') as f:
        # idx are the integer locations of the 'good' samples!
        for j,i in enumerate(idx):
            nonlinear_p = f['samples'][i]
            P, phi0, ecc, omega, s = np.array(nonlinear_p).astype(np.float64)

            ivar = get_ivar(data, s)
            A = design_matrix(nonlinear_p, data, joker_params)
            ATA, p, chi2 = tensor_vector_scalar(A, ivar, data.rv.value)

            cov = np.linalg.inv(ATA)
            K, *v_terms = rnd.multivariate_normal(p, cov)

            if K < 0:
                # log.warning("Swapping K")
                K = np.abs(K)
                omega += np.pi
                omega = omega % (2*np.pi) # HACK: I think this is safe

            row = [P, phi0, ecc, omega, s, K] + v_terms
            if return_logprobs:
                ln_prior = f['ln_prior_probs'][i]
                ln_like = marginal_ln_likelihood(nonlinear_p, data,
                                                 joker_params,
                                                 tvsi=(ATA, p, chi2, ivar))
                row = row + [ln_prior, ln_like]

            pars[j] = row

    return pars

def sample_indices_to_full_samples(good_samples_idx, prior_cache_file, data,
                                   joker_params, pool, global_seed=None,
                                   return_logprobs=False):
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
    return_logprobs : bool (optional)
        Also return the log-probabilities of the prior samples.

    """

    n_samples = len(good_samples_idx)
    args = [prior_cache_file, data, joker_params, global_seed, return_logprobs]
    tasks = chunk_tasks(n_samples, arr=good_samples_idx, pool=pool, args=args)

    samples = [r for r in pool.map(_sample_vector_worker, tasks)]
    samples = np.concatenate(samples)

    assert len(samples) == n_samples
    samples = samples.reshape(-1, samples.shape[-1])

    if return_logprobs:
        # samples, ln(prior), ln(likelihood)
        return samples[:, :-2], samples[:, -2], samples[:, -1]

    else:
        return samples
