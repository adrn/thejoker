# Third-party
import h5py
import numpy as np
import tables as tb

# Project
from ..logging import logger
from .fast_likelihood import CJokerHelper

__all__ = ['compute_likelihoods', 'get_good_sample_indices',
           'sample_indices_to_full_samples']


def chunk_tasks(n_tasks, n_batches, arr=None, args=None, start_idx=0):
    """Split the tasks into some number of batches to sent out to MPI workers.

    Parameters
    ----------
    n_tasks : int
        The total number of tasks to divide.
    n_batches : int
        The number of batches to split the tasks into. Often, you may want to do
        ``n_batches=pool.size`` for equal sharing amongst MPI workers.
    arr : iterable (optional)
        Instead of returning indices that specify the batches, you can also
        directly split an array into batches.
    args : iterable (optional)
        Other arguments to add to each task.
    start_idx : int (optional)
        What index in the tasks to start from?

    """
    if args is None:
        args = []
    args = list(args)

    tasks = []
    if n_batches > 0 and n_tasks > n_batches:
        # chunk by the number of batches, often the pool size
        base_chunk_size = n_tasks // n_batches
        rmdr = n_tasks % n_batches

        i1 = start_idx
        for i in range(n_batches):
            i2 = i1 + base_chunk_size
            if i < rmdr:
                i2 += 1

            if arr is None:  # store indices
                tasks.append([(i1, i2), i1] + args)

            else:  # store sliced array
                tasks.append([arr[i1:i2], i1] + args)

            i1 = i2

    else:
        if arr is None:  # store indices
            tasks.append([(start_idx, n_tasks+start_idx), start_idx] + args)

        else:  # store sliced array
            tasks.append([arr[start_idx:n_tasks+start_idx], start_idx] + args)

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
    start_stop, chunk_index, prior_cache_file, joker_helper = task

    # read a chunk of the prior samples
    with h5py.File(prior_cache_file, 'r') as f:
        chunk = np.array(f['samples'][start_stop[0]:start_stop[1]])

    chunk = chunk.astype(np.float64)

    # memoryview is returned
    ll = joker_helper.batch_marginal_ln_likelihood(chunk)
    return np.array(ll)


def compute_likelihoods(n_prior_samples, prior_cache_file, start_idx, data,
                        prior, trend_M, pool, n_batches=None):
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
    n_batches : int (optional)
        How many batches to divide the work into. Defaults to ``pool.size``.

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
    # TODO: get trend_M from prior
    joker_helper = CJokerHelper(data, prior, trend_M)

    args = [prior_cache_file, joker_helper]
    if n_batches is None:
        n_batches = pool.size
    tasks = chunk_tasks(n_prior_samples, n_batches=n_batches, args=args,
                        start_idx=start_idx)

    results = [r for r in pool.map(_marginal_ll_worker, tasks)]
    marg_ll = np.concatenate(results)

    if len(marg_ll) != n_prior_samples:
        raise RuntimeError("Unexpected failure: number of likelihoods "
                           "returned from workers does not match number sent "
                           "out to workers.")

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


def _sample_vector_worker(task):
    """
    This is meant to be
        ``map``ped using one of the ``Pool`` classes by the functions below and
        is not supposed to be in the public API.
    """

    (idx, chunk_index, prior_cache_file, joker_helper, global_seed,
     return_logprobs) = task

    if global_seed is not None:
        seed = global_seed + chunk_index
        rnd = np.random.RandomState(seed)
        logger.log(0, "worker with chunk {} has seed {}".format(idx[0], seed))

    else:
        rnd = np.random.RandomState()
        logger.log(0, "worker with chunk {} not seeded".format(idx[0]))

    # read a chunk of the prior samples
    with h5py.File(prior_cache_file, 'r') as f:
        tmp = np.zeros(len(f['samples']), dtype=bool)
        tmp[idx] = True
        chunk = np.array(f['samples'][tmp, :])

        if return_logprobs:
            ln_prior = np.array(f['ln_prior_probs'][tmp])

    chunk = chunk.astype(np.float64)

    pars = joker_helper.batch_get_posterior_samples(chunk, rnd, return_logprobs)

    if return_logprobs:
        pars = np.hstack((pars[:, :-1], ln_prior[:, None], pars[:, -1:]))
    return pars


def sample_indices_to_full_samples(good_samples_idx, prior_cache_file, data,
                                   prior, trend_M, max_n_samples, pool,
                                   global_seed=None, return_logprobs=False,
                                   n_batches=None):
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
    max_n_samples : int
        The maximum number of samples to return.
    pool : `~schwimmbad.pool.BasePool` or subclass
        An instance of a processing pool - must have a ``.map()`` method.
    global_seed : int (optional)
        The global level random number seed.
    return_logprobs : bool (optional)
        Also return the log-probabilities of the prior samples.
    n_batches : int (optional)
        How many batches to divide the work into. Defaults to ``pool.size``.

    """

    # TODO: get trend_M from prior
    joker_helper = CJokerHelper(data, prior, trend_M)

    good_samples_idx = good_samples_idx[:max_n_samples]
    n_samples = len(good_samples_idx)
    args = [prior_cache_file, joker_helper, global_seed, return_logprobs]
    if n_batches is None:
        n_batches = pool.size
    tasks = chunk_tasks(n_samples, n_batches=n_batches, arr=good_samples_idx,
                        args=args)

    samples = [r for r in pool.map(_sample_vector_worker, tasks)]
    samples = np.concatenate(samples)

    assert len(samples) == n_samples
    samples = samples.reshape(-1, samples.shape[-1])

    if return_logprobs:
        # samples, ln(prior), ln(likelihood)
        return samples[:, :-2], samples[:, -2], samples[:, -1]

    else:
        return samples


# -----------------------------------------------------------------------------
from ..data_helpers import _validate_data


def read_chunk(prior_samples_file, start, stop):
    # TODO: units and shit!

    # TODO: get order from elsewhere...like set on helper?
    names = ['P', 'e', 'omega', 'M0']

    chunk = np.zeros((stop-start, len(names)))
    with tb.open_file(prior_samples_file, mode='r') as f:
        for i, name in enumerate(names):
            chunk[:, i] = f.root.samples.read(start, stop, field=name)

    return chunk


def marginal_ln_likelihood_worker(task):
    start_stop, task_id, prior_samples_file, joker_helper = task

    # read this chunk of the prior samples
    chunk = read_chunk(prior_samples_file, *start_stop)

    # memoryview is returned
    ll = joker_helper.batch_marginal_ln_likelihood(chunk)

    return np.array(ll)


def marginal_ln_likelihood_helper(data, prior, prior_samples_file, pool,
                                  n_batches=None):
    all_data, ids, trend_M = _validate_data(data, prior)
    joker_helper = CJokerHelper(data, prior, trend_M, None)

    with tb.open_file(prior_samples_file, mode='r') as f:
        n_samples = f.root.samples.shape[0]

    if n_batches is None:
        n_batches = max(1, pool.size)

    tasks = chunk_tasks(n_samples, n_batches=n_batches,
                        args=(prior_samples_file, joker_helper))

    all_ll = []
    for res in pool.map(marginal_ln_likelihood_worker, tasks):
        all_ll.append(res)

    return np.concatenate(all_ll)
