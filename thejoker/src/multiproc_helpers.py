# Third-party
import h5py
import numpy as np
import tables as tb

# Project
from ..logging import logger
from ..data_helpers import _validate_data
from .fast_likelihood import CJokerHelper
from .utils import batch_tasks, read_batch

# __all__ = ['compute_likelihoods', 'get_good_sample_indices',
#            'sample_indices_to_full_samples']


def marginal_ln_likelihood_worker(task):
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
    (start, stop), task_id, prior_samples_file, joker_helper = task

    # read this batch of the prior samples
    batch = read_batch(prior_samples_file, joker_helper.packed_order,
                       start, stop, units=joker_helper.internal_units)

    # memoryview is returned
    ll = joker_helper.batch_marginal_ln_likelihood(batch)

    return np.array(ll)


def marginal_ln_likelihood_helper(data, prior, prior_samples_file, pool,
                                  n_batches=None):
    all_data, ids, trend_M = _validate_data(data, prior)
    joker_helper = CJokerHelper(data, prior, trend_M, None)

    with tb.open_file(prior_samples_file, mode='r') as f:
        n_samples = f.root.samples.shape[0]

    if n_batches is None:
        n_batches = max(1, pool.size)

    tasks = batch_tasks(n_samples, n_batches=n_batches,
                        args=(prior_samples_file, joker_helper))

    all_ll = []
    for res in pool.map(marginal_ln_likelihood_worker, tasks):
        all_ll.append(res)

    return np.concatenate(all_ll)


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

    (idx, batch_index, prior_cache_file, joker_helper, global_seed,
     return_logprobs) = task

    if global_seed is not None:
        seed = global_seed + batch_index
        rnd = np.random.RandomState(seed)
        logger.log(0, "worker with batch {} has seed {}".format(idx[0], seed))

    else:
        rnd = np.random.RandomState()
        logger.log(0, "worker with batch {} not seeded".format(idx[0]))

    # read a batch of the prior samples
    with h5py.File(prior_cache_file, 'r') as f:
        tmp = np.zeros(len(f['samples']), dtype=bool)
        tmp[idx] = True
        batch = np.array(f['samples'][tmp, :])

        if return_logprobs:
            ln_prior = np.array(f['ln_prior_probs'][tmp])

    batch = batch.astype(np.float64)

    pars = joker_helper.batch_get_posterior_samples(batch, rnd, return_logprobs)

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
    tasks = batch_tasks(n_samples, n_batches=n_batches, arr=good_samples_idx,
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
