# Third-party
import numpy as np
import tables as tb

# Project
from ..logging import logger
from ..samples import JokerSamples
from .utils import batch_tasks, read_batch


def run_worker(worker, pool, prior_samples_file, task_args=(), n_batches=None,
               n_prior_samples=None, samples_idx=None):

    with tb.open_file(prior_samples_file, mode='r') as f:
        n_samples = f.root[JokerSamples._hdf5_path].shape[0]

    if n_prior_samples is not None and samples_idx is not None:
        raise ValueError("TODO: dont specify both")

    elif samples_idx is not None:
        n_samples = len(samples_idx)

    elif n_prior_samples is not None:
        n_samples = int(n_prior_samples)

    if n_batches is None:
        n_batches = max(1, pool.size)

    if samples_idx is not None:
        tasks = batch_tasks(n_samples, n_batches=n_batches, arr=samples_idx,
                            args=task_args)
    else:
        tasks = batch_tasks(n_samples, n_batches=n_batches, args=task_args)

    results = []
    for res in pool.map(worker, tasks):
        results.append(res)

    return results


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
    slice_or_idx, task_id, prior_samples_file, joker_helper = task

    # Read the batch of prior samples
    batch = read_batch(prior_samples_file, joker_helper.packed_order,
                       slice_or_idx, units=joker_helper.internal_units)

    # memoryview is returned
    ll = joker_helper.batch_marginal_ln_likelihood(batch)

    return np.array(ll)


def marginal_ln_likelihood_helper(joker_helper, prior_samples_file, pool,
                                  n_batches=None, n_prior_samples=None,
                                  samples_idx=None):

    task_args = (prior_samples_file,
                 joker_helper)
    results = run_worker(marginal_ln_likelihood_worker, pool,
                         prior_samples_file,
                         task_args=task_args, n_batches=n_batches,
                         samples_idx=samples_idx)
    return np.concatenate(results)


def make_full_samples_worker(task):
    (slice_or_idx,
     task_id,
     prior_samples_file,
     joker_helper,
     n_linear_samples,
     global_random_state) = task

    random_state = np.random.RandomState()
    random_state.set_state(global_random_state.get_state())
    random_state.seed(task_id)  # TODO: is this safe?

    # Read the batch of prior samples
    batch = read_batch(prior_samples_file,
                       columns=joker_helper.packed_order,
                       slice_or_idx=slice_or_idx,
                       units=joker_helper.internal_units)

    raw_samples, _ = joker_helper.batch_get_posterior_samples(batch,
                                                              n_linear_samples,
                                                              random_state)

    return raw_samples


def make_full_samples(joker_helper, prior_samples_file, pool, random_state,
                      samples_idx, n_linear_samples=1, n_batches=None):

    task_args = (prior_samples_file,
                 joker_helper,
                 n_linear_samples,
                 random_state)
    results = run_worker(make_full_samples_worker, pool, prior_samples_file,
                         task_args=task_args, n_batches=n_batches,
                         samples_idx=samples_idx)

    # Concatenate all of the raw samples arrays
    raw_samples = np.concatenate(results)

    # unpack the raw samples
    samples = JokerSamples.unpack(raw_samples,
                                  joker_helper.internal_units,
                                  poly_trend=joker_helper.prior.poly_trend,
                                  t0=joker_helper.data.t0)

    return samples


def rejection_sample_helper(joker_helper, prior_samples_file, pool,
                            random_state,
                            n_prior_samples=None,
                            max_posterior_samples=None,
                            n_linear_samples=1,
                            return_logprobs=False,
                            n_batches=None,
                            randomize_prior_order=False):

    # Total number of samples in the cache:
    with tb.open_file(prior_samples_file, mode='r') as f:
        n_total_samples = f.root[JokerSamples._hdf5_path].shape[0]

    if n_prior_samples is None:
        n_prior_samples = n_total_samples
    elif n_prior_samples > n_total_samples:
        raise ValueError("TODO:")

    if max_posterior_samples is None:
        max_posterior_samples = n_prior_samples

    # Keyword arguments to be passed to marginal_ln_likelihood_helper:
    ll_kw = dict(joker_helper=joker_helper,
                 prior_samples_file=prior_samples_file,
                 pool=pool,
                 n_batches=n_batches)

    if randomize_prior_order:
        # Generate a random ordering for the samples
        idx = random_state.choice(n_total_samples, size=n_prior_samples,
                                  replace=False)
        ll_kw['samples_idx'] = idx
    else:
        ll_kw['n_prior_samples'] = n_prior_samples

    # compute likelihoods
    lls = marginal_ln_likelihood_helper(**ll_kw)

    # get indices of samples that pass rejection step
    uu = random_state.uniform(size=len(lls))
    good_samples_idx = np.where(np.exp(lls - lls.max()) > uu)[0]
    good_samples_idx = good_samples_idx[:max_posterior_samples]

    if randomize_prior_order:
        full_samples_idx = idx[good_samples_idx]
    else:
        full_samples_idx = good_samples_idx
    samples_ll = lls[good_samples_idx]

    # generate linear parameters
    samples = make_full_samples(joker_helper, prior_samples_file, pool,
                                random_state, full_samples_idx,
                                n_linear_samples=n_linear_samples,
                                n_batches=n_batches)

    # TODO: deal with return_logprobs=True case...

    return samples


def iterative_sample_helper(joker_helper, prior_samples_file, pool,
                            random_state,
                            n_prior_samples=None,
                            max_posterior_samples=None,
                            n_linear_samples=1,
                            return_logprobs=False,
                            n_batches=None,
                            randomize_prior_order=False):

    # TODO!

    return samples
