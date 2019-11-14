# Third-party
import numpy as np
import tables as tb

# Project
from .logging import logger
from .samples import JokerSamples
from .utils import batch_tasks, read_batch, table_contains_column


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

        if return_logprobs:
            if not table_contains_column(f.root, 'ln_prior'):
                raise RuntimeError("return_logprobs=True but ln_prior values "
                                   "not found in prior cache: make sure you "
                                   "generate prior samples with prior.sample "
                                   "(..., return_logprobs=True) before saving "
                                   "the prior samples.")

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

    # generate linear parameters
    samples = make_full_samples(joker_helper, prior_samples_file, pool,
                                random_state, full_samples_idx,
                                n_linear_samples=n_linear_samples,
                                n_batches=n_batches)

    if return_logprobs:
        samples['ln_likelihood'] = lls[good_samples_idx]

        with tb.open_file(prior_samples_file, mode='r') as f:
            data = f.root[JokerSamples._hdf5_path]
            samples['ln_prior'] = data.read_coordinates(full_samples_idx)

    return samples


def iterative_rejection_helper(joker_helper, prior_samples_file, pool,
                               random_state,
                               n_requested_samples,
                               init_batch_size=None,
                               growth_factor=128,
                               max_prior_samples=None,
                               n_linear_samples=1,
                               return_logprobs=False,
                               n_batches=None,
                               randomize_prior_order=False):

    # Total number of samples in the cache:
    with tb.open_file(prior_samples_file, mode='r') as f:
        n_total_samples = f.root[JokerSamples._hdf5_path].shape[0]

        if return_logprobs:
            if not table_contains_column(f.root, 'ln_prior'):
                raise RuntimeError("return_logprobs=True but ln_prior values "
                                   "not found in prior cache: make sure you "
                                   "generate prior samples with prior.sample "
                                   "(..., return_logprobs=True) before saving "
                                   "the prior samples.")

    if max_prior_samples is None:
        max_prior_samples = n_total_samples

    # The "magic numbers" below control how fast the iterative batches grow
    # in size, and the maximum number of iterations
    maxiter = 128  # MAGIC NUMBER
    safety_factor = 4  # MAGIC NUMBER
    if init_batch_size is None:
        n_process = growth_factor * n_requested_samples  # MAGIC NUMBER
    else:
        n_process = init_batch_size

    if n_process > max_prior_samples:
        raise ValueError("Prior sample library not big enough! For "
                         "iterative sampling, you have to have at least "
                         "growth_factor * n_requested_samples = "
                         f"{growth_factor * n_requested_samples} samples in "
                         "the prior samples cache file. You have, or have "
                         f"limited to, {max_prior_samples} samples.")

    if randomize_prior_order:
        # Generate a random ordering for the samples
        all_idx = random_state.choice(n_total_samples, size=max_prior_samples,
                                      replace=False)
    else:
        all_idx = np.arange(0, max_prior_samples, 1)

    all_marg_lls = np.array([])
    start_idx = 0
    for i in range(maxiter):
        logger.log(1, f"iteration {i}, computing {n_process} likelihoods")

        marg_lls = marginal_ln_likelihood_helper(
            joker_helper, prior_samples_file, pool,
            n_batches=None, n_prior_samples=None,
            samples_idx=all_idx[start_idx:start_idx + n_process])

        all_marg_lls = np.concatenate((all_marg_lls, marg_lls))

        # get indices of samples that pass rejection step
        uu = random_state.uniform(size=len(all_marg_lls))
        aa = np.exp(all_marg_lls - all_marg_lls.max())
        good_samples_idx = np.where(aa > uu)[0]

        if len(good_samples_idx) == 0:
            raise RuntimeError("Failed to find any good samples!")

        n_good = len(good_samples_idx)
        logger.log(1, f"{n_good} good samples after rejection sampling")

        if n_good >= n_requested_samples:
            logger.debug("Enough samples found!")
            break

        start_idx += n_process

        n_ll_evals = len(all_marg_lls)
        n_need = n_requested_samples - n_good
        n_process = int(safety_factor * n_need / n_good * n_ll_evals)

        if start_idx + n_process > max_prior_samples:
            n_process = max_prior_samples - start_idx

        if n_process <= 0:
            break

    else:
        # We should never get here!!
        raise RuntimeError("Hit maximum number of iterations!")

    full_samples_idx = all_idx[good_samples_idx]

    # generate linear parameters
    samples = make_full_samples(joker_helper, prior_samples_file, pool,
                                random_state, full_samples_idx,
                                n_linear_samples=n_linear_samples,
                                n_batches=n_batches)

    # TODO: copy-pasted from function above
    if return_logprobs:
        samples['ln_likelihood'] = all_marg_lls[good_samples_idx]

        with tb.open_file(prior_samples_file, mode='r') as f:
            data = f.root[JokerSamples._hdf5_path]
            samples['ln_prior'] = data.read_coordinates(full_samples_idx)

    return samples