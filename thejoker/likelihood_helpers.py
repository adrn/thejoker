# Third-party
import numpy as np

# Project
from .logging import logger


def get_constant_term_design_matrix(data, ids=None):
    """
    Construct the portion of the design matrix relevant for the linear
    parameters of The Joker beyond the amplitude, ``K``.
    """

    if ids is None:
        ids = np.zeros(len(data), dtype=int)
    ids = np.array(ids)

    unq_ids = np.unique(ids)
    constant_part = np.zeros((len(data), len(unq_ids)))

    constant_part[:, 0] = 1.
    for j, id_ in enumerate(unq_ids[1:]):
        constant_part[ids == id_, j+1] = 1.

    return constant_part


def get_trend_design_matrix(data, ids, poly_trend):
    """
    Construct the full design matrix for linear parameters, without the K column
    """
    # Combine design matrix for constant term, which may contain columns for
    # sampling over v0 offsets, with the rest of the long-term trend columns
    const_M = get_constant_term_design_matrix(data, ids)
    dt = data._t_bmjd - data._t_ref_bmjd
    trend_M = np.vander(dt, N=poly_trend, increasing=True)[:, 1:]
    return np.hstack((const_M, trend_M))


def marginal_ln_likelihood_inmem(joker_helper, prior_samples_batch):
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

    if prior_samples_batch.dtype != np.float64:
        prior_samples_batch = prior_samples_batch.astype(np.float64)

    # memoryview is returned
    ll = joker_helper.batch_marginal_ln_likelihood(prior_samples_batch)

    return np.array(ll)


def make_full_samples_inmem(joker_helper, prior_samples_batch, random_state,
                            n_linear_samples=1):
    from .samples import JokerSamples

    if prior_samples_batch.dtype != np.float64:
        prior_samples_batch = prior_samples_batch.astype(np.float64)

    raw_samples, _ = joker_helper.batch_get_posterior_samples(
        prior_samples_batch, n_linear_samples, random_state)

    # unpack the raw samples
    samples = JokerSamples.unpack(raw_samples,
                                  joker_helper.internal_units,
                                  t_ref=joker_helper.data.t_ref,
                                  poly_trend=joker_helper.prior.poly_trend,
                                  n_offsets=joker_helper.prior.n_offsets)

    return samples


def rejection_sample_inmem(joker_helper, prior_samples_batch, random_state,
                           ln_prior=None,
                           max_posterior_samples=None,
                           n_linear_samples=1,
                           return_all_logprobs=False):

    if max_posterior_samples is None:
        max_posterior_samples = len(prior_samples_batch)

    # compute likelihoods
    lls = marginal_ln_likelihood_inmem(joker_helper, prior_samples_batch)

    # get indices of samples that pass rejection step
    uu = random_state.uniform(size=len(lls))
    good_samples_idx = np.where(np.exp(lls - lls.max()) > uu)[0]
    good_samples_idx = good_samples_idx[:max_posterior_samples]

    # generate linear parameters
    samples = make_full_samples_inmem(joker_helper,
                                      prior_samples_batch[good_samples_idx],
                                      random_state,
                                      n_linear_samples=n_linear_samples)

    if ln_prior is not None and ln_prior is not False:
        samples['ln_prior'] = ln_prior[good_samples_idx]
        samples['ln_likelihood'] = lls[good_samples_idx]

    if return_all_logprobs:
        return samples, lls

    else:
        return samples


def iterative_rejection_inmem(joker_helper, prior_samples_batch, random_state,
                              n_requested_samples,
                              ln_prior=None,
                              init_batch_size=None,
                              growth_factor=128,
                              n_linear_samples=1):

    n_total_samples = len(prior_samples_batch)

    # The "magic numbers" below control how fast the iterative batches grow
    # in size, and the maximum number of iterations
    maxiter = 128  # MAGIC NUMBER
    safety_factor = 1  # MAGIC NUMBER
    if init_batch_size is None:
        n_process = growth_factor * n_requested_samples  # MAGIC NUMBER
    else:
        n_process = init_batch_size

    if n_process > n_total_samples:
        raise ValueError("Prior sample library not big enough! For "
                         "iterative sampling, you have to have at least "
                         "growth_factor * n_requested_samples = "
                         f"{growth_factor * n_requested_samples} samples in "
                         "the prior samples cache file. You have, or have "
                         f"limited to, {n_total_samples} samples.")

    all_idx = np.arange(0, n_total_samples, 1)

    all_marg_lls = np.array([])
    start_idx = 0
    for i in range(maxiter):
        logger.log(1, f"iteration {i}, computing {n_process} likelihoods")

        marg_lls = marginal_ln_likelihood_inmem(
            joker_helper, prior_samples_batch[start_idx:start_idx + n_process])
        all_marg_lls = np.concatenate((all_marg_lls, marg_lls))

        if np.any(~np.isfinite(all_marg_lls)):
            return RuntimeError("There are NaN or Inf likelihood values in "
                                f"iteration step {i}!")
        elif len(all_marg_lls) == 0:
            return RuntimeError("No likelihood values returned in iteration "
                                f"step {i}")

        # get indices of samples that pass rejection step
        uu = random_state.uniform(size=len(all_marg_lls))
        aa = np.exp(all_marg_lls - all_marg_lls.max())
        good_samples_idx = np.where(aa > uu)[0]

        if len(good_samples_idx) == 0:
            raise RuntimeError("Failed to find any good samples!")

        n_good = len(good_samples_idx)
        logger.log(1, f"{n_good} good samples after rejection sampling")

        if n_good >= n_requested_samples:
            logger.log(1, "Enough samples found!")
            break

        start_idx += n_process

        n_ll_evals = len(all_marg_lls)
        n_need = n_requested_samples - n_good
        n_process = int(safety_factor * n_need / n_good * n_ll_evals)

        if start_idx + n_process > n_total_samples:
            n_process = n_total_samples - start_idx

        if n_process <= 0:
            break

    else:
        # We should never get here!!
        raise RuntimeError("Hit maximum number of iterations!")

    good_samples_idx = good_samples_idx[:n_requested_samples]
    full_samples_idx = all_idx[good_samples_idx]

    # generate linear parameters
    samples = make_full_samples_inmem(joker_helper,
                                      prior_samples_batch[full_samples_idx],
                                      random_state,
                                      n_linear_samples=n_linear_samples)

    # FIXME: copy-pasted from function above
    if ln_prior is not None and ln_prior is not False:
        samples['ln_prior'] = ln_prior[full_samples_idx]
        samples['ln_likelihood'] = all_marg_lls[good_samples_idx]

    return samples


def ln_normal(x, mu, var):
    return -0.5 * (np.log(2*np.pi * var) + (x - mu)**2 / var)
