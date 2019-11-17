# Standard library
import os

# Third-party
import numpy as np

# Project
from .logging import logger
from .data_helpers import validate_prepare_data
from .prior import JokerPrior
from .src.fast_likelihood import CJokerHelper
from thejoker.utils import tempfile_decorator

__all__ = ['TheJoker']


class TheJoker:
    """
    A custom Monte-Carlo sampler for two-body systems.

    Parameters
    ----------
    prior : `~thejoker.JokerPrior`
        The specification of the prior probability distribution over all
        parameters used in The Joker.
    pool : ``schwimmbad.BasePool`` (optional)
        A processing pool (default is a ``schwimmbad.SerialPool`` instance).
    random_state : `numpy.random.RandomState` (optional)
        A ``RandomState`` instance to serve as a parent for the random
        number generators. See the :ref:`random numbers <random-numbers>` page
        for more information.
    tempfile_path : str (optional)
        A location on disk where The Joker may store some temporary files. Any
        files written here by The Joker should be cleaned up: If any files in
        this path persist, something must have gone wrong within The Joker.
        Default: ``~/.thejoker``
    """

    def __init__(self, prior, pool=None, random_state=None, tempfile_path=None):

        # set the processing pool
        if pool is None:
            import schwimmbad
            pool = schwimmbad.SerialPool()
        elif not hasattr(pool, 'map') or not hasattr(pool, 'close'):
            raise TypeError("Input pool object must have .map() and .close() "
                            "methods. We recommend using `schwimmbad` pools.")
        self.pool = pool

        # Set the parent random state - child processes get different states
        # based on the parent
        if random_state is None:
            random_state = np.random.RandomState()
        elif not isinstance(random_state, np.random.RandomState):
            raise TypeError("Random state object must be a numpy RandomState "
                            "instance, not '{0}'".format(type(random_state)))
        self.random_state = random_state

        # check if a JokerParams instance was passed in to specify the state
        if not isinstance(prior, JokerPrior):
            raise TypeError("The input prior must be a JokerPrior instance.")
        self.prior = prior

        if tempfile_path is None:
            self._tempfile_path = os.path.expanduser('~/.thejoker/')
        else:
            self._tempfile_path = os.path.abspath(
                os.path.expanduser(tempfile_path))

    @property
    def tempfile_path(self):
        os.makedirs(self._tempfile_path, exist_ok=True)
        return self._tempfile_path

    def _make_joker_helper(self, data):
        all_data, ids, trend_M = validate_prepare_data(data, self.prior)
        joker_helper = CJokerHelper(all_data, self.prior, trend_M)
        return joker_helper

    @tempfile_decorator
    def marginal_ln_likelihood(self, data, prior_samples, n_batches=None):
        f"""
        Compute the marginal log-likelihood at each of the input prior samples.

        Parameters
        ----------
        data : `thejoker.RVData`, iterable, dict
            The radial velocity data, or an iterable containing ``RVData``
            objects for each data source.
        prior_samples : str, `thejoker.JokerSamples`
            Either a path to a file containing prior samples generated from The
            Joker, or a `~thejoker.JokerSamples` instance containing the prior
            samples.
        n_batches : int (optional)
            The number of batches to split the prior samples into before
            distributing for computation. If using the (default) serial
            computation pool, this doesn't have any impact. If using
            multiprocessing or MPI, this determines how many batches to split
            the samples into before scattering over all workers.

        Returns
        -------
        ln_likelihood : `numpy.ndarray`
            The marginal log-likelihood computed at the location of each prior
            sample.
        """
        from .multiproc_helpers import marginal_ln_likelihood_helper
        joker_helper = self._make_joker_helper(data)  # also validates data
        return marginal_ln_likelihood_helper(joker_helper, prior_samples,
                                             self.pool, n_batches=n_batches)

    @tempfile_decorator
    def rejection_sample(self, data, prior_samples,
                         n_prior_samples=None,
                         max_posterior_samples=None,
                         n_linear_samples=1,
                         return_logprobs=False,
                         n_batches=None,
                         randomize_prior_order=False):
        f"""
        Run The Joker's rejection sampling on prior samples to get posterior
        samples for the input data.

        You must either specify the number of prior samples to generate and
        use for rejection sampling, ``n_prior_samples``, or the path to a file
        containing prior samples, ``prior_cache_file``.

        Parameters
        ----------
        data : `~thejoker.data.RVData`
            The radial velocity data, or an iterable containing ``RVData``
            objects for each data source.
        prior_samples : str, `~thejoker.JokerSamples`
            Either a path to a file containing prior samples generated from The
            Joker, or a `~thejoker.JokerSamples` instance containing the prior
            samples.
        n_prior_samples : int (optional)
            The number of prior samples to run on. This is only used if passing
            in a string filename: If the file contains a large number of prior
            samples, you may want to set this to only run on a subset.
        max_posterior_samples : int (optional)
            The maximum number of posterior samples to generate. If using a
            large library of prior samples, and running on uninformative data,
            you may want to set this to a small but reasonable number (like,
            256).
        n_linear_samples : int (optional)
            The number of linear parameter samples to generate for each
            nonlinear parameter sample returned from the rejection sampling
            step.
        return_logprobs : bool (optional)
            Also return the log-prior and (marginal) log-likelihood values
            evaluated at each sample.
        n_batches : int (optional)
            The number of batches to split the prior samples into before
            distributing for computation. If using the (default) serial
            computation pool, this doesn't have any impact. If using
            multiprocessing or MPI, this determines how many batches to split
            the samples into before scattering over all workers.
        randomize_prior_order : bool (optional)
            Randomly shuffle the prior samples before reading and running the
            rejection sampler. This is only useful if you are using a large
            library of prior samples, and choosing to run on a subset of those
            samples.

        Returns
        -------
        samples : `~thejoker.JokerSamples`
            The posterior samples produced from The Joker.

        """
        from .multiproc_helpers import rejection_sample_helper
        joker_helper = self._make_joker_helper(data)  # also validates data
        samples = rejection_sample_helper(
            joker_helper,
            prior_samples,
            pool=self.pool,
            random_state=self.random_state,
            n_prior_samples=n_prior_samples,
            max_posterior_samples=max_posterior_samples,
            n_linear_samples=n_linear_samples,
            return_logprobs=return_logprobs,
            n_batches=n_batches,
            randomize_prior_order=randomize_prior_order)

        return samples

    @tempfile_decorator
    def iterative_rejection_sample(self, data, prior_samples,
                                   n_requested_samples,
                                   max_prior_samples=None,
                                   n_linear_samples=1,
                                   return_logprobs=False,
                                   n_batches=None,
                                   randomize_prior_order=False,
                                   init_batch_size=None,
                                   growth_factor=128):

        """This is an experimental sampling method that adaptively generates
        posterior samples given a large library of prior samples. The advantage
        of this function over the standard ``rejection_sample`` method is that
        it will try to adaptively figure out how many prior samples it needs to
        evaluate the likelihood at in order to return the desired number of
        posterior samples.

        Parameters
        ----------
        data : `~thejoker.data.RVData`
            The radial velocity data, or an iterable containing ``RVData``
            objects for each data source.
        prior_samples : str, `~thejoker.JokerSamples`
            Either a path to a file containing prior samples generated from The
            Joker, or a `~thejoker.JokerSamples` instance containing the prior
            samples.
        n_requested_samples : int (optional)
            The number of posterior samples desired.
        max_prior_samples : int (optional)
            The maximum number of prior samples to process.
        n_linear_samples : int (optional)
            The number of linear parameter samples to generate for each
            nonlinear parameter sample returned from the rejection sampling
            step.
        return_logprobs : bool (optional)
            Also return the log-prior and (marginal) log-likelihood values
            evaluated at each sample.
        n_batches : int (optional)
            The number of batches to split the prior samples into before
            distributing for computation. If using the (default) serial
            computation pool, this doesn't have any impact. If using
            multiprocessing or MPI, this determines how many batches to split
            the samples into before scattering over all workers.
        randomize_prior_order : bool (optional)
            Randomly shuffle the prior samples before reading and running the
            rejection sampler. This is only useful if you are using a large
            library of prior samples, and choosing to run on a subset of those
            samples.
        init_batch_size : int (optional)
            The initial batch size of likelihoods to compute, before growing
            the batches using the multiplicative growth factor, below.
        growth_factor : int (optional)
            A factor used to adaptively grow the number of prior samples to
            evaluate on. Larger numbers make the trial batches grow faster.

        Returns
        -------
        samples : `~thejoker.JokerSamples`
            The posterior samples produced from The Joker.
        """
        from .multiproc_helpers import iterative_rejection_helper
        joker_helper = self._make_joker_helper(data)  # also validates data
        samples = iterative_rejection_helper(
            joker_helper,
            prior_samples,
            init_batch_size=init_batch_size,
            growth_factor=growth_factor,
            pool=self.pool,
            random_state=self.random_state,
            n_requested_samples=n_requested_samples,
            max_prior_samples=max_prior_samples,
            n_linear_samples=n_linear_samples,
            return_logprobs=return_logprobs,
            n_batches=n_batches,
            randomize_prior_order=randomize_prior_order)

        return samples

