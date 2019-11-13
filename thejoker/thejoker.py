# TODO: inside TheJoker, when sampling, validate that number of RVData's passed in equals the number of (offsets+1)
# prior = JokerPrior.from_default(..., v0_offsets=[pm.Normal(...)])
# joker = TheJoker(prior)
# joker.rejection_sample([data1, data2], ...)

# Standard library
import os

# Third-party
import astropy.units as u
import h5py
import numpy as np

# Project
from .logging import logger
from .data import RVData
from .data_helpers import validate_prepare_data
from .prior import JokerPrior
from .samples import JokerSamples
from .src.fast_likelihood import CJokerHelper
from .utils import tempfile_decorator

__all__ = ['TheJoker']


_data_doc = "The radial velocity data."
_prior_samples_doc = "TODO"


class TheJoker:
    """A custom Monte-Carlo sampler for two-body systems.

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
            self._rnd_passed = False
            random_state = np.random.RandomState()

        elif not isinstance(random_state, np.random.RandomState):
            raise TypeError("Random state object must be a numpy RandomState "
                            "instance, not '{0}'".format(type(random_state)))

        else:
            self._rnd_passed = True

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
        joker_helper = CJokerHelper(data, self.prior, trend_M)
        return joker_helper

    @tempfile_decorator
    def marginal_ln_likelihood(self, data, prior_samples, n_batches=None):
        f"""
        Compute the marginal log-likelihood at each of the input prior samples.

        Parameters
        ----------
        data : `thejoker.RVData`, iterable, dict
            {_data_doc}
        prior_samples : str, `thejoker.JokerSamples`
            {_prior_samples_doc}

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

    def rejection_sample(self, data, prior_samples,
                         n_prior_samples=None,
                         max_posterior_samples=None,
                         n_linear_samples=1, return_logprobs=False,
                         n_batches=None, randomize_prior_order=False):
        f"""
        Run The Joker's rejection sampling on prior samples to get posterior
        samples for the input data.

        You must either specify the number of prior samples to generate and
        use for rejection sampling, ``n_prior_samples``, or the path to a file
        containing prior samples, ``prior_cache_file``.

        Parameters
        ----------
        data : `~thejoker.data.RVData`
            {_data_doc}
        prior_samples : str, `~thejoker.JokerSamples`
            {_prior_samples_doc}
        n_prior_samples : int (optional)
            TODO
        max_posterior_samples : int (optional)
            TODO
        n_linear_samples : int (optional)
            TODO
        return_logprobs : bool (optional)
            Also return the log-probabilities.
        n_batches : int (optional)
            TODO

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

    def iterative_rejection_sample(self, data, prior_samples,
                                   n_requested_samples,
                                   prior_cache_file=None, n_prior_samples=None,
                                   return_logprobs=False, init_n_process=None,
                                   magic_fudge=128):
        """This is an experimental sampling method that adaptively generates
        posterior samples given a large library of prior samples. The advantage
        of this function over the standard ``rejection_sample`` method is that
        it will try to adaptively figure out how many prior samples it needs to
        evaluate the likelihood at in order to return the desired number of
        posterior samples.

        Parameters
        ----------
        data : `~thejoker.data.RVData`
            The radial velocity data.
        n_requested_samples : int (optional)
            The number of posterior samples desired.
        prior_cache_file : str
            A path to an HDF5 cache file containing prior samples.
        n_prior_samples : int (optional)
            The maximum number of prior samples to use.
        return_logprobs : bool (optional)
            Also return the log-probabilities.
        init_n_process : int (optional)
            The initial batch size of likelihoods to compute, before growing
            the batches using the multiplicative ``magic_fudge`` factor.
        magic_fudge : int (optional)
            A magic fudge factor to use when adaptively determining the number
            of prior samples to evaluate at. Larger numbers make the trial
            batches larger more rapidly.
        """
        from .multiproc_helpers import iterative_rejection_helper
        joker_helper = self._make_joker_helper(data)  # also validates data

        # TODO: left off here in this module

        # The "magic numbers" below control how fast the iterative batches grow
        # in size, and the maximum number of iterations
        maxiter = 128  # MAGIC NUMBER
        safety_factor = 4  # MAGIC NUMBER
        if init_n_process is None:
            n_process = magic_fudge * n_requested_samples  # MAGIC NUMBER
        else:
            n_process = init_n_process

        if n_process > n_prior_samples:
            raise ValueError("Prior sample library not big enough! For "
                             "iterative sampling, you have to have at least "
                             "magic_fudge * n_requested_samples samples in the "
                             "prior samples cache file. You have {0}"
                             .format(n_prior_samples))

        all_marg_lls = np.array([])

        # TODO: it's a little...unclean to always make a tempfile

        if cache_exists:
            logger.log(1, "Cache file exists at: {0}"
                       .format(prior_cache_file))
            with h5py.File(prior_cache_file, 'r') as f:
                prior_units = [u.Unit(uu) for uu in f.attrs['units']]
            close_f = False

        else:
            f = tempfile.NamedTemporaryFile(mode='r+', dir=self.tempfile_path)
            close_f = True
            prior_cache_file = f.name
            logger.log(1, "Cache file not found - creating prior samples "
                       "and saving them to: {0}".format(prior_cache_file))

            # first do prior sampling, cache to temporary file
            prior_samples, lnp = self.sample_prior(size=n_prior_samples,
                                                   return_logprobs=True)
            prior_units = save_prior_samples(f.name, prior_samples,
                                             data.rv.unit,
                                             ln_prior_probs=lnp)

        for i in range(maxiter):  # we just need to iterate for a long time
            logger.log(1, "The Joker iteration {0}, computing {1} "
                       "likelihoods".format(i, n_process))
            marg_lls = compute_likelihoods(n_process, prior_cache_file,
                                           start_idx, data, self.params,
                                           pool=self.pool,
                                           n_batches=self.n_batches)

            all_marg_lls = np.concatenate((all_marg_lls, marg_lls))

            good_samples_idx = get_good_sample_indices(all_marg_lls,
                                                       seed=seed)

            if len(good_samples_idx) == 0:
                # self.pool.close()
                raise RuntimeError("Failed to find any good samples!")

            n_good = len(good_samples_idx)
            logger.log(1, "{0} good samples after rejection sampling"
                       .format(n_good))

            if len(good_samples_idx) >= n_requested_samples:
                logger.debug("Enough samples found! {0}"
                             .format(len(good_samples_idx)))
                break

            start_idx += n_process

            n_ll_evals = len(all_marg_lls)
            n_need = n_requested_samples - n_good
            n_process = int(safety_factor * n_need / n_good * n_ll_evals)

            if start_idx + n_process > n_prior_samples:
                n_process = n_prior_samples - start_idx

            if n_process <= 0:
                break

        else:
            # We should never get here!!
            raise RuntimeError("Hit maximum number of iterations!")

        result = sample_indices_to_full_samples(
            good_samples_idx, prior_cache_file, data,
            self.params, n_requested_samples, pool=self.pool, global_seed=seed,
            return_logprobs=return_logprobs)

        if close_f:
            logger.log(1, "Closing prior cache tempfile")
            f.close()

        return self._unpack_full_samples(result, prior_units, t0=data.t0,
                                         return_logprobs=return_logprobs)
