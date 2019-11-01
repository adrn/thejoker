# Standard library
import sys
import tempfile
import time

# Third-party
import astropy.units as u
import h5py
import numpy as np

# Project
from .utils import logger
from .data import RVData
from .prior import JokerPrior
from .multiproc_helpers import (get_good_sample_indices, compute_likelihoods,
                                sample_indices_to_full_samples)
from .samples import JokerSamples

__all__ = ['TheJoker']


class TheJoker:
    """A custom Monte-Carlo sampler for two-body systems.

    Parameters
    ----------
    pool : ``schwimmbad.BasePool`` (optional)
        A processing pool (default is a ``schwimmbad.SerialPool`` instance).
    random_state : `numpy.random.RandomState` (optional)
        A ``RandomState`` instance to serve as a parent for the random
        number generators. See the :ref:`random numbers <random-numbers>` page
        for more information.
    n_batches : int (optional)
        When using multiprocessing to split the likelihood evaluations, this
        sets the number of batches to split the work into. Defaults to
        ``pool.size``, meaning equal work to each worker. For very large prior
        sample caches, you may need to set this to a larger number (e.g.,
        ``100*pool.size``) to avoid memory issues.
    tempfile_path : str (optional)
        Path to create temporary files needed for executing the sampler.
        Defaults to whatever Python's ``tempfile`` thinks is a good temporary
        directory.
    """

    def __init__(self, prior, pool=None, random_state=None,
                 n_batches=None, tempfile_path=None):

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
            raise TypeError("TODO: prior must be a prior...")
        self.prior = prior

        # TODO: store or remember number of v0 offsets...

        self.n_batches = n_batches
        self.tempfile_path = tempfile_path

    def _unpack_full_samples(self, result, prior_units, return_logprobs,
                             t0=None):
        """Unpack an array of The Joker samples into a dictionary-like object of
        Astropy Quantity objects (with units). This is meant to be used
        internally.

        Parameters
        ----------
        result : tuple
            A tuple of output directly from _rejection_sample_from_cache().
            Depending on the value of ``return_logprobs``, this is either just
            the sample values as a 2D array, or a lenth 3 tuple with the 2D
            samples array, prior values, and likelihood values.
        prior_units : list
            List of units for the prior samples.
        return_logprobs : bool
            Are we also returning the log prior values?
        t0 : `~astropy.time.Time` (optional)
            Passed to `thejoker.JokerSamples`.

        Returns
        -------
        samples : `~thejoker.sampler.samples.JokerSamples`

        """

        if return_logprobs:
            samples_arr, ln_prior, ln_like = result

        else:
            samples_arr = result

        n, n_params = samples_arr.shape

        samples = JokerSamples(t0=t0, poly_trend=self.params.poly_trend)

        # TODO: need to keep track of this elsewhere...
        nonlin_params = ['P', 'M0', 'e', 'omega', 'jitter']
        for k, key in enumerate(nonlin_params):
            samples[key] = samples_arr[:, k] * prior_units[k]

        samples['K'] = samples_arr[:, k + 1] * prior_units[-1]  # jitter unit

        for i in range(self.params.poly_trend):
            _unit = prior_units[-1] / u.day**i  # HACK: jitter unit per day
            samples['v'+str(i)] = samples_arr[:, k + 2 + i] * _unit

        if return_logprobs:
            return samples, ln_prior, ln_like

        else:
            return samples

    def marginal_ln_likelihood(self, data, prior_samples):
        pass

    def _rejection_sample_from_cache(self, data, n_prior_samples, max_n_samples,
                                     cache_file, start_idx, seed,
                                     return_logprobs=False):
        """Perform The Joker's rejection sampling on a cache file containing
        prior samples. This is meant to be used internally.
        """

        # Get indices of good samples from the cache file
        # TODO: I have some implementation questions about whether this should
        #   return a boolean array (in which case I need to process all
        #   likelihood values) or an array of integers...Right now,
        #   _marginal_ll_worker has to return the values because we then compare
        #   with the maximum value of the likelihood
        marg_lls = compute_likelihoods(n_prior_samples, cache_file, start_idx,
                                       data, self.params, pool=self.pool,
                                       n_batches=self.n_batches)
        good_samples_idx = get_good_sample_indices(marg_lls, seed=seed)

        if len(good_samples_idx) == 0:
            logger.error("Failed to find any good samples!")
            self.pool.close()
            sys.exit(1)

        n_good = len(good_samples_idx)
        s_or_not = 's' if n_good > 1 else ''
        logger.info("{0} good sample{1} after rejection sampling"
                    .format(n_good, s_or_not))

        if max_n_samples is None:
            max_n_samples = n_good

        # For samples that pass the rejection step, we now have their indices
        # in the prior cache file. Here, we read the actual values:
        result = sample_indices_to_full_samples(
            good_samples_idx, cache_file, data, self.params, max_n_samples,
            pool=self.pool, global_seed=seed, return_logprobs=return_logprobs)

        return result

    def _validate_prior_cache(self, n_prior_samples, prior_cache_file):
        """Internal method used to either validate the prior cache file, or
        create one based on the number of samples input.
        """

        if n_prior_samples is None and prior_cache_file is None:
            raise ValueError("You either have to specify the number of prior "
                             "samples to generate, or a path to an HDF5 file "
                             "containing cached prior samples.")

        if prior_cache_file is not None:
            # read prior units from cache file
            with h5py.File(prior_cache_file, 'r') as f:
                if n_prior_samples is None:
                    n_prior_samples = len(f['samples'])

                # TODO: also validate keys in cache?

            cache_exists = True
            logger.log(1, "Prior cache file found and validated.")

        else:
            cache_exists = False
            logger.log(1, "Prior cache file not found or invalid.")

        return n_prior_samples, cache_exists

    def rejection_sample(self, data, n_prior_samples=None, max_n_samples=None,
                         prior_cache_file=None, return_logprobs=False,
                         start_idx=0):
        """Run The Joker's rejection sampling on prior samples to get posterior
        samples for the input data.

        You must either specify the number of prior samples to generate and
        use for rejection sampling, ``n_prior_samples``, or the path to a file
        containing prior samples, ``prior_cache_file``.

        Parameters
        ----------
        data : `~thejoker.data.RVData`
            The radial velocity data.
        n_prior_samples : int (optional)
            If ``prior_cache_file`` is not specified, this sets the number of
            prior samples to generate and use to do the rejection sampling. If
            ``prior_cache_file`` is specified, this sets the number of prior
            samples to load from the cache file.
        prior_cache_file : str (optional)
            A path to an HDF5 cache file containing prior samples. TODO: more
            information
        return_logprobs : bool (optional)
            Also return the log-probabilities.
        start_idx : int (optional)
            Index to start reading from in the prior cache file.

        """

        # validate input data
        if not isinstance(data, RVData):
            raise TypeError("Input data must be an RVData instance, not '{0}'"
                            .format(type(data)))

        # compute full parameter vectors for all good samples
        if self._rnd_passed:
            seed = self.random_state.randint(np.random.randint(2**16))
        else:
            seed = None

        n_prior_samples, cache_exists = self._validate_prior_cache(
            n_prior_samples, prior_cache_file)

        if cache_exists:
            with h5py.File(prior_cache_file, 'r') as f:
                prior_units = [u.Unit(uu) for uu in f.attrs['units']]

            result = self._rejection_sample_from_cache(
                data, n_prior_samples, max_n_samples, prior_cache_file,
                start_idx, seed=seed, return_logprobs=return_logprobs)

        else:
            with tempfile.NamedTemporaryFile(mode='r+',
                                             dir=self.tempfile_path) as f:
                prior_cache_file = f.name

                # first do prior sampling, cache to temporary file
                prior_samples, lnp = self.sample_prior(size=n_prior_samples,
                                                       return_logprobs=True)
                prior_units = save_prior_samples(prior_cache_file,
                                                 prior_samples,
                                                 data.rv.unit,
                                                 ln_prior_probs=lnp)

                result = self._rejection_sample_from_cache(
                    data, n_prior_samples, max_n_samples, prior_cache_file,
                    start_idx, seed=seed, return_logprobs=return_logprobs)

        return self._unpack_full_samples(result, prior_units, t0=data.t0,
                                         return_logprobs=return_logprobs)

    def iterative_rejection_sample(self, data, n_requested_samples,
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

        # validate input data
        if not isinstance(data, RVData):
            raise TypeError("Input data must be an RVData instance, not '{}'"
                            .format(type(data)))

        # a bit of a hack to make the K, v0 samples deterministic
        if self._rnd_passed:
            seed = self.random_state.randint(np.random.randint(2**16))
        else:
            seed = None

        # HACK: must start from 0
        start_idx = 0

        n_prior_samples, cache_exists = self._validate_prior_cache(
            n_prior_samples, prior_cache_file)

        # There are some magic numbers below used to control how fast the
        # iterative batches grow in size
        maxiter = 128 # MAGIC NUMBER
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
