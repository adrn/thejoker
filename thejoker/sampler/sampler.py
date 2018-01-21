# Standard library
from os import path
import sys
import tempfile

# Third-party
import astropy.units as u
import h5py
import numpy as np
from scipy.stats import beta, norm

# Project
from ..log import log as logger
from ..data import RVData
from .params import JokerParams
from .multiproc_helpers import (get_good_sample_indices, compute_likelihoods,
                                sample_indices_to_full_samples)
from .io import save_prior_samples
from .samples import JokerSamples

__all__ = ['TheJoker']


class TheJoker(object):
    """A custom Monte-Carlo sampler for two-body systems.

    Parameters
    ----------
    params : `~thejoker.sampler.params.JokerParams`
        Object specifying hyper-parameters for The Joker.
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
    """
    def __init__(self, params, pool=None, random_state=None, n_batches=None):

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
        if not isinstance(params, JokerParams):
            raise TypeError("Parameter specification must be a JokerParams "
                            "instance, not a '{0}'".format(type(params)))
        self.params = params

        self.n_batches = n_batches

    def sample_prior(self, size=1, return_logprobs=False):
        """Generate samples from the prior. Logarithmic in period, uniform in
        phase and argument of pericenter, Beta distribution in eccentricity.

        Parameters
        ----------
        size : int
            Number of samples to generate.
        return_logprobs : bool (optional)
            If ``True``, will also return the log-value of the prior at each
            sample.

        Returns
        -------
        samples : `~thejoker.sampler.samples.JokerSamples`
            Keys: `['P', 'M0', 'e', 'omega']`, each as
            `astropy.units.Quantity` objects (i.e. with units).

        TODO
        ----
        - All prior distributions are fixed. These should be customizable.
        """
        rnd = self.random_state

        # Create an empty, dictionary-like 'samples' object to fill
        samples = JokerSamples()

        # sample from priors in nonlinear parameters
        a, b = (np.log(self.params.P_min.to(u.day).value),
                np.log(self.params.P_max.to(u.day).value))
        samples['P'] = np.exp(rnd.uniform(a, b, size=size)) * u.day

        samples['M0'] = rnd.uniform(0, 2 * np.pi, size=size) * u.radian

        # MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
        samples['e'] = rnd.beta(a=0.867, b=3.03, size=size)

        samples['omega'] = rnd.uniform(0, 2 * np.pi, size=size) * u.radian

        # Store the value of the prior at each prior sample
        # TODO: should we store the value for each parameter independently?
        if return_logprobs:
            ln_prior_val = np.zeros(size)

            # P
            ln_prior_val += -np.log(b - a) - np.log(samples['P'].value)

            # M0
            ln_prior_val += -np.log(2 * np.pi)

            # e - MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
            ln_prior_val += beta.logpdf(samples['e'], 0.867, 3.03)

            # omega
            ln_prior_val += -np.log(2 * np.pi)

        if not self.params._fixed_jitter:
            # Gaussian prior in log(s^2)
            log_s2 = rnd.normal(*self.params.jitter, size=size)
            samples['jitter'] = np.sqrt(np.exp(log_s2)) * self.params._jitter_unit

            if return_logprobs:
                Jac = (2 / samples['jitter'].value) # Jacobian
                ln_prior_val += norm.logpdf(log_s2,
                                            loc=self.params.jitter[0],
                                            scale=self.params.jitter[1]) * Jac

        else:
            samples['jitter'] = np.ones(size) * self.params.jitter

        if return_logprobs:
            return samples, ln_prior_val
        else:
            return samples

    def _unpack_full_samples(self, samples, prior_units, t0=None):
        """Unpack an array of The Joker samples into a dictionary-like object of
        Astropy Quantity objects (with units). This is meant to be used
        internally.

        Parameters
        ----------
        samples : `numpy.ndarray`
            A 2D array of posterior samples output from The Joker.
        prior_units : list
            List of units for the prior samples.
        t0 : `~astropy.time.Time` (optional)
            Passed to `thejoker.JokerSamples`.

        Returns
        -------
        samples : `~thejoker.sampler.samples.JokerSamples`

        """

        n, n_params = samples.shape

        joker_samples = JokerSamples(t0=t0)

        # TODO: need to keep track of this elsewhere...
        nonlin_params = ['P', 'M0', 'e', 'omega', 'jitter']
        for k, key in enumerate(nonlin_params):
            joker_samples[key] = samples[:, k] * prior_units[k]

        joker_samples['K'] = samples[:, k+1] * prior_units[-1] # jitter unit
        joker_samples['v0'] = samples[:, k+2] * prior_units[-1] # jitter unit

        return joker_samples

    def _rejection_sample_from_cache(self, data, n_prior_samples, cache_file,
                                     start_idx, seed, return_logprobs=False):
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

        # For samples that pass the rejection step, we now have their indices
        # in the prior cache file. Here, we read the actual values:
        result = sample_indices_to_full_samples(
            good_samples_idx, cache_file, data, self.params,
            pool=self.pool, global_seed=seed, return_logprobs=return_logprobs)

        return result

    def rejection_sample(self, data, n_prior_samples=None,
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
            The radial velocity.
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

        if n_prior_samples is None and prior_cache_file is None:
            raise ValueError("You either have to specify the number of prior "
                             "samples to generate, or a path to an HDF5 file "
                             "containing cached prior samples. If you want to "
                             "try an experimental adaptive method, use "
                             ".iterative_rejection_sample() instead.")

        # compute full parameter vectors for all good samples
        if self._rnd_passed:
            seed = self.random_state.randint(np.random.randint(2**16))
        else:
            seed = None

        if prior_cache_file is not None:
            # read prior units from cache file
            with h5py.File(prior_cache_file, 'r') as f:
                prior_units = [u.Unit(uu) for uu in f.attrs['units']]

                if n_prior_samples is None:
                    n_prior_samples = len(f['samples'])

            result = self._rejection_sample_from_cache(
                data, n_prior_samples, prior_cache_file, start_idx, seed=seed,
                return_logprobs=return_logprobs)

        else:
            with tempfile.NamedTemporaryFile(mode='r+') as f:
                # first do prior sampling, cache to temporary file
                prior_samples = self.sample_prior(size=n_prior_samples)
                prior_units = save_prior_samples(f.name, prior_samples,
                                                 data.rv.unit)
                result = self._rejection_sample_from_cache(
                    data, n_prior_samples, f.name, start_idx, seed=seed,
                    return_logprobs=return_logprobs)

        if return_logprobs:
            samples, ln_prior, ln_like = result

        else:
            samples = result

        samples = self._unpack_full_samples(samples, prior_units, t0=data.t0)

        if return_logprobs:
            return samples, ln_prior

        else:
            return samples

    def iterative_rejection_sample(self, data, n_requested_samples,
                                   prior_cache_file, n_prior_samples=None,
                                   return_logprobs=False, magic_fudge=128):
        """ For now: prior_cache_file is required

        Parameters
        ----------
        data : `~thejoker.RVData`
        n_requested_samples : int
        prior_cache_file : str
        n_prior_samples : int (optional)
        return_logprobs : bool (optional)
        magic_fudge : int (optional)
        """

        # validate input data
        if not isinstance(data, RVData):
            raise TypeError("Input data must be an RVData instance, not '{}'"
                            .format(type(data)))

        if n_prior_samples is None and prior_cache_file is None:
            raise ValueError("You either have to specify the number of prior "
                             "samples to generate, or a path to a file "
                             "containing cached prior samples.")

        # a bit of a hack to make the K, v0 samples deterministic
        if self._rnd_passed:
            seed = self.random_state.randint(np.random.randint(2**16))
        else:
            seed = None

        # read prior units from cache file
        with h5py.File(prior_cache_file, 'r') as f:
            prior_units = [u.Unit(uu) for uu in f.attrs['units']]

            if n_prior_samples is None: # take all samples if not specified
                n_prior_samples = len(f['samples'])

        # Start from the beginning of the prior cache file
        start_idx = 0

        safety_factor = 2 # MAGIC NUMBER
        n_process = magic_fudge * n_requested_samples # MAGIC NUMBER

        if n_process > n_prior_samples:
            raise ValueError("Prior sample library not big enough! For "
                             "iterative sampling, you have to have at least "
                             "magic_fudge * n_requested_samples samples in the "
                             "prior samples cache file. You have {0}"
                             .format(n_prior_samples))

        all_marg_lls = np.array([])

        maxiter = 128
        for i in range(maxiter): # we just need to iterate for a long time
            logger.log(1, "The Joker iteration {0}, computing {1} likelihoods"
                       .format(i, n_process))
            marg_lls = compute_likelihoods(n_process, prior_cache_file,
                                           start_idx, data, self.params,
                                           pool=self.pool,
                                           n_batches=self.n_batches)

            all_marg_lls = np.concatenate((all_marg_lls, marg_lls))

            good_samples_idx = get_good_sample_indices(all_marg_lls, seed=seed)

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
            good_samples_idx, prior_cache_file, data, self.params,
            pool=self.pool, global_seed=seed,
            return_logprobs=return_logprobs)

        if return_logprobs:
            full_samples, ln_prior, ln_like = result

        else:
            full_samples = result

        samples_dict = self._unpack_full_samples(full_samples,
                                                 prior_units, t0=data.t0)

        if return_logprobs:
            return samples_dict, ln_prior

        else:
            return samples_dict
