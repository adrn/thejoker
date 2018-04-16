# Standard library
import sys
import tempfile
import time

# Third-party
import astropy.units as u
import h5py
import numpy as np
from scipy.stats import scoreatpercentile

# Project
from ..log import log as logger
from ..data import RVData
from ..stats import beta_logpdf, norm_logpdf
from .params import JokerParams
from .multiproc_helpers import (get_good_sample_indices, compute_likelihoods,
                                sample_indices_to_full_samples)
from .io import save_prior_samples
from .samples import JokerSamples
from .mcmc import TheJokerMCMCModel

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
        samples['e'] = rnd.beta(a=0.867, b=3.03, size=size) * u.one

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
            ln_prior_val += beta_logpdf(samples['e'].value, 0.867, 3.03)

            # omega
            ln_prior_val += -np.log(2 * np.pi)

        if not self.params._fixed_jitter:
            # Gaussian prior in log(s^2)
            log_s2 = rnd.normal(*self.params.jitter, size=size)
            samples['jitter'] = np.sqrt(
                np.exp(log_s2)) * self.params._jitter_unit

            if return_logprobs:
                Jac = np.log(2 / samples['jitter'].value)  # Jacobian
                ln_prior_val += norm_logpdf(log_s2,
                                            self.params.jitter[0],
                                            self.params.jitter[1]) + Jac

        else:
            samples['jitter'] = np.ones(size) * self.params.jitter

        if return_logprobs:
            return samples, ln_prior_val
        else:
            return samples

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

        samples = JokerSamples(t0=t0)

        # TODO: need to keep track of this elsewhere...
        nonlin_params = ['P', 'M0', 'e', 'omega', 'jitter']
        for k, key in enumerate(nonlin_params):
            samples[key] = samples_arr[:, k] * prior_units[k]

        samples['K'] = samples_arr[:, k + 1] * prior_units[-1]  # jitter unit
        samples['v0'] = samples_arr[:, k + 2] * prior_units[-1]  # jitter unit

        if return_logprobs:
            return samples, ln_prior

        else:
            return samples

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

        else:
            cache_exists = False

        return n_prior_samples, cache_exists

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

        # compute full parameter vectors for all good samples
        if self._rnd_passed:
            seed = self.random_state.randint(np.random.randint(2**16))
        else:
            seed = None

        n_prior_samples, cache_exists = self._validate_prior_cache(
            n_prior_samples, prior_cache_file)

        if cache_exists:
            with h5py.File(prior_cache_file) as f:
                prior_units = [u.Unit(uu) for uu in f.attrs['units']]

            result = self._rejection_sample_from_cache(
                data, n_prior_samples, prior_cache_file, start_idx, seed=seed,
                return_logprobs=return_logprobs)

        else:
            with tempfile.NamedTemporaryFile(mode='r+') as f:
                prior_cache_file = f.name

                # first do prior sampling, cache to temporary file
                prior_samples = self.sample_prior(size=n_prior_samples)
                prior_units = save_prior_samples(prior_cache_file,
                                                 prior_samples,
                                                 data.rv.unit)

                result = self._rejection_sample_from_cache(
                    data, n_prior_samples, prior_cache_file, start_idx,
                    seed=seed, return_logprobs=return_logprobs)

        return self._unpack_full_samples(result, prior_units, t0=data.t0,
                                         return_logprobs=return_logprobs)

    def iterative_rejection_sample(self, data, n_requested_samples,
                                   prior_cache_file=None, n_prior_samples=None,
                                   return_logprobs=False, magic_fudge=128):
        """TODO: docstring For now: prior_cache_file is required

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
        safety_factor = 2  # MAGIC NUMBER
        n_process = magic_fudge * n_requested_samples  # MAGIC NUMBER

        if n_process > n_prior_samples:
            raise ValueError("Prior sample library not big enough! For "
                             "iterative sampling, you have to have at least "
                             "magic_fudge * n_requested_samples samples in the "
                             "prior samples cache file. You have {0}"
                             .format(n_prior_samples))

        all_marg_lls = np.array([])

        # TODO: it's a little...unclean to always make a tempfile

        with tempfile.NamedTemporaryFile(mode='r+') as f:
            if cache_exists:
                with h5py.File(prior_cache_file) as f:
                    prior_units = [u.Unit(uu) for uu in f.attrs['units']]

            else:
                prior_cache_file = f.name

                # first do prior sampling, cache to temporary file
                prior_samples = self.sample_prior(size=n_prior_samples)
                prior_units = save_prior_samples(f.name, prior_samples,
                                                 data.rv.unit)

            maxiter = 128
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
                good_samples_idx, prior_cache_file, data, self.params,
                pool=self.pool, global_seed=seed,
                return_logprobs=return_logprobs)

        return self._unpack_full_samples(result, prior_units, t0=data.t0,
                                         return_logprobs=return_logprobs)

    # ========================================================================
    # MCMC

    def mcmc_sample(self, data, samples0, n_steps=1024,
                    n_walkers=256, n_burn=8192, return_sampler=False,
                    ball_scale=1E-5):
        """Run standard MCMC (using `emcee <http://emcee.readthedocs.io/>`_) to
        generate posterior samples in orbital parameters.

        Parameters
        ----------
        data : `~thejoker.RVData`
            The data to fit orbits to.
        samples0 : `~thejoker.JokerSamples`
            This can either be (a) a single sample to use as initial conditions
            for the MCMC walkers, or (b) a set of samples, in which case the
            mean of the samples will be used as initial conditions.
        n_steps : int
            The number of MCMC steps to run for.
        n_walkers : int (optional)
            The number of walkers to use in the ``emcee`` ensemble.
        n_burn : int (optional)
            If specified, the number of steps to burn in for.
        return_sampler : bool (optional)
            Also return the sampler object.

        Returns
        -------
        model : `~thejoker.TheJokerMCMCModel`
        samples : `~thejoker.JokerSamples`
            The posterior samples.
        sampler : `emcee.EnsembleSampler`
            If ``return_sampler == True``.
        """
        import emcee

        if not isinstance(samples0, JokerSamples):
            raise TypeError('Input samples initial position must be ')

        model = TheJokerMCMCModel(joker_params=self.params, data=data)

        if len(samples0) > 1:
            samples0 = samples0.mean()

        p0_mean = np.squeeze(model.pack_samples(samples0))

        # P, M0, e, omega, jitter, K, v0
        p0 = np.zeros((n_walkers, len(p0_mean)))
        for i in range(p0.shape[1]):
            if i in [2, 4]: # eccentricity, jitter
                p0[:, i] = np.abs(np.random.normal(p0_mean[i], ball_scale,
                                                   size=n_walkers))

            else:
                p0[:, i] = np.random.normal(p0_mean[i], ball_scale,
                                            size=n_walkers)

        p0 = model.to_mcmc_params(p0.T).T

        # Because jitter is always carried through in the transform above, now
        # we have to remove the jitter parameter if it's fixed!
        if self.params._fixed_jitter:
            p0 = np.delete(p0, 5, axis=1)

        n_dim = p0.shape[1]
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, model,
                                        pool=self.pool)

        if n_burn is not None and n_burn > 0:
            logger.debug('Burning in MCMC for {0} steps...'.format(n_burn))
            time0 = time.time()
            pos, *_ = sampler.run_mcmc(p0, n_burn)
            logger.debug('...time spent burn-in: {0}'.format(time.time()-time0))

            p0 = pos
            sampler.reset()

        logger.debug('Running MCMC for {0} steps...'.format(n_steps))
        time0 = time.time()
        _ = sampler.run_mcmc(p0, n_steps)
        logger.debug('...time spent sampling: {0}'.format(time.time()-time0))

        acc_frac = sampler.acceptance_fraction
        if scoreatpercentile(acc_frac, 10) < 0.1:
            logger.warning('Walkers have low acceptance fractions: 10/50/90 '
                           'percentiles = {0:.2f}, {1:.2f}, {2:.2f}'
                           .format(*scoreatpercentile(acc_frac, [10, 50, 90])))

        samples = model.unpack_samples_mcmc(sampler.chain[:, -1])
        samples.t0 = samples0.t0

        if return_sampler:
            return model, samples, sampler

        else:
            return model, samples
