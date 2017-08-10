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
    """
    A custom Monte-Carlo sampler for two-body systems.

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
    """
    def __init__(self, params, pool=None, random_state=None):

        # set the processing pool
        if pool is None:
            import schwimmbad
            pool = schwimmbad.SerialPool()

        elif not hasattr(pool, 'map') or not hasattr(pool, 'close'):
            raise TypeError("Input pool object must have .map() and .close() "
                            "methods. We recommend using `schwimmbad` pools.")

        self.pool = pool

        # set the parent random state - child processes get different states based on the parent
        if random_state is None:
            self._rnd_passed = False
            random_state = np.random.RandomState()

        elif not isinstance(random_state, np.random.RandomState):
            raise TypeError("Random state object must be a numpy RandomState instance, "
                            "not '{}'".format(type(random_state)))

        else:
            self._rnd_passed = True

        self.random_state = random_state

        # check if a JokerParams instance was passed in to specify the state
        if not isinstance(params, JokerParams):
            raise TypeError("Parameter specification must be a JokerParams instance, "
                            "not a '{}'".format(type(params)))
        self.params = params

    def sample_prior(self, size=1, return_logprobs=False):
        """
        Generate samples from the prior. Logarithmic in period, uniform in
        phase and argument of pericenter, Beta distribution in eccentricity.

        Parameters
        ----------
        size : int
            Number of samples to generate.
        return_logprobs : bool (optional)
            If ``True``, will also return the log-value of the prior at each sample.

        Returns
        -------
        samples : `~thejoker.sampler.samples.JokerSamples`
            Keys: `['P', 'phi0', 'ecc', 'omega']`, each as
            `astropy.units.Quantity` objects (i.e. with units).

        TODO
        ----
        - All prior distributions are essentially fixed. These should be
            customizable in some way...

        """
        rnd = self.random_state

        pars = JokerSamples(self.params.trend_cls)

        ln_prior_val = np.zeros(size)

        # sample from priors in nonlinear parameters
        a,b = (np.log(self.params.P_min.to(u.day).value),
               np.log(self.params.P_max.to(u.day).value))
        pars['P'] = np.exp(rnd.uniform(a, b, size=size)) * u.day
        ln_prior_val += -np.log(b-a) - np.log(pars['P'].value) # Jacobian

        pars['phi0'] = rnd.uniform(0, 2*np.pi, size=size) * u.radian
        ln_prior_val += -np.log(2*np.pi)

        # MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
        pars['ecc'] = rnd.beta(a=0.867, b=3.03, size=size)
        ln_prior_val += beta.logpdf(pars['ecc'], 0.867, 3.03)

        pars['omega'] = rnd.uniform(0, 2*np.pi, size=size) * u.radian
        ln_prior_val += -np.log(2*np.pi)

        if not self.params._fixed_jitter:
            # Gaussian prior in log(s^2)
            log_s2 = rnd.normal(*self.params.jitter, size=size)
            pars['jitter'] = np.sqrt(np.exp(log_s2)) * self.params._jitter_unit

            Jac = (2 / pars['jitter'].value) # Jacobian
            ln_prior_val += norm.logpdf(log_s2,
                                        loc=self.params.jitter[0],
                                        scale=self.params.jitter[1]) * Jac

        else:
            pars['jitter'] = np.ones(size) * self.params.jitter

        if return_logprobs:
            return pars, ln_prior_val
        else:
            return pars

    def _rejection_sample_from_cache(self, data, n_prior_samples, cache_file,
                                     start_idx, seed, return_logprobs=False):
        """
        """

        # Get indices of good samples from the cache file
        # TODO: I have some implementation questions about whether this should
        #   return a boolean array (in which case I need to process all
        #   likelihood values) or an array of integers...Right now,
        #   _marginal_ll_worker has to return the values because we then compare
        #   with the maximum value of the likelihood
        marg_lls = compute_likelihoods(n_prior_samples, cache_file, start_idx,
                                       data, self.params, pool=self.pool)
        good_samples_idx = get_good_sample_indices(marg_lls, seed=seed)

        if len(good_samples_idx) == 0:
            logger.error("Failed to find any good samples!")
            self.pool.close()
            sys.exit(0)

        n_good = len(good_samples_idx)
        logger.info("{} good samples after rejection sampling".format(n_good))

        full_samples = sample_indices_to_full_samples(
            good_samples_idx, cache_file, data, self.params,
            pool=self.pool, global_seed=seed, return_logprobs=return_logprobs)

        return full_samples

    def rejection_sample(self, data, n_prior_samples=None,
                         prior_cache_file=None, start_idx=0):
        """
        Run The Joker's rejection sampling on prior samples to get
        posterior samples for the input data.

        Parameters
        ----------
        data : `~thejoker.data.RVData`
            The radial velocity.
        n_prior_samples : int (optional)
        prior_cache_file : str (optional)
        start_idx : int (optional)
            Index to start reading from in the prior cache file.

        """

        # validate input data
        if not isinstance(data, RVData):
            raise TypeError("Input data must be an RVData instance, not '{0}'"
                            .format(type(data)))

        if n_prior_samples is None and prior_cache_file is None:
            raise ValueError("You either have to specify the number of prior "
                             "samples to generate, or a path to a file "
                             "containing cached prior samples in (TODO: what "
                             "format?). If you want to try an experimental "
                             "adaptive method, try .rejection_sample_adapt()")

        # compute full parameter vectors for all good samples
        if self._rnd_passed:
            seed = self.random_state.randint(np.random.randint(2**16))
        else:
            seed = None

        if prior_cache_file is not None:
            # read prior units from cache file
            with h5py.File(prior_cache_file, 'r') as f:
                prior_units = [u.Unit(uu) for uu in f.attrs['units']]

                # TODO: test this
                if n_prior_samples is None:
                    n_prior_samples = len(f['samples'])

            samples = self._rejection_sample_from_cache(data, n_prior_samples,
                                                        prior_cache_file,
                                                        start_idx, seed=seed)

        else:
            with tempfile.NamedTemporaryFile(mode='r+') as f:
                # first do prior sampling, cache to file
                prior_samples = self.sample_prior(size=n_prior_samples)
                prior_units = save_prior_samples(f.name, prior_samples, data.rv.unit)
                samples = self._rejection_sample_from_cache(data, n_prior_samples,
                                                            f.name, start_idx,
                                                            seed=seed)

        return self.unpack_full_samples(samples, data.t_offset, prior_units)

    def unpack_full_samples(self, samples, t_offset, prior_units):
        """
        Unpack an array of Joker samples into a dictionary of Astropy
        Quantity objects (with units). Note that the phase of pericenter
        returned here is now relative to BMJD = 0.

        Parameters
        ----------
        samples : `numpy.ndarray`
            TODO
        t_offset : numeric TODO
        prior_units : list
            List of units for the prior samples.

        Returns
        -------
        samples : `~thejoker.sampler.samples.JokerSamples`

        """
        sample_dict = JokerSamples(self.params.trend_cls)

        n,n_params = samples.shape

        # TODO: need to keep track of this elsewhere...
        nonlin_params = ['P', 'phi0', 'ecc', 'omega', 'jitter']
        for k,key in enumerate(nonlin_params):
            sample_dict[key] = samples[:,k] * prior_units[k]

        k += 1
        sample_dict['K'] = samples[:,k] * prior_units[-1] # jitter unit

        k += 1

        for j, par_name in enumerate(self.params.trend_cls.parameters):
            k += j
            sample_dict[par_name] = samples[:,k] * prior_units[-1] / u.day**j

        # convert phi0 from relative to t=data.t_offset to relative to mjd=0
        dphi = (2*np.pi*t_offset/sample_dict['P'].to(u.day).value * u.radian) % (2*np.pi*u.radian)
        sample_dict['phi0'] = (sample_dict['phi0'] + dphi) % (2*np.pi*u.radian)

        return sample_dict

    def iterative_rejection_sample(self, data, n_requested_samples,
                                   prior_cache_file, n_prior_samples=None,
                                   return_logprobs=False):
        """ For now: prior_cache_file is required """

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

        # TODO: here's where we need to do the iterative bullshit
        # cache_path, _filename = path.split(prior_cache_file)
        # prob_cache_tmpfile = path.join(cache_path, 'tmp_{0}'.format(_filename))

        # if path.exists(prob_cache_tmpfile):
        #     raise RuntimeError('SHIT!') # TODO: make this nicer or figure out what to do

        # with h5py.File(prob_cache_tmpfile, 'a') as f:
        #     f.create_dataset('probs', (0, 0), maxshape=(None, 2))

        # Start from the beginning of the prior cache file
        start_idx = 0

        safety_factor = 2 # MAGIC NUMBER
        n_process = 128 * n_requested_samples # 32 = MAGIC NUMBER

        all_marg_lls = np.array([])

        maxiter = 128
        for i in range(maxiter): # we just need to iterate for a long time
            logger.log(1, "The Joker iteration {0}, computing {1} likelihoods"
                       .format(i, n_process))
            marg_lls = compute_likelihoods(n_process, prior_cache_file,
                                           start_idx, data, self.params,
                                           pool=self.pool)

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

        full_samples, ln_prior, ln_like = sample_indices_to_full_samples(
            good_samples_idx, prior_cache_file, data, self.params,
            pool=self.pool, global_seed=seed,
            return_logprobs=True)

        samples_dict = self.unpack_full_samples(full_samples, data.t_offset,
                                                prior_units)

        if return_logprobs:
            return samples_dict, ln_prior

        else:
            return samples_dict
