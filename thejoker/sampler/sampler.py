# Standard library
import sys
import tempfile

# Third-party
import astropy.units as u
import h5py
import numpy as np

# Project
from ..log import log as logger
from ..data import RVData
from .params import JokerParams
from .multiproc_helpers import get_good_sample_indices, sample_indices_to_full_samples
from .io import save_prior_samples

__all__ = ['TheJoker']

class TheJoker(object):
    """
    A custom Monte-Carlo sampler for two-body systems.

    Parameters
    ----------
    data : `~thejoker.data.RVData`
        The radial velocity data.
    params : `~thejoker.sampler.params.JokerParams`
        TODO
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
            self._rnd_passed = True
            random_state = np.random.RandomState()

        elif not isinstance(random_state, np.random.RandomState):
            raise TypeError("Random state object must be a numpy RandomState instance, "
                            "not '{}'".format(type(random_state)))

        else:
            self._rnd_passed = False

        self.random_state = random_state

        # check if a JokerParams instance was passed in to specify the state
        if not isinstance(params, JokerParams):
            raise TypeError("Parameter specification must be a JokerParams instance, "
                            "not a '{}'".format(type(params)))
        self.params = params

    def sample_prior(self, size=1):
        """
        Generate samples from the prior. Logarithmic in period, uniform in
        phase and argument of pericenter, Beta distribution in eccentricity.

        Parameters
        ----------
        size : int
            Number of samples to generate.

        Returns
        -------
        prior_samples : dict
            Keys: `['P', 'phi0', 'ecc', 'omega']`, each as
            `astropy.units.Quantity` objects (i.e. with units).

        TODO
        ----
        - All prior distributions are essentially fixed. These should be
            customizable in some way...

        """
        rnd = self.random_state

        pars = dict()

        # sample from priors in nonlinear parameters
        pars['P'] = np.exp(rnd.uniform(np.log(self.params.P_min.to(u.day).value),
                                       np.log(self.params.P_max.to(u.day).value),
                                       size=size)) * u.day
        pars['phi0'] = rnd.uniform(0, 2*np.pi, size=size) * u.radian

        # MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
        pars['ecc'] = rnd.beta(a=0.867, b=3.03, size=size)
        pars['omega'] = rnd.uniform(0, 2*np.pi, size=size) * u.radian

        if not self.params._fixed_jitter:
            # Gaussian prior in log(s^2)
            log_s2 = rnd.normal(*self.params.jitter, size=size)
            pars['jitter'] = np.sqrt(np.exp(log_s2)) * self.params._jitter_unit

        else:
            pars['jitter'] = np.ones(size) * self.params.jitter

        return pars

    def _rejection_sample_from_cache(self, data, n_prior_samples, cache_file):
        """
        """

        # Get indices of good samples from the cache file
        # TODO: I have some implementation questions about whether this should return
        #   a boolean array (in which case I need to process all likelihood values) or
        #   an array of integers...Right now, _marginal_ll_worker has to return the values
        #   because we then compare with the maximum value of the likelihood
        good_samples_idx = get_good_sample_indices(n_prior_samples, cache_file, data,
                                                   self.params, pool=self.pool)
        if len(good_samples_idx) == 0:
            logger.error("Failed to find any good samples!")
            self.pool.close()
            sys.exit(0)

        n_good = len(good_samples_idx)
        logger.info("{} good samples after rejection sampling".format(n_good))

        # compute full parameter vectors for all good samples
        if self._rnd_passed:
            seed = self.random_state.randint(np.random.randint(2**16))
        else:
            seed = None

        full_samples = sample_indices_to_full_samples(good_samples_idx, cache_file,
                                                      data, self.params,
                                                      pool=self.pool, global_seed=seed)

        return full_samples

    def rejection_sample(self, data, n_prior_samples=None, prior_cache_file=None):
        """
        Parameters
        ----------
        data :
        n_prior_samples :
        prior_cache_file :
        """

        # validate input data
        if not isinstance(data, RVData):
            raise TypeError("Input data must be an RVData instance, not '{}'"
                            .format(type(data)))
        self.data = data

        if n_prior_samples is None and prior_cache_file is None:
            raise ValueError("You either have to specify the number of prior samples "
                             "to generate, or a path to a file containing cached prior "
                             "samples in (TODO: what format?). If you want to try an "
                             "experimental adaptive method, try .rejection_sample_adapt()")

        if prior_cache_file is not None:
            # read prior units from cache file
            with h5py.File(prior_cache_file, 'r') as f:
                prior_units = [u.Unit(uu) for uu in f.attrs['units']]
            samples = self._rejection_sample_from_cache(data, n_prior_samples, prior_cache_file)

        else:
            with tempfile.NamedTemporaryFile(mode='r+') as f:
                # first do prior sampling, cache to file
                prior_samples = self.sample_prior(size=n_prior_samples)
                prior_units = save_prior_samples(f.name, prior_samples, self.data.rv.unit)
                samples = self._rejection_sample_from_cache(data, n_prior_samples, f.name)

        return self.unpack_full_samples(samples, prior_units)

    def unpack_full_samples(self, samples, prior_units):
        """
        Unpack an array of Joker samples into a dictionary of Astropy
        Quantity objects (with units). Note that the phase of pericenter
        returned here is now relative to BMJD = 0.

        Parameters
        ----------
        samples : `numpy.ndarray`
            TODO
        prior_units : list
            List of units for the prior samples.

        Returns
        -------
        samples_dict : dict
            TODO

        """
        sample_dict = dict()

        n,n_params = samples.shape

        # TODO: need to keep track of this elsewhere...
        nonlin_params = ['P', 'phi0', 'ecc', 'omega', 'jitter']
        for k,key in enumerate(nonlin_params):
            sample_dict[key] = samples[:,k] * prior_units[k]

        k += 1
        sample_dict['K'] = samples[:,k] * prior_units[-1] # jitter unit

        k += 1
        for j in range(self.params.trend.n_terms):
            k += j
            sample_dict['v{}'.format(j)] = samples[:,k] * prior_units[-1] / u.day**j

        # convert phi0 from relative to t=data.t_offset to relative to mjd=0
        dphi = (2*np.pi*self.data.t_offset/sample_dict['P'].to(u.day).value * u.radian) % (2*np.pi*u.radian)
        sample_dict['phi0'] = (sample_dict['phi0'] + dphi) % (2*np.pi*u.radian)

        return sample_dict

    # def rejection_sample_adapt(self, data, min_n, prior_chunk_size=1024, max_prior_samples=2**24,
    #                            prior_cache_file=None):
    #     """
    #     """

    #     # TODO: always cache prior samples for simplicity?!

    #     if prior_cache_file is None and isinstance(self.pool, schwimmbad.SerialPool):
    #         maxiter = 1000 # TODO: make arg?
    #         iters = 0
    #         n_good_samples = 0
    #         n_prior_samples = prior_chunk_size
    #         while n_good_samples < n and n_prior_samples < max_prior_samples and iters < maxiter:
    #             prior_samples = self.sample_prior(size=n_prior_samples)
    #             good_samples = self._rejection_sample(prior_samples, data) # what else?

    #             if len(good_samples) > 1:
    #                 # TODO: save the samples
    #                 pass

    #             else:
    #                 # ignore the sample because only one was returned
    #                 pass

    #             n_prior_samples *= 2

    #         with tempfile.NamedTemporaryFile(mode='r+') as f:
    #             # TODO: generate prior samples, save to
    #             pass

    #     else:
    #         # need to read samples from filename
    #         pass
