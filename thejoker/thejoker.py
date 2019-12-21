# Standard library
import os

# Third-party
import numpy as np

# Project
from .logging import logger
from .data_helpers import validate_prepare_data
from .samples_helpers import is_P_unimodal
from .likelihood_helpers import get_trend_design_matrix
from .prior_helpers import validate_n_offsets, validate_poly_trend
from .prior import JokerPrior, _validate_model
from .src.fast_likelihood import CJokerHelper
from .utils import tempfile_decorator
from .samples import JokerSamples

__all__ = ['TheJoker']


class TheJoker:
    """
    A custom Monte-Carlo sampler for two-body systems.

    Parameters
    ----------
    prior : `~thejoker.JokerPrior`
        The specification of the prior probability distribution over all
        parameters used in The Joker.
    pool : `schwimmbad.BasePool` (optional)
        A processing pool (default is a `schwimmbad.SerialPool` instance).
    random_state : `numpy.random.RandomState` (optional)
        A `numpy.random.RandomState` instance for controlling random number
        generation.
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
        all_data, ids, trend_M = validate_prepare_data(data,
                                                       self.prior.poly_trend,
                                                       self.prior.n_offsets)
        joker_helper = CJokerHelper(all_data, self.prior, trend_M)
        return joker_helper

    def marginal_ln_likelihood(self, data, prior_samples, n_batches=None,
                               in_memory=False):
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
        in_memory : bool (optional)
            Load all prior samples or keep all prior samples in memory and run
            all calculations without creating a temporary cache file.

        Returns
        -------
        ln_likelihood : `numpy.ndarray`
            The marginal log-likelihood computed at the location of each prior
            sample.
        """
        from .multiproc_helpers import marginal_ln_likelihood_helper
        from .likelihood_helpers import marginal_ln_likelihood_inmem

        joker_helper = self._make_joker_helper(data)  # also validates data

        if in_memory:
            if isinstance(prior_samples, JokerSamples):
                prior_samples, _ = prior_samples.pack(
                    units=joker_helper.internal_units,
                    names=joker_helper.packed_order)
            return marginal_ln_likelihood_inmem(joker_helper, prior_samples)

        else:
            return marginal_ln_likelihood_helper(joker_helper, prior_samples,
                                                 pool=self.pool,
                                                 n_batches=n_batches)

    def rejection_sample(self, data, prior_samples,
                         n_prior_samples=None,
                         max_posterior_samples=None,
                         n_linear_samples=1,
                         return_logprobs=False,
                         n_batches=None,
                         randomize_prior_order=False,
                         in_memory=False):
        f"""
        Run The Joker's rejection sampling on prior samples to get posterior
        samples for the input data.

        You must either specify the number of prior samples to generate and
        use for rejection sampling, ``n_prior_samples``, or the path to a file
        containing prior samples, ``prior_cache_file``.

        Parameters
        ----------
        data : `~thejoker.RVData`
            The radial velocity data, or an iterable containing ``RVData``
            objects for each data source.
        prior_samples : str, `~thejoker.JokerSamples`, int
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
        in_memory : bool (optional)
            Load all prior samples or keep all prior samples in memory and run
            all calculations without creating a temporary cache file.

        Returns
        -------
        samples : `~thejoker.JokerSamples`
            The posterior samples produced from The Joker.

        """
        from .multiproc_helpers import rejection_sample_helper
        from .likelihood_helpers import rejection_sample_inmem

        joker_helper = self._make_joker_helper(data)  # also validates data

        if isinstance(prior_samples, int):
            # If an integer, generate that many prior samples first
            N = prior_samples
            prior_samples = self.prior.sample(size=N,
                                              return_logprobs=return_logprobs)

        if in_memory:
            if isinstance(prior_samples, JokerSamples):
                ln_prior = None
                if return_logprobs:
                    ln_prior = prior_samples['ln_prior']

                prior_samples, _ = prior_samples.pack(
                    units=joker_helper.internal_units,
                    names=joker_helper.packed_order)
            else:
                ln_prior = return_logprobs

            samples = rejection_sample_inmem(
                joker_helper,
                prior_samples,
                random_state=self.random_state,
                ln_prior=ln_prior,
                max_posterior_samples=max_posterior_samples,
                n_linear_samples=n_linear_samples)

        else:
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
                                   max_prior_samples=None,
                                   n_linear_samples=1,
                                   return_logprobs=False,
                                   n_batches=None,
                                   randomize_prior_order=False,
                                   init_batch_size=None,
                                   growth_factor=128,
                                   in_memory=False):

        """This is an experimental sampling method that adaptively generates
        posterior samples given a large library of prior samples. The advantage
        of this function over the standard ``rejection_sample`` method is that
        it will try to adaptively figure out how many prior samples it needs to
        evaluate the likelihood at in order to return the desired number of
        posterior samples.

        Parameters
        ----------
        data : `~thejoker.RVData`
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
        in_memory : bool (optional)
            Load all prior samples or keep all prior samples in memory and run
            all calculations without creating a temporary cache file.

        Returns
        -------
        samples : `~thejoker.JokerSamples`
            The posterior samples produced from The Joker.
        """
        from .multiproc_helpers import iterative_rejection_helper
        from .likelihood_helpers import iterative_rejection_inmem

        joker_helper = self._make_joker_helper(data)  # also validates data

        if in_memory:
            if isinstance(prior_samples, JokerSamples):
                ln_prior = None
                if return_logprobs:
                    ln_prior = prior_samples['ln_prior']

                prior_samples, _ = prior_samples.pack(
                    units=joker_helper.internal_units,
                    names=joker_helper.packed_order)
            else:
                ln_prior = return_logprobs

            samples = iterative_rejection_inmem(
                joker_helper,
                prior_samples,
                random_state=self.random_state,
                n_requested_samples=n_requested_samples,
                ln_prior=ln_prior,
                init_batch_size=init_batch_size,
                growth_factor=growth_factor,
                n_linear_samples=n_linear_samples)

        else:
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

    def setup_mcmc(self, data, joker_samples, model=None, use_cached=False):
        """
        Setup the model to run MCMC using pymc3.

        Parameters
        ----------
        data : `~thejoker.RVData`
            The radial velocity data, or an iterable containing ``RVData``
            objects for each data source.
        joker_samples : `~thejoker.JokerSamples`
            If a single sample is passed in, this is packed into a pymc3
            initialization dictionary and returned after setting up. If
            multiple samples are passed in, the median (along period) sample is
            taken and returned after setting up for MCMC.
        model : `pymc3.Model`
            This is either required, or this function must be called within a
            pymc3 model context.

        Returns
        -------
        mcmc_init : dict

        """
        import pymc3 as pm
        import exoplanet as xo
        import exoplanet.units as xu
        import theano.tensor as tt

        model = _validate_model(model)

        # Reduce data, strip units:
        data, ids, _ = validate_prepare_data(data,
                                             self.prior.poly_trend,
                                             self.prior.n_offsets)
        x = data._t_bmjd - data._t0_bmjd
        y = data.rv.value
        err = data.rv_err.to_value(data.rv.unit)

        # First, prepare the joker_samples:
        if not isinstance(joker_samples, JokerSamples):
            raise TypeError("You must pass in a JokerSamples instance to the "
                            "joker_samples argument.")

        if len(joker_samples) > 1:
            # check if unimodal in P, if not, warn
            if not is_P_unimodal(joker_samples, data):
                logger.warn("TODO: samples ain't unimodal")

            joker_samples = joker_samples.median()

        mcmc_init = dict()
        for name in self.prior.par_names:
            unit = getattr(self.prior.pars[name], xu.UNIT_ATTR_NAME)
            if joker_samples[name].shape == ():
                mcmc_init[name] = joker_samples[name].to_value(unit)
            else:
                mcmc_init[name] = joker_samples[name].to_value(unit)[0]

        if 't_peri' in model.named_vars and use_cached:
            logger.debug('pymc3 model has already been setup for running MCMC '
                         '- using the previously setup model parameters.')
            return mcmc_init

        # remove previously-added parameters:
        names = ['t_peri', 'obs']
        to_remove = []
        for name in names:
            for i, par in enumerate(model.vars):
                if par.name == name:
                    to_remove.append(par)
                    del model.named_vars[name]
        [model.vars.remove(p) for p in to_remove]

        p = self.prior.pars
        with model:
            t_peri = pm.Deterministic('t_peri',
                                      p['P'] * p['M0'] / (2*np.pi))

            # Set up the orbit model
            orbit = xo.orbits.KeplerianOrbit(period=p['P'],
                                             ecc=p['e'],
                                             omega=p['omega'],
                                             t_periastron=t_peri)

        # design matrix
        M = get_trend_design_matrix(data, ids, self.prior.poly_trend)

        # deal with v0_offsets, trend here:
        _, offset_names = validate_n_offsets(self.prior.n_offsets)
        _, vtrend_names = validate_poly_trend(self.prior.poly_trend)

        with model:
            v_pars = ([p['v0']]
                      + [p[name] for name in offset_names]
                      + [p[name] for name in vtrend_names[1:]])  # skip v0
            v_trend_vec = tt.stack(v_pars, axis=0)
            trend = tt.dot(M, v_trend_vec)

            rv_model = orbit.get_radial_velocity(x, K=p['K']) + trend

            err = tt.sqrt(err**2 + p['s']**2)
            pm.Normal("obs", mu=rv_model, sd=err, observed=y)

        return mcmc_init

    def trace_to_samples(self, trace, data, names=None):
        """
        Create a ``JokerSamples`` instance from a pymc3 trace object.

        Parameters
        ----------
        trace : `~pymc3.backends.base.MultiTrace`
        """
        import pymc3 as pm
        import exoplanet.units as xu

        df = pm.trace_to_dataframe(trace)

        data, *_ = validate_prepare_data(data,
                                         self.prior.poly_trend,
                                         self.prior.n_offsets)

        samples = JokerSamples(poly_trend=self.prior.poly_trend,
                               n_offsets=self.prior.n_offsets,
                               t0=data.t0)

        if names is None:
            names = self.prior.par_names

        for name in names:
            par = self.prior.pars[name]
            unit = getattr(par, xu.UNIT_ATTR_NAME)
            samples[name] = df[name].values * unit

        return samples
