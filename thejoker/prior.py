# Third-party
from astropy.utils.misc import isiterable
import astropy.units as u
import numpy as np
import pymc3 as pm
from pymc3.distributions import draw_values
import theano.tensor as tt
import exoplanet as xo
import exoplanet.units as xu

__all__ = ['JokerPrior']


class JokerPrior:

    def __init__(self, pars, unpars=None):
        """This class controls the prior probability distributions for the
        parameters used in The Joker.

        TODO: words
        TODO: note, use uniform_logP for old functionality

        Retrieve the prior parameters for the nonlinear Joker parameters as
        pymc3 variables. If not specified, most parameters have sensible
        defaults. However, specifying the period prior is required and must be
        done either by passing it in explicitly (i.e. to the ``pars`` argument),
        or by specifying the limits of the default prior (i.e. via ``P_lim``).

        Parameters
        ----------
        pars : dict, list (optional)
            Either a list of pymc3 variables, or a dictionary of variables with
            keys set to the variable names. If any of these variables are
            defined as deterministic transforms from other variables, see the
            next parameter below.
        unpars : dict (optional)
            For parameters that have defined deterministic transforms that go
            from the parameters used for sampling to the standard Joker
            nonlinear parameters (P, e, omega, M0), you must also pass in the
            un-transformed variables keyed on the name of the transformed
            parameters through this argument.

        Examples
        --------
        """

        # Parse and clean up the input
        # pars can be a dict or list
        if pars is None:
            pars = dict()

        elif isinstance(pars, tt.TensorVariable):  # a single variable
            # Note: this has to go before the next clause because TensorVariable
            # instances are iterable...
            pars = {pars.name: pars}

        else:
            try:
                pars = dict(pars)  # try to coerce to a dictionary
            except Exception:
                # if that fails, assume it is an iterable, like a list or tuple
                try:
                    pars = {p.name: p for p in pars}
                except Exception:
                    raise ValueError("Invalid input parameters: The input "
                                     "`pars` must either be a dictionary, "
                                     "list, or a single pymc3 variable, not a "
                                     "'{}'.".format(type(pars)))

        # unpars must be a dict
        if unpars is None:
            unpars = dict()

        else:
            try:
                unpars = dict(unpars)
            except Exception:
                raise ValueError("Invalid input for untransformed parameters: "
                                 "The input `unpars` must be a dictionary, not"
                                 " '{}'".format(type(unpars)))

        # User must specify a prior on period, P
        if 'P' not in pars:
            raise ValueError("TODO: you must ... use .uniform_logP")

        # Set up the default priors for parameters with defaults
        with pm.Model() as model:
            default_pars = {
                'e': xo.distributions.eccentricity.kipping13('e'),
                'omega': xu.with_unit(pm.Uniform('omega',
                                                 lower=0, upper=2*np.pi),
                                      u.radian),
                'M0': xu.with_unit(pm.Uniform('M0', lower=0, upper=2*np.pi),
                                   u.radian),
                'jitter': xu.with_unit(pm.Constant('jitter', 0.),
                                       u.m/u.s)  # TODO: default units
            }

        # Fill the parameter dictionary with defaults if not specified
        for k in default_pars.keys():
            pars[k] = pars.get(k, default_pars.get(k))
            unpars[k] = unpars.get(k, None)

        self.pars = pars
        self.unpars = unpars

    @classmethod
    @u.quantity_input(P_min=u.day, P_max=u.day)
    def uniform_logP(cls, P_min, P_max, **kwargs):
        pars = kwargs.pop('pars', dict())
        unpars = kwargs.pop('unpars', dict())

        # At this point, P is not in pars but P_lim has been specified:
        P_max = P_max.to(P_min.unit)
        with pm.Model() as model:
            unpars['P'] = pm.Uniform('logP',
                                     np.log10(P_min.value),
                                     np.log10(P_max.value))
            pars['P'] = xu.with_unit(pm.Deterministic('P', 10**unpars['P']),
                                     P_min.unit)

        kwargs['pars'] = pars
        kwargs['unpars'] = unpars

        return cls(**kwargs)

    def sample(self, size=1, return_logprobs=False):
        """Note: this is an internal function. To generate samples from the
        prior, use ``TheJoker.sample_prior()`` instead.

        Parameters
        ----------
        pars : dict
            A dictionary of variables with keys set to the variable names. If
            any of these variables are defined as deterministic transforms from
            other variables, see the next parameter below.
        unpars : dict (optional)
            For parameters that have defined deterministic transforms that go
            from the parameters used for sampling to the standard Joker
            nonlinear parameters (P, e, omega, M0), you must also pass in the
            un-transformed variables keyed on the name of the transformed
            parameters through this argument.
        size : int (optional)
        return_logprobs : bool (optional)
            Return the log-prior probability at the position of each sample, for
            each parameter separately
        """
        pars_list = list(self.pars.values())
        npars = len(pars_list)

        log_prior = []
        if return_logprobs:
            # Add deterministic variables to track the value of the prior at
            # each sample generated:
            with pm.Model() as model:
                for par in pars_list:
                    if (par.name in self.unpars.keys()
                            and self.unpars[par.name] is not None):
                        upar = self.unpars[par.name]
                        logp_var = pm.Deterministic(
                            f'{upar.name}_log_prior',
                            upar.distribution.logp(upar))
                    else:
                        logp_var = pm.Deterministic(
                            f'{par.name}_log_prior',
                            par.distribution.logp(par))
                    log_prior.append(logp_var)

        samples_values = draw_values(pars_list + log_prior, size=size)
        prior_samples = {p.name: samples
                         for p, samples in zip(pars_list,
                                               samples_values[:npars])}

        # Apply units if they are specified:
        for p in pars_list:
            if xu.has_unit(p):
                unit = getattr(p, xu.UNIT_ATTR_NAME)
            else:
                unit = u.one
            prior_samples[p.name] = prior_samples[p.name] * unit

        if not return_logprobs:
            return prior_samples

        log_prior_vals = {p.name: vals
                          for p, vals in zip(pars_list,
                                             samples_values[npars:])}

        return prior_samples, log_prior_vals
