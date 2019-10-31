# Third-party
from astropy.utils.misc import isiterable
import astropy.units as u
import numpy as np
import pymc3 as pm
from pymc3.distributions import draw_values
import theano.tensor as tt
import exoplanet as xo
import exoplanet.units as xu


class JokerPrior:

    def __init__(self, pars=None, unpars=None, P_lim=None):
        """
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
        P_lim : iterable (optional)
            If the period prior is not specified explicitly, this sets the
            bounds of the period prior, assumed to be proportional to 1/P
            (uniform in log(P)).

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

        # Initialize the default prior:
        # - P is special, we allow passing in a range, assuming 1/P period:
        pars['P'], unpars['P'] = self._get_P(pars, unpars, P_lim)

        # Now set up the default priors on
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

    @staticmethod
    @u.quantity_input(P_lim=u.day)
    def _get_P(pars, unpars, P_lim=None):
        """Note: this is an internal function.

        Retrieve the period prior as a pymc3 variable given semi-flexible input.
        """

        if 'P' in pars and P_lim is not None:
            raise ValueError("Period appears in the input parameters, but "
                             "period limits are also specified (via P_lim). "
                             "Specify one or the other, but on both.")

        elif 'P' not in pars and P_lim is None:
            raise ValueError("The period prior requires some specification: "
                            "Either pass in an explicit pymc3 variable with a "
                            "defined distribution, or pass in limits, P_lim, "
                            "for an assumed (default) prior proportional "
                            "to 1/P")

        # Short-circuit if P is already defined
        if 'P' in pars:
            return pars['P'], unpars.get('P', None)

        # At this point, P is not in pars but P_lim has been specified:
        with pm.Model() as model:
            logP_kw = dict()
            if P_lim is not None:
                logP_kw['lower'] = np.log10(P_lim.value[0])
                logP_kw['upper'] = np.log10(P_lim.value[1])

            logP = pm.Uniform('logP', **logP_kw)
            default_P = xu.with_unit(pm.Deterministic('P', 10**logP),
                                     P_lim.unit)

        return default_P, logP


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
