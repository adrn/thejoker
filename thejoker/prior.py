# Third-party
from astropy.table import QTable, Table
import astropy.units as u
import numpy as np
import pymc3 as pm
from pymc3.distributions import draw_values
import theano.tensor as tt
import exoplanet as xo
import exoplanet.units as xu

# Project
from .samples import JokerSamples

__all__ = ['JokerPrior']


@u.quantity_input(P_min=u.day, P_max=u.day)
def default_nonlinear_prior(P_min, P_max, model=None):
    """Retrieve pymc3 variables that specify the default prior on the nonlinear
    parameters of The Joker.

    The nonlinear parameters an default prior forms are:

    * ``P``, period: :math:`p(P) \propto 1/P`, over the domain
      :math:`(P_{\rm min}, P_{\rm max})`
    * ``e``, eccentricity: the short-period form from Kipping (2013)
    * ``M0``, phase: uniform over the domain :math:`(0, 2\pi)`
    * ``omega``, argument of pericenter: uniform over the domain
      :math:`(0, 2\pi)`
    * ``s``, additional extra variance added in quadrature to data
      uncertainties: delta-function at 0

    Parameters
    ----------
    P_min : `~astropy.units.Quantity`
        Minimum period for the default 1/P prior.
    P_max : `~astropy.units.Quantity`
        Maximum period for the default 1/P prior.
    model : `pymc3.Model`
        This is either required, or this function must be called within a pymc3
        model context.
    """
    model = pm.modelcontext(model)

    pars = dict()
    unpars = dict()

    P_max = P_max.to(P_min.unit)
    with model:
        # Set up the default priors for parameters with defaults
        pars['e'] = xo.distributions.eccentricity.kipping13('e')
        pars['omega'] = xu.with_unit(pm.Uniform('omega',
                                                lower=0, upper=2*np.pi),
                                     u.radian)
        pars['M0'] =  xu.with_unit(pm.Uniform('M0', lower=0, upper=2*np.pi),
                                   u.radian)

        # TODO: these default units are a little sloppy, but it doesn't matter
        #       in practice...
        pars['s'] = xu.with_unit(pm.Constant('s', 0.),
                                 u.m/u.s)

        # Default period prior is uniform in log period:
        unpars['P'] = pm.Uniform('logP',
                                 np.log10(P_min.value),
                                 np.log10(P_max.value))
        pars['P'] = xu.with_unit(pm.Deterministic('P', 10**unpars['P']),
                                 P_min.unit)

    return pars, unpars


@u.quantity_input(sigma_K0=u.km/u.s, sigma_v0=u.km/u.s)
def default_linear_prior(nonlinear_pars, sigma_K0, sigma_v0, model=None):
    """Retrieve pymc3 variables that specify the default prior on the linear
    parameters of The Joker.

    The linear parameters an default prior forms are:

    * ``K``, velocity semi-amplitude: Normal distribution, but with a variance
      that scales with period and eccentricity such that:

      .. math::

        \sigma_K^2 = \sigma_{K, 0}^2 \, (P/1~{\rm year})^{-2/3} \, (1-e^2)^{-1}

    * ``v0``, systemic velocity: Normal distribution

    Parameters
    ----------
    nonlinear_pars : `dict`
        A dictionary with parameter name keys, and parameter object values.
    sigma_K0 : `~astropy.units.Quantity`
        The scale factor
    sigma_v0 : `~astropy.units.Quantity`
        The standard deviation of the constant velocity prior.
    model : `pymc3.Model`
        This is either required, or this function must be called within a pymc3
        model context.
    """
    model = pm.modelcontext(model)

    pars = dict()

    K_unit = sigma_K0.unit
    sigma_v0 = sigma_v0.to(K_unit)
    with model:
        # Default prior on semi-amplitude: scales with period and eccentricity
        # such that it is flat with companion mass
        P = nonlinear_pars['P']
        e = nonlinear_pars['e']
        varK = sigma_K0.value**2 * (P / 365)**(-2/3) / (1 - e**2)
        pars['K'] = xu.with_unit(pm.Normal('K', 0., tt.sqrt(varK)),
                                 K_unit)

        # Default prior on constant velocity is a single gaussian component
        pars['v0'] = xu.with_unit(pm.Normal('v0', 0., sigma_v0.value),
                                  K_unit)

    # return an empty dict for untransformed parameters, for consistency...
    return pars, dict()


class JokerPrior:

    # TODO: inside TheJoker, when sampling, validate that number of RVData's passed in equals the number of (offsets+1)
    # prior = JokerPrior.from_default(..., v0_offsets=[pm.Normal(...)])
    # joker = TheJoker(prior)
    # joker.rejection_sample([data1, data2], ...)
    def __init__(self, pars, unpars=None, poly_trend=1, v0_offsets=None,
                 model=None):
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
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. The
            default here is ``polytrend=1``, meaning one term: the (constant)
            systemtic velocity. For example, ``poly_trend=3`` will sample over
            parameters of a long-term quadratic velocity trend.

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

        # TODO: validate that this is a pymc3 model
        if model is None:
            model = pm.Model()
        self.model = model

        # Set the number of polynomial trend parameters
        self.poly_trend = int(poly_trend)

        # Store the names of the default parameters, used for validating input:
        # Note: these are not the units assumed internally by the code, but
        # are only used to validate that the units for each parameter are
        # equivalent to these
        self._nonlinear_params = {
            'P': u.day,
            'e': u.one,
            'omega': u.radian,
            'M0': u.radian,
            's': u.m/u.s,
        }

        self.poly_trend = int(poly_trend)
        self._linear_params = {
            'K': u.m/u.s,
            **{'v{0}'.format(i): u.m/u.s/u.day**i
               for i in range(self.poly_trend)}
        }

        # Enforce that the prior on linear parameters are gaussian
        for name in self._linear_params.keys():
            if not isinstance(pars[name].distribution, pm.Normal):
                raise ValueError("Priors on the linear parameters (K, v0, "
                                 "etc.) must be independent Normal "
                                 "distributions, not '{}'"
                                 .format(type(pars[name].distribution)))

        # TODO: internally, need to use these to construct the constant part of # the design matrix
        if v0_offsets is not None:
            try:
                v0_offsets = list(v0_offsets)
            except Exception:
                raise TypeError("Constant velocity offsets must be an iterable "
                                "of pymc3 variables that define the priors on "
                                "each offset term.")

            for offset in v0_offsets:
                if not isinstance(offset.distribution, pm.Normal):
                    raise ValueError("Priors on the constant offset parameters "
                                     "must be independent Normal "
                                     "distributions, not '{}'"
                                     .format(type(offset.distribution)))
        self.v0_offsets = v0_offsets

        self.pars = pars
        self.unpars = unpars

    @classmethod
    def from_default(cls, P_min, P_max, sigma_K0, sigma_v0, model=None):
        if model is None:
            model = pm.Model()

        nl_pars, nl_unpars = default_nonlinear_prior(P_min, P_max, model=model)
        l_pars, l_unpars = default_linear_prior(nl_pars, sigma_K0, sigma_v0,
                                                model=model)

        pars = {**nl_pars, **l_pars}
        unpars = {**nl_unpars, **l_unpars}

        return cls(pars=pars, unpars=unpars, model=model)

    @property
    def param_names(self):
        return (list(self._nonlinear_params.keys()) +
                     list(self._linear_params.keys()))

    @property
    def param_units(self):
        return {p.name: getattr(p, xu.UNIT_ATTR_NAME, u.one) for p in self.pars}

    def sample(self, size=1, return_logprobs=False):
        """TODO

        Parameters
        ----------
        size : int (optional)
            The number of samples to generate.
        return_logprobs : bool (optional)
            Return the log-prior probability at the position of each sample, for
            each parameter separately

        Returns
        -------
        samples : `thejoker.Jokersamples`
            TODO

        """
        pars_list = list(self.pars.values())
        npars = len(pars_list)

        log_prior = []
        if return_logprobs:
            # Add deterministic variables to track the value of the prior at
            # each sample generated:
            with self.model:
                for par in pars_list:
                    if (par.name in self.unpars.keys()
                            and self.unpars[par.name] is not None):
                        upar = self.unpars[par.name]
                        logp_name = f'{upar.name}_log_prior'
                        dist = upar.distribution.logp(upar)

                    else:
                        logp_name = f'{par.name}_log_prior'
                        dist = par.distribution.logp(par)

                    if logp_name in self.model.named_vars:
                        logp_var = self.model.named_vars[logp_name]
                    else:
                        # doesn't exist in the model yet, so add it
                        logp_var = pm.Deterministic(logp_name, dist)

                    log_prior.append(logp_var)

        samples_values = draw_values(pars_list + log_prior, size=size)
        raw_samples = {p.name: samples
                       for p, samples in zip(pars_list,
                                             samples_values[:npars])}

        # Apply units if they are specified:
        prior_samples = JokerSamples(prior=self)
        for name in self.param_names:
            p = self.pars[name]
            unit = getattr(p, xu.UNIT_ATTR_NAME, u.one)

            if p.name not in prior_samples._valid_units.keys():
                continue

            prior_samples[p.name] = np.atleast_1d(raw_samples[p.name]) * unit

        if not return_logprobs:
            return prior_samples

        log_prior = {p.name: vals for p, vals in zip(pars_list,
                                                     samples_values[npars:])}
        log_prior = {k: np.atleast_1d(v)
                        for k, v in log_prior.items()}
        log_prior = Table(log_prior)[self.param_names]

        return prior_samples, log_prior
