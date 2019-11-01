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


@u.quantity_input(P_min=u.day, P_max=u.day)
def default_nonlinear_prior(P_min, P_max, model=None):

    pars = dict()
    unpars = dict()

    P_max = P_max.to(P_min.unit)
    with pm.Model(model=model):
        # Set up the default priors for parameters with defaults
        pars['e'] = xo.distributions.eccentricity.kipping13('e')
        pars['omega'] = xu.with_unit(pm.Uniform('omega',
                                                lower=0, upper=2*np.pi),
                                     u.radian)
        pars['M0'] =  xu.with_unit(pm.Uniform('M0', lower=0, upper=2*np.pi),
                                   u.radian)

        # TODO: these default units are a little sloppy, but it doesn't matter
        #       in practice...
        pars['jitter'] = xu.with_unit(pm.Constant('jitter', 0.),
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
    pars = dict()

    K_unit = sigma_K0.unit
    sigma_v0 = sigma_v0.to(K_unit)
    with pm.Model(model=model):
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

    # TODO: offsets can be additional Gaussian parameters that specify the
    # number of offsets!
    # TODO: inside TheJoker, when sampling, validate that number of RVData's passed in equals the number of (offsets+1)
    # prior = JokerPrior.from_default(..., v0_offsets=[pm.Normal(...)])
    # joker = TheJoker(prior)
    # joker.rejection_sample([data1, data2], ...)
    def __init__(self, pars, unpars=None, poly_trend=1, v0_offsets=None):
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

        # Set the number of polynomial trend parameters
        self.poly_trend = int(poly_trend)

        # Store the names of the default parameters, used for validating input:
        self._nonlinear_param_names = ['P', 'M0', 'e', 'omega', 'jitter']

        self.poly_trend = int(poly_trend)
        self._linear_param_names = ['K'] + ['v{0}'.format(i)
                                            for i in range(self.poly_trend)]

        # Enforce that the prior on linear parameters are gaussian
        for name in self.param_names[5:]:
            if not isinstance(pars[name].distribution, pm.Normal):
                raise ValueError("Priors on the linear parameters (K, v0, "
                                 "etc.) must be independent Normal "
                                 "distributions, not '{}'"
                                 .format(type(pars[name].distribution)))

        #
        self.v0_offsets = v0_offsets

        self.pars = pars
        self.unpars = unpars

    @classmethod
    def from_default(cls, P_min, P_max, sigma_K0, sigma_v0):
        nl_pars, nl_unpars = default_nonlinear_prior(P_min, P_max)
        l_pars, l_unpars = default_linear_prior(nl_pars, sigma_K0, sigma_v0)

        pars = {**nl_pars, **l_pars}
        unpars = {**nl_unpars, **l_unpars}

        return cls(pars=pars, unpars=unpars)

    @property
    def param_names(self):
        return self._nonlinear_param_names + self._linear_param_names

    def sample(self, size=1, return_logprobs=False):
        """TODO

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
            The number of samples to generate.
        return_logprobs : bool (optional)
            Return the log-prior probability at the position of each sample, for
            each parameter separately

        Returns
        -------
        samples : dict
            TODO

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
