# Third-party
from astropy.table import Table
import astropy.units as u
import numpy as np
import pymc3 as pm
from pymc3.distributions import draw_values
import theano.tensor as tt
import exoplanet as xo
import exoplanet.units as xu

# Project
from .samples import JokerSamples
from .prior_helpers import UniformLog, FixedCompanionMass

__all__ = ['JokerPrior']


@u.quantity_input(P_min=u.day, P_max=u.day)
def default_nonlinear_prior(P_min, P_max, model=None, pars=None):
    r"""Retrieve pymc3 variables that specify the default prior on the nonlinear
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

    if pars is None:
        pars = dict()

    out_pars = dict()

    P_max = P_max.to(P_min.unit)
    with model:
        # Set up the default priors for parameters with defaults

        # Note: we have to do it this way (as opposed to with .get(...,
        # default)because this can only get executed if the param is not already
        # defined, otherwise a pymc3 error is thrown
        if 'e' in pars:
            out_pars['e'] = pars['e']
        else:
            out_pars['e'] = xo.distributions.eccentricity.kipping13('e')

        if 'omega' in pars:
            out_pars['omega'] = pars['omega']
        else:

            out_pars['omega'] = xu.with_unit(pm.Uniform('omega',
                                                        lower=0,
                                                        upper=2*np.pi),
                                             u.radian)

        if 'M0' in pars:
            out_pars['M0'] = pars['M0']
        else:
            out_pars['M0'] = xu.with_unit(pm.Uniform('M0',
                                                     lower=0, upper=2*np.pi),
                                          u.radian)

        if 's' in pars:
            out_pars['s'] = pars['s']
        else:
            # These default units are a little sloppy, but it's 0 so ...
            out_pars['s'] = xu.with_unit(pm.Constant('s', 0.), u.m/u.s)

        if 'P' in pars:
            out_pars['P'] = pars['P']
        else:
            # Default period prior is uniform in log period:
            out_pars['P'] = xu.with_unit(UniformLog('P',
                                                    P_min.value,
                                                    P_max.value),
                                         P_min.unit)

    return out_pars


@u.quantity_input(sigma_K0=u.km/u.s)
def default_linear_prior(nonlinear_pars, sigma_K0, P0, sigma_v, poly_trend=1,
                         mu_v=None, model=None, pars=None):
    r"""Retrieve pymc3 variables that specify the default prior on the linear
    parameters of The Joker.

    The linear parameters an default prior forms are:

    * ``K``, velocity semi-amplitude: Normal distribution, but with a variance
      that scales with period and eccentricity such that:

      .. math::

        \sigma_K^2 = \sigma_{K, 0}^2 \, (P/P_0)^{-2/3} \, (1-e^2)^{-1}

    * ``v0``, ``v1``, etc. polynomial velocity trend parameters: Independent
      Normal distributions.

    Parameters
    ----------
    nonlinear_pars : `dict`
        A dictionary with parameter name keys, and parameter object values.
    sigma_K0 : `~astropy.units.Quantity` [speed]
        The scale factor, :math:`\sigma_{K, 0}` in the equation above.
    P0 : `~astropy.units.Quantity` [time]
        The reference period, :math:`P_0` in the equation above.
    sigma_v : iterable of `~astropy.units.Quantity`
        The standard deviations of the velocity trend priors.
    model : `pymc3.Model`
        This is either required, or this function must be called within a pymc3
        model context.
    """
    model = pm.modelcontext(model)

    poly_trend = int(poly_trend)
    v_names = ['v{0}'.format(i) for i in range(poly_trend)]

    if isinstance(sigma_v, u.Quantity):
        if not sigma_v.isscalar:
            raise ValueError("If poly_trend=1, i.e. the velocity trend "
                             "model is a constant velocity offset only, "
                             "you must pass in a scalar value for sigma_v")
        sigma_v = (sigma_v, )

    try:
        if len(sigma_v) != poly_trend:
            raise ValueError("You must pass in a single sigma value for each "
                             "velocity trend parameter: You passed in "
                             "{} values, but poly_trend={}"
                             .format(len(sigma_v), poly_trend))

    except TypeError:
        raise TypeError("Invalid input for velocity trend prior sigma "
                        "values. This must either be a scalar Quantity (if "
                        "poly_trend=1) or an iterable of Quantity objects "
                        "(if poly_trend>1)")

    # TODO: validate if not None! Currently, this just trusts...
    if mu_v is None:
        mu_v = [0*u.m/u.s/u.day**i for i in range(poly_trend)]

    if pars is None:
        pars = dict()

    out_pars = dict()

    with model:
        if 'K' in pars:
            out_pars['K'] = pars['K']
        else:
            # Default prior on semi-amplitude: scales with period and
            # eccentricity such that it is flat with companion mass
            P = nonlinear_pars['P']
            e = nonlinear_pars['e']
            out_pars['K'] = xu.with_unit(FixedCompanionMass('K', P=P, e=e,
                                                            sigma_K0=sigma_K0,
                                                            P0=P0),
                                         sigma_K0.unit)

        P_unit = getattr(P, xu.UNIT_ATTR_NAME)
        for i, name in enumerate(v_names):
            if name in pars:
                out_pars[name] = pars[name]
            else:
                unit = sigma_K0.unit / P_unit**i

                # Default priors are independent gaussians
                # TODO: FIXME: make mean (0. below) customizable
                out_pars[name] = xu.with_unit(
                    pm.Normal(name,
                              mu_v[i].to_value(unit),
                              sigma_v[i].to_value(unit)),
                    unit)

    # return an empty dict for untransformed parameters, for consistency...
    return out_pars


class JokerPrior:

    def __init__(self, pars, poly_trend=1, v0_offsets=None, model=None):
        """This class controls the prior probability distributions for the
        parameters used in The Joker.

        This initializer is meant to be flexible, allowing you to specify the
        prior distributions on the linear and nonlinear parameters used in The
        Joker. However, for many use cases, you may want to just use the
        default prior: To initialize this object using the default prior, see
        the alternate initializer `JokerPrior.default()`.

        Parameters
        ----------
        pars : dict, list (optional)
            Either a list of pymc3 variables, or a dictionary of variables with
            keys set to the variable names. If any of these variables are
            defined as deterministic transforms from other variables, see the
            next parameter below.
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. The
            default here is ``polytrend=1``, meaning one term: the (constant)
            systemtic velocity. For example, ``poly_trend=3`` will sample over
            parameters of a long-term quadratic velocity trend.
        v0_offsets : list (optional)
            A list of additional Gaussian parameters that set systematic offsets
            of subsets of the data. TODO: link to tutorial here
        model : `pymc3.Model`
            This is either required, or this function must be called within a
            pymc3 model context.

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

        # TODO: validate that input pars have units!

        # validate input model
        if model is None:
            try:
                # check to see if we are in a context
                model = pm.modelcontext(None)
            except TypeError:  # we are not!
                # if no model is specified, create one and hold onto it
                model = pm.Model()
        self.model = model
        if not isinstance(self.model, pm.Model):
            raise TypeError("Input model must be a pymc3.Model instance, not "
                            "a {}".format(type(self.model)))

        # Set the number of polynomial trend parameters
        self.poly_trend = int(poly_trend)

        # Store the names of the default parameters, used for validating input:
        # Note: these are not the units assumed internally by the code, but
        # are only used to validate that the units for each parameter are
        # equivalent to these
        self._nonlinear_pars = {
            'P': u.day,
            'e': u.one,
            'omega': u.radian,
            'M0': u.radian,
            's': u.m/u.s,
        }

        self.poly_trend = int(poly_trend)
        self._linear_pars = {
            'K': u.m/u.s,
            **{'v{0}'.format(i): u.m/u.s/u.day**i
               for i in range(self.poly_trend)}
        }

        # Enforce that the prior on linear parameters are gaussian
        for name in self._linear_pars.keys():
            if not isinstance(pars[name].distribution, pm.Normal):
                raise ValueError("Priors on the linear parameters (K, v0, "
                                 "etc.) must be independent Normal "
                                 "distributions, not '{}'"
                                 .format(type(pars[name].distribution)))

        # TODO: enable support for this
        if v0_offsets is not None:
            # TODO: FIXME
            raise NotImplementedError("Support for this is coming - sorry!")

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

    @classmethod
    def default(cls, P_min, P_max, sigma_K0, sigma_v, P0=1*u.year, mu_v=None,
                poly_trend=1, model=None):
        """TODO:"""

        if model is None:
            model = pm.Model()

        nl_pars = default_nonlinear_prior(P_min, P_max, model=model)
        l_pars = default_linear_prior(nl_pars, sigma_K0=sigma_K0, P0=P0,
                                      sigma_v=sigma_v, mu_v=mu_v,
                                      poly_trend=poly_trend, model=model)

        pars = {**nl_pars, **l_pars}
        obj = cls(pars=pars, model=model, poly_trend=poly_trend)

        return obj

    @property
    def par_names(self):
        return (list(self._nonlinear_pars.keys()) +
                list(self._linear_pars.keys()))

    @property
    def par_units(self):
        return {p.name: getattr(p, xu.UNIT_ATTR_NAME, u.one) for p in self.pars}

    def __repr__(self):
        return f'<JokerPrior [{", ".join(self.par_names)}]>'

    def __str__(self):
        return ", ".join(self.par_names)

    def sample(self, size=1, generate_linear=False, return_logprobs=False):
        """Generate random samples from the prior.

        Parameters
        ----------
        size : int (optional)
            The number of samples to generate.
        generate_linear : bool (optional)
            Also generate samples in the linear parameters.
        return_logprobs : bool (optional)
            Return the log-prior probability at the position of each sample, for
            each parameter separately

        Returns
        -------
        samples : `thejoker.Jokersamples`
            The random samples.
        log_prior : `astropy.table.Table` (optional)
            The log-prior probability at the position of each sample. This is
            only returned if ``return_logprobs=True``.

        """
        sub_pars = {k: p for k, p in self.pars.items()
                    if k in self._nonlinear_pars
                    or (k in self._linear_pars and generate_linear)}

        if generate_linear:
            # TODO: we could warn that this is usually slow because of pymc3?
            par_names = self.par_names
        else:
            par_names = list(self._nonlinear_pars.keys())

        pars_list = list(sub_pars.values())
        npars = len(pars_list)

        log_prior = []
        if return_logprobs:
            # TODO: warn that right now, this is slow. Waiting for upstream
            # fixes to pymc3

            # Add deterministic variables to track the value of the prior at
            # each sample generated:
            with self.model:
                for par in pars_list:
                    logp_name = f'{par.name}_log_prior'
                    dist = par.distribution.logp(par)

                    if logp_name in self.model.named_vars.keys():
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
        for name in par_names:
            p = sub_pars[name]
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
        log_prior = Table(log_prior)[par_names]

        return prior_samples, log_prior
