# Third-party
import astropy.units as u
import numpy as np

# Project
from .prior_helpers import validate_poly_trend, validate_sigma_v
from .prior import JokerPrior, _validate_model, default_nonlinear_prior
from .samples import JokerSamples

__all__ = ['JokerSB2Prior']


class JokerSB2Prior(JokerPrior):
    _sb2 = True

    def __init__(self, pars=None, poly_trend=1, model=None):
        """
        This class controls the prior probability distributions for the
        parameters used in The Joker.

        TODO:

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
        model : `pymc3.Model`
            This is either required, or this function must be called within a
            pymc3 model context.

        """
        super().__init__(pars=pars, poly_trend=poly_trend, model=model)

    @classmethod
    def default(cls, P_min=None, P_max=None,
                sigma_K0_1=None, P0_1=1*u.year,
                sigma_K0_2=None, P0_2=1*u.year,
                sigma_v=None, s=None, poly_trend=1,
                model=None, pars=None):
        r"""
        An alternative initializer to set up the default prior for The Joker.

        The default prior is:

        .. math::

            p(P) \propto \frac{1}{P} \quad ; \quad P \in (P_{\rm min}, P_{\rm max})\\
            p(e) = B(a_e, b_e)\\
            p(\omega) = \mathcal{U}(0, 2\pi)\\
            p(M_0) = \mathcal{U}(0, 2\pi)\\
            p(s) = 0\\
            p(K) = \mathcal{N}(K \,|\, \mu_K, \sigma_K)\\
            \sigma_K = \sigma_{K, 0} \, \left(\frac{P}{P_0}\right)^{-1/3} \, \left(1 - e^2\right)^{-1/2}

        and the priors on any polynomial trend parameters are assumed to be
        independent, univariate Normals.

        This prior has sensible choices for typical binary star or exoplanet
        use cases, but if you need more control over the prior distributions
        you might need to use the standard initializer (i.e.
        ``JokerPrior(...)```) and specify all parameter distributions manually.
        See `the documentation <http://thejoker.readthedocs.io>`_ for tutorials
        that demonstrate this functionality.

        Parameters
        ----------
        P_min : `~astropy.units.Quantity` [time]
            Minimum period for the default period prior.
        P_max : `~astropy.units.Quantity` [time]
            Maximum period for the default period prior.
        sigma_K0 : `~astropy.units.Quantity` [speed]
            The scale factor, :math:`\sigma_{K, 0}` in the equation above that
            sets the scale of the semi-amplitude prior at the reference period,
            ``P0``.
        P0 : `~astropy.units.Quantity` [time]
            The reference period, :math:`P_0`, used in the prior on velocity
            semi-amplitude (see equation above).
        sigma_v : `~astropy.units.Quantity` (or iterable of)
            The standard deviations of the velocity trend priors.
        s : `~astropy.units.Quantity` [speed]
            The jitter value, assuming it is constant.
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. The
            default here is ``polytrend=1``, meaning one term: the (constant)
            systemtic velocity. For example, ``poly_trend=3`` will sample over
            parameters of a long-term quadratic velocity trend.
        v0_offsets : list (optional)
            A list of additional Gaussian parameters that set systematic offsets
            of subsets of the data. TODO: link to tutorial here
        model : `pymc3.Model` (optional)
            If not specified, this will create a model instance and store it on
            the prior object.
        pars : dict, list (optional)
            Either a list of pymc3 variables, or a dictionary of variables with
            keys set to the variable names. If any of these variables are
            defined as deterministic transforms from other variables, see the
            next parameter below.
        """

        model = _validate_model(model)

        nl_pars = default_nonlinear_prior(P_min, P_max, s=s,
                                          model=model, pars=pars)
        l_pars = default_linear_prior_sb2(sigma_K0_1=sigma_K0_1, P0_1=P0_1,
                                          sigma_K0_2=sigma_K0_2, P0_2=P0_2,
                                          sigma_v=sigma_v,
                                          poly_trend=poly_trend, model=model,
                                          pars=pars)

        pars = {**nl_pars, **l_pars}
        obj = cls(pars=pars, model=model, poly_trend=poly_trend)

        return obj

    def __repr__(self):
        return f'<JokerSB2Prior [{", ".join(self.par_names)}]>'

    def sample(self, size=1, generate_linear=False, return_logprobs=False,
               random_state=None, dtype=None, **kwargs):
        """
        Generate random samples from the prior.

        .. note::

            Right now, generating samples with the prior values is slow (i.e.
            with ``return_logprobs=True``) because of pymc3 issues (see
            discussion here:
            https://discourse.pymc.io/t/draw-values-speed-scaling-with-transformed-variables/4076).
            This will hopefully be resolved in the future...

        Parameters
        ----------
        size : int (optional)
            The number of samples to generate.
        generate_linear : bool (optional)
            Also generate samples in the linear parameters.
        return_logprobs : bool (optional)
            Generate the log-prior probability at the position of each sample.
        **kwargs
            Additional keyword arguments are passed to the
            `~thejoker.JokerSamples` initializer.

        Returns
        -------
        samples : `thejoker.Jokersamples`
            The random samples.

        """
        import exoplanet.units as xu

        raw_samples, sub_pars, log_prior = self._get_raw_samples(
            size, generate_linear, return_logprobs, random_state, dtype,
            **kwargs)

        if generate_linear:
            raw_samples['K2'] = (np.sign(raw_samples['K1']) *
                                 np.sign(raw_samples['K2']) *
                                 raw_samples['K2'])

        if generate_linear:
            par_names = self.par_names
        else:
            par_names = list(self._nonlinear_equiv_units.keys())

        # Split into primary / secondary:
        prior_samples = []
        for k in range(2):
            # Apply units if they are specified:
            s = JokerSamples(poly_trend=self.poly_trend,
                             n_offsets=self.n_offsets,
                             **kwargs)
            for name in par_names:
                p = sub_pars[name]
                unit = getattr(p, xu.UNIT_ATTR_NAME, u.one)

                if (p.name == 'K1' and k == 0) or (p.name == 'K2' and k == 1):
                    name = 'K'

                elif p.name not in s._valid_units.keys():
                    continue

                s[name] = np.atleast_1d(raw_samples[p.name]) * unit

            if return_logprobs:
                s['ln_prior'] = log_prior

            prior_samples.append(s)

        # Secondary argument of pericenter:
        prior_samples[1]['omega'] = prior_samples[1]['omega'] - 180*u.deg

        # TODO: right now, elsewhere, we assume the log_prior is a single value
        # for each sample (i.e. the total prior value). In principle, we could
        # store all of the individual log-prior values (for each parameter),
        # like here:
        # log_prior = {k: np.atleast_1d(v)
        #              for k, v in log_prior.items()}
        # log_prior = Table(log_prior)[par_names]

        return prior_samples


@u.quantity_input(sigma_K0=u.km/u.s, P0=u.day)
def default_linear_prior_sb2(sigma_K0_1=None, P0_1=None,
                             sigma_K0_2=None, P0_2=None,
                             sigma_v=None,
                             poly_trend=1, model=None, pars=None):
    r"""
    Retrieve pymc3 variables that specify the default prior on the linear
    parameters of The Joker. See docstring of `JokerPrior.default()` for more
    information.

    The linear parameters an default prior forms are:

    * ``K1``, velocity semi-amplitude for primary: Normal distribution, but with
      a variance that scales with period and eccentricity.
    * ``K2``, velocity semi-amplitude for secondary: Normal distribution, but
      with a variance that scales with period and eccentricity.
    * ``v0``, ``v1``, etc. polynomial velocity trend parameters: Independent
      Normal distributions.

    Parameters
    ----------
    sigma_K0_1 : `~astropy.units.Quantity` [speed]
    P0_1 : `~astropy.units.Quantity` [time]
    sigma_K0_2 : `~astropy.units.Quantity` [speed]
    P0_2 : `~astropy.units.Quantity` [time]
    sigma_v : iterable of `~astropy.units.Quantity`
    model : `pymc3.Model`
        This is either required, or this function must be called within a pymc3
        model context.
    """
    import pymc3 as pm
    import exoplanet.units as xu
    from .distributions import FixedCompanionMass

    model = pm.modelcontext(model)

    if pars is None:
        pars = dict()

    # dictionary of parameters to return
    out_pars = dict()

    # set up poly. trend names:
    poly_trend, v_names = validate_poly_trend(poly_trend)

    # get period/ecc from dict of nonlinear parameters
    P = model.named_vars.get('P', None)
    e = model.named_vars.get('e', None)
    if P is None or e is None:
        raise ValueError("Period P and eccentricity e must both be defined as "
                         "nonlinear parameters on the model.")

    if v_names and 'v0' not in pars:
        sigma_v = validate_sigma_v(sigma_v, poly_trend, v_names)

    with model:
        if 'K1' not in pars:
            if sigma_K0_1 is None or P0_1 is None:
                raise ValueError("If using the default prior form on K, you "
                                 "must pass in a variance scale (sigma_K0) "
                                 "and a reference period (P0) for both the "
                                 "primary and secondary")

            # Default prior on semi-amplitude: scales with period and
            # eccentricity such that it is flat with companion mass
            v_unit = sigma_K0_1.unit
            out_pars['K1'] = xu.with_unit(
                FixedCompanionMass('K1', P=P, e=e,
                                   sigma_K0=sigma_K0_1, P0=P0_1),
                v_unit)
        else:
            v_unit = getattr(pars['K1'], xu.UNIT_ATTR_NAME, u.one)

        if 'K2' not in pars:
            if sigma_K0_2 is None or P0_2 is None:
                raise ValueError("If using the default prior form on K, you "
                                 "must pass in a variance scale (sigma_K0) "
                                 "and a reference period (P0) for both the "
                                 "primary and secondary")

            # Default prior on semi-amplitude: scales with period and
            # eccentricity such that it is flat with companion mass
            v_unit = sigma_K0_1.unit
            out_pars['K2'] = xu.with_unit(
                FixedCompanionMass('K2', P=P, e=e,
                                   sigma_K0=sigma_K0_2, P0=P0_2),
                v_unit)
        else:
            v_unit = getattr(pars['K2'], xu.UNIT_ATTR_NAME, u.one)

        for i, name in enumerate(v_names):
            if name not in pars:
                # Default priors are independent gaussians
                # FIXME: make mean, mu_v, customizable
                out_pars[name] = xu.with_unit(
                    pm.Normal(name, 0.,
                              sigma_v[name].value),
                    sigma_v[name].unit)

    for k in pars.keys():
        out_pars[k] = pars[k]

    return out_pars
