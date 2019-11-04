# Third-party
import astropy.units as u
import numpy as np
import pymc3 as pm
import exoplanet as xo
import exoplanet.units as xu
import pytest
import theano.tensor as tt

# Project
from ..prior import JokerPrior, default_nonlinear_prior, default_linear_prior


def get_valid_objs():
    """This function also implicitly tests valid initialization schemes, but
    below we use it to parametrize some tests that require valid prior objects.
    """

    priors = []
    expected_units = []

    # default units expected for output
    default_expected_units = {'P': u.day, 'e': u.one,
                              'omega': u.radian, 'M0': u.radian,
                              'jitter': u.m/u.s,
                              'K': u.km/u.s, 'v0': u.km/u.s}

    # Default prior with standard parameters:
    priors.append(dict(P_min=1*u.day, P_max=100*u.day,
                       sigma_K0=25*u.km/u.s, sigma_v0=100*u.km/u.s))
    expected_units.append(default_expected_units)

    # Get default pars
    with pm.Model() as model:
        nl_pars, nl_unpars = default_nonlinear_prior(P_min=1*u.day,
                                                     P_max=1e5*u.day)
        l_pars, l_unpars = default_linear_prior(nl_pars,
                                                sigma_K0=25*u.km/u.s,
                                                sigma_v0=100*u.km/u.s)
    default_pars = {**nl_pars, **l_pars}
    default_unpars = {**nl_unpars, **l_unpars}

    # No transformed parameters
    pars = {}  # pars to replace
    unpars = {}
    units = default_expected_units.copy()
    with pm.Model() as model:
        pars['P'] = xu.with_unit(pm.Normal('P', 10, 0.5),
                                 u.year)
        units['P'] = u.year

        pars['e'] = xu.with_unit(pm.Uniform('e', 0, 1),
                                 u.one)

        pars['omega'] = xu.with_unit(pm.Normal('omega', 25., 1),
                                     u.degree)
        units['omega'] = u.degree

    for k in default_pars.keys():
        if k not in pars:
            pars[k] = default_pars[k]

            if k in default_unpars:
                unpars[k] = default_unpars[k]

    priors.append(dict(pars=pars))
    expected_units.append(units)

    priors.append(dict(pars=pars, unpars=unpars))
    expected_units.append(units)

    # Additional transformed parameter
    pars = {}  # pars to replace
    unpars = {}
    units = default_expected_units.copy()
    with pm.Model() as model:
        logs = pm.Normal('logs', -1, 0.5)
        jitter = xu.with_unit(pm.Deterministic('jitter', tt.exp(logs)),
                              u.km/u.s)
        unpars['jitter'] = logs
        pars['jitter'] = jitter
    units['jitter'] = u.km/u.s

    for k in default_pars.keys():
        if k not in pars:
            pars[k] = default_pars[k]

            if k in default_unpars:
                unpars[k] = default_unpars[k]

    priors.append(dict(pars=pars, unpars=unpars))
    expected_units.append(units)

    return priors, expected_units


@pytest.mark.parametrize('kw,expected_units', list(zip(*get_valid_objs())))
def test_init_sample(kw, expected_units):
    # Running this function is enough to test valid initialization schemes:
    if 'P_min' in kw:
        prior = JokerPrior.from_default(**kw)
    else:
        prior = JokerPrior(**kw)

    samples = prior.sample()
    for k in samples.param_names:
        assert hasattr(samples[k], 'unit')
        assert samples[k].unit == expected_units[k]
