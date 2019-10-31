# Third-party
import astropy.units as u
import numpy as np
import pymc3 as pm
import exoplanet as xo
import exoplanet.units as xu
import pytest
import theano.tensor as tt

# Project
from ..prior import JokerPrior


def get_valid_objs():
    """This function also implicitly tests valid initialization schemes, but
    below we use it to parametrize some tests that require valid prior objects.
    """

    priors = []
    expected_units = []

    # default units expected for output
    default_expected_units = {'P': u.day, 'e': u.one,
                              'omega': u.radian, 'M0': u.radian,
                              'jitter': u.m/u.s}

    # Test valid initialization patterns:
    priors.append(dict(P_lim=[1, 100] * u.day))  # legacy
    expected_units.append(default_expected_units)

    # No transformed parameters
    with pm.Model() as model:
        P = xu.with_unit(pm.Normal('P', 10, 0.5),
                         u.day)
        e = xu.with_unit(pm.Uniform('e', 0, 1),
                         u.one)
        omega = xu.with_unit(pm.Normal('omega', 0.5, 1),
                             u.radian)

    priors.append(dict(pars=P))
    expected_units.append(default_expected_units)

    priors.append(dict(pars={'P': P}))
    expected_units.append(default_expected_units)

    priors.append(dict(pars=[P, e]))
    expected_units.append(default_expected_units)

    priors.append(dict(pars=(P, e)))
    expected_units.append(default_expected_units)

    priors.append(dict(pars=[P, omega]))
    expected_units.append(default_expected_units)

    priors.append(dict(pars=[P, omega, e]))
    expected_units.append(default_expected_units)

    priors.append(dict(pars={'P': P, 'e': e, 'omega': omega}))
    expected_units.append(default_expected_units)

    with pm.Model() as model:
        # Unit-less eccentricity
        e_nounit = pm.Uniform('e', 0, 1)

        # Omega in different units:
        omega_deg = xu.with_unit(pm.Uniform('omega', 5, 20),
                                 u.degree)

    priors.append(dict(pars=[P, e_nounit]))
    expected_units.append(default_expected_units)

    priors.append(dict(pars=[P, omega_deg]))
    _units = default_expected_units.copy()
    _units['omega'] = u.deg
    expected_units.append(_units)

    # Transformed parameter
    with pm.Model() as model:
        logP = pm.Uniform('logP',
                          lower=np.log10(1),
                          upper=np.log10(1e5))
        P_tr = xu.with_unit(pm.Deterministic('P', 10**logP),
                            u.year)

        logs = pm.Normal('logs', -1, 0.5)
        jitter = xu.with_unit(pm.Deterministic('jitter', tt.exp(logs)),
                              u.km/u.s)

    _units = default_expected_units.copy()
    _units['P'] = u.year

    priors.append(dict(pars=P_tr, unpars={'P': logP}))
    expected_units.append(_units)

    priors.append(dict(pars=[P_tr], unpars={'P': logP}))
    expected_units.append(_units)

    priors.append(dict(pars=[P_tr, e, omega],
                       unpars={'P': logP}))
    expected_units.append(_units)

    priors.append(dict(pars={'P': P_tr, 'e': e},
                       unpars={'P': logP}))
    expected_units.append(_units)

    priors.append(dict(pars=[P_tr, e, omega, jitter],
                       unpars={'P': logP, 'jitter': logs}))
    _units = _units.copy()
    _units['jitter'] = u.km/u.s
    expected_units.append(_units)

    return priors, expected_units


@pytest.mark.parametrize('kw,expected_units', list(zip(*get_valid_objs())))
def test_init_sample(kw, expected_units):
    # Running this function is enough to test valid initialization schemes:
    prior = JokerPrior(**kw)

    samples = prior.sample()
    for k in samples.keys():
        assert hasattr(samples[k], 'unit')
        assert samples[k].unit == expected_units[k]
