# Third-party
import astropy.units as u
import pymc3 as pm
import exoplanet.units as xu
import pytest

# Project
from ..prior import JokerPrior, default_nonlinear_prior, default_linear_prior


def init_helper(case=None):
    default_expected_units = {'P': u.day, 'e': u.one,
                              'omega': u.radian, 'M0': u.radian,
                              's': u.m/u.s,
                              'K': u.km/u.s, 'v0': u.km/u.s}

    if case == 0:
        # Default prior with standard parameters:
        with pm.Model() as model:
            default_nonlinear_prior(P_min=1*u.day, P_max=10*u.year)
            default_linear_prior(sigma_K0=25*u.km/u.s,
                                 P0=1*u.year,
                                 sigma_v=10*u.km/u.s)
            prior = JokerPrior(model=model)

        return prior, default_expected_units

    elif case == 1:
        # Replace a nonlinear parameter
        units = default_expected_units.copy()
        with pm.Model() as model:
            P = xu.with_unit(pm.Normal('P', 10, 0.5),
                             u.year)
            units['P'] = u.year

            default_nonlinear_prior(pars={'P': P})
            default_linear_prior(sigma_K0=25*u.km/u.s,
                                 P0=1*u.year,
                                 sigma_v=10*u.km/u.s)

        prior = JokerPrior(model=model)

        return prior, units

    elif case == 2:
        # Replace a linear parameter
        units = default_expected_units.copy()
        with pm.Model() as model:
            K = xu.with_unit(pm.Normal('K', 10, 0.5),
                             u.m/u.s)
            units['K'] = u.m/u.s

            default_nonlinear_prior(P_min=1*u.day, P_max=10*u.day)
            default_linear_prior(sigma_v=10*u.km/u.s,
                                 pars={'K': K})

        prior = JokerPrior(model=model)

        return prior, units

    elif case == 3:
        # Pass pars instead of relying on model
        with pm.Model() as model:
            nl_pars = default_nonlinear_prior(P_min=1*u.day, P_max=10*u.year)
            l_pars = default_linear_prior(sigma_K0=25*u.km/u.s,
                                          P0=1*u.year,
                                          sigma_v=10*u.km/u.s)
        pars = {**nl_pars, **l_pars}
        prior = JokerPrior(pars=pars)

        return prior, default_expected_units

    elif case == 4:
        # Try with more poly_trends
        units = default_expected_units.copy()
        with pm.Model() as model:
            nl_pars = default_nonlinear_prior(P_min=1*u.day, P_max=10*u.year)
            l_pars = default_linear_prior(sigma_K0=25*u.km/u.s,
                                          P0=1*u.year,
                                          sigma_v=[10*u.km/u.s,
                                                   0.1*u.km/u.s/u.year],
                                          poly_trend=2)
        pars = {**nl_pars, **l_pars}
        prior = JokerPrior(pars=pars)
        units['v1'] = u.km/u.s/u.year

        return prior, units

    elif case == 5:
        # Default prior with .default()
        prior = JokerPrior.default(P_min=1*u.day, P_max=10*u.year,
                                   sigma_K0=25*u.km/u.s,
                                   P0=1*u.year,
                                   sigma_v=10*u.km/u.s)
        return prior, default_expected_units

    elif case == 6:
        # poly_trend with .default()
        units = default_expected_units.copy()
        prior = JokerPrior.default(P_min=1*u.day, P_max=10*u.year,
                                   sigma_K0=25*u.km/u.s,
                                   P0=1*u.year,
                                   sigma_v=[10*u.km/u.s,
                                            0.1*u.km/u.s/u.year],
                                   poly_trend=2)
        units['v1'] = u.km/u.s/u.year
        return prior, units

    return 7


@pytest.mark.parametrize('case', range(init_helper()))
def test_init_sample(case):
    prior, expected_units = init_helper(case)

    samples = prior.sample()
    for k in samples.par_names:
        assert hasattr(samples[k], 'unit')
        assert samples[k].unit == expected_units[k]

    samples = prior.sample(size=10)
    for k in samples.par_names:
        assert hasattr(samples[k], 'unit')
        assert samples[k].unit == expected_units[k]

    samples, logprior = prior.sample(size=10, return_logprobs=True)
    for k in samples.par_names:
        assert hasattr(samples[k], 'unit')
        assert samples[k].unit == expected_units[k]
    assert len(logprior) == len(samples)

    samples, logprior = prior.sample(size=10, generate_linear=True,
                                     return_logprobs=True)
    for k in samples.par_names:
        assert hasattr(samples[k], 'unit')
        assert samples[k].unit == expected_units[k]
    assert len(logprior) == len(samples)
