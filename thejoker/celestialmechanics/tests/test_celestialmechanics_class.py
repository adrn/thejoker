from __future__ import division, print_function

# Third-party
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np

from ..celestialmechanics_class import SimulatedRVOrbit

def test_simulatedrvorbit():
    orbit = SimulatedRVOrbit(P=30.*u.day, a_sin_i=0.1*u.au,
                             ecc=0.1, omega=0.*u.radian, phi0=0.25524*u.radian)

    t = np.random.uniform(55612., 55792, 128)
    rv = orbit.generate_rv_curve(t)

    # TODO: write unit tests!

def test_parameter_convert():
    n_trials = 128

    # right now, round-trip tests
    Ps = np.exp(np.random.uniform(np.log(1.), np.log(1E4), n_trials)) * u.day
    Ks = np.random.uniform(0.1, 100, n_trials)*u.km/u.s
    eccs = np.random.random(n_trials)

    for P,K,ecc in zip(Ps, Ks, eccs):
        _mf, _asini = SimulatedRVOrbit.P_K_ecc_to_mf_asini_ecc(P, K, ecc)
        _P, _K = SimulatedRVOrbit.mf_asini_ecc_to_P_K(_mf, _asini, ecc)
        assert quantity_allclose(P, _P)
        assert quantity_allclose(K, _K)

    # right now, round-trip test
    mfs = np.exp(np.random.normal(np.log(2.5), 2., n_trials)) * u.Msun
    asinis = np.exp(np.random.uniform(np.log(1), np.log(1000), n_trials)) * u.au
    eccs = np.random.random(n_trials)

    for mf,asini,ecc in zip(mfs, asinis, eccs):
        _P, _K = SimulatedRVOrbit.mf_asini_ecc_to_P_K(mf, asini, ecc)
        _mf, _asini = SimulatedRVOrbit.P_K_ecc_to_mf_asini_ecc(_P, _K, ecc)

        assert quantity_allclose(mf, _mf)
        assert quantity_allclose(asini, _asini)

