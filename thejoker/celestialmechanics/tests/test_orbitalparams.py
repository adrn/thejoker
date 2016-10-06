from __future__ import division, print_function

# Standard library
import tempfile

# Third-party
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import h5py
import numpy as np
import pytest

# Project
from ..orbitalparams import OrbitalParams

def test_orbitalparams(tmpdir):

    op = OrbitalParams(P=15*u.day, K=1*u.km/u.s, ecc=0.212, omega=15*u.degree,
                       phi0=218*u.degree, v0=152.6163*u.km/u.s)
    for key in op._name_to_unit.keys():
        assert getattr(op, key).ndim == 1

    op = OrbitalParams(P=[15]*u.day, K=[1.]*u.km/u.s, ecc=[0.212], omega=[15]*u.degree,
                       phi0=[218]*u.degree, v0=[152.6163]*u.km/u.s)
    for key in op._name_to_unit.keys():
        assert getattr(op, key).ndim == 1

    with pytest.raises(ValueError):
        v = np.random.random(size=(100,2))
        OrbitalParams(P=v*u.day, K=v*u.m/u.s, ecc=v, omega=v*u.degree,
                      phi0=v*u.degree, v0=v*u.km/u.s)

    # copy
    op1 = OrbitalParams(P=[15]*u.day, K=[1]*u.m/u.s, ecc=[0.212], omega=[15]*u.degree,
                        phi0=[218]*u.degree, v0=[152.6163]*u.km/u.s)
    op2 = op1.copy()
    for key in op1._name_to_unit.keys():
        v1 = getattr(op1, "_{}".format(key))
        v2 = getattr(op1, "_{}".format(key))
        assert np.allclose(v1, v2)
        assert v1.base is not v2 # make sure it truly is a copy and not just shared memory

    # from_hdf5
    n_test = 16
    with tempfile.NamedTemporaryFile(dir=str(tmpdir)) as fp:
        with h5py.File(fp.name, 'w') as f:
            for key,unit in op1._name_to_unit.items():
                f[key] = np.random.random(size=n_test)
                f[key].attrs['unit'] = str(unit)

        op1 = OrbitalParams.from_hdf5(fp.name)
        with h5py.File(fp.name, 'r') as f:
            op2 = OrbitalParams.from_hdf5(f)

        assert np.allclose(op1._P, op2._P)

    # check pack()
    samples = op1.pack()
    assert samples.shape == (n_test,7)

    # rv_orbit()
    orbit = op1.rv_orbit(0)

    # slicing
    op2 = op1[:3]
    for key in op2._name_to_unit.keys():
        assert len(getattr(op2, key)) == 3

    # # test_parameter_convert
    # n_trials = 128

    # # right now, round-trip tests
    # Ps = np.exp(np.random.uniform(np.log(1.), np.log(1E4), n_trials)) * u.day
    # Ks = np.random.uniform(0.1, 100, n_trials)*u.km/u.s
    # eccs = np.random.random(n_trials)

    # for P,K,ecc in zip(Ps, Ks, eccs):
    #     _mf, _asini = OrbitalParams.P_K_ecc_to_mf_asini_ecc(P, K, ecc)
    #     _P, _K = OrbitalParams.mf_asini_ecc_to_P_K(_mf, _asini, ecc)
    #     assert quantity_allclose(P, _P)
    #     assert quantity_allclose(K, _K)

    # # right now, round-trip test
    # mfs = np.exp(np.random.normal(np.log(2.5), 2., n_trials)) * u.Msun
    # asinis = np.exp(np.random.uniform(np.log(1), np.log(1000), n_trials)) * u.au
    # eccs = np.random.random(n_trials)

    # for mf,asini,ecc in zip(mfs, asinis, eccs):
    #     _P, _K = SimulatedRVOrbit.mf_asini_ecc_to_P_K(mf, asini, ecc)
    #     _mf, _asini = SimulatedRVOrbit.P_K_ecc_to_mf_asini_ecc(_P, _K, ecc)

    #     assert quantity_allclose(mf, _mf)
    #     assert quantity_allclose(asini, _asini)

