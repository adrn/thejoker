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
from ...units import usys

def test_orbitalparams(tmpdir):

    op = OrbitalParams(P=15*u.day, asini=1E-3*u.au, ecc=0.212, omega=15*u.degree,
                       phi0=218*u.degree, v0=152.6163*u.km/u.s)
    for key in op._name_phystype.keys():
        assert getattr(op, key).ndim == 1

    op = OrbitalParams(P=[15]*u.day, asini=[1E-3]*u.au, ecc=[0.212], omega=[15]*u.degree,
                       phi0=[218]*u.degree, v0=[152.6163]*u.km/u.s)
    for key in op._name_phystype.keys():
        assert getattr(op, key).ndim == 1

    with pytest.raises(ValueError):
        v = np.random.random(size=(100,2))
        OrbitalParams(P=v*u.day, asini=v*u.au, ecc=v, omega=v*u.degree,
                      phi0=v*u.degree, v0=v*u.km/u.s)

    # copy
    op1 = OrbitalParams(P=[15]*u.day, asini=[1E-3]*u.au, ecc=[0.212], omega=[15]*u.degree,
                        phi0=[218]*u.degree, v0=[152.6163]*u.km/u.s)
    op2 = op1.copy()
    for key in op1._name_phystype.keys():
        v1 = getattr(op1, "_{}".format(key))
        v2 = getattr(op1, "_{}".format(key))
        assert np.allclose(v1, v2)
        assert v1.base is not v2 # make sure it truly is a copy and not just shared memory

    # from_hdf5
    n_test = 16
    with tempfile.NamedTemporaryFile(dir=str(tmpdir)) as fp:
        with h5py.File(fp.name, 'w') as f:
            for key,phystype in op1._name_phystype.items():
                f[key] = np.random.random(size=n_test)
                if phystype is not None:
                    f[key].attrs['unit'] = str(usys[phystype])

        op1 = OrbitalParams.from_hdf5(fp.name)
        with h5py.File(fp.name, 'r') as f:
            op2 = OrbitalParams.from_hdf5(f)

        assert np.allclose(op1._P, op2._P)

    # check pack()
    samples = op1.pack()
    assert samples.shape == (n_test,6)

    samples_plot = op1.pack(plot_units=True)
    assert samples_plot.shape == (n_test,6)
    assert not np.allclose(samples, samples_plot)

    # rv_orbit()
    orbit = op1.rv_orbit(0)

    # slicing
    op2 = op1[:3]
    for key,phystype in op2._name_phystype.items():
        assert len(getattr(op2, key)) == 3
