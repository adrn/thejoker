from __future__ import division, print_function

# Third-party
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np
import pytest

# Project
from ..orbitalparams import OrbitalParams

def test_init():

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
