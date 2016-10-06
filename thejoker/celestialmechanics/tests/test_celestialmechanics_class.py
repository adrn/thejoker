from __future__ import division, print_function

# Third-party
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np

from ..orbitalparams import OrbitalParams
from ..celestialmechanics_class import SimulatedRVOrbit

def test_simulatedrvorbit():
    pars = OrbitalParams(P=30.*u.day, K=100.*u.m/u.s, ecc=0.11239, v0=0*u.km/u.s,
                         omega=0.*u.radian, phi0=0.25524*u.radian)
    orbit = SimulatedRVOrbit(pars)

    t = np.random.uniform(55612., 55792, 128)
    rv = orbit.generate_rv_curve(t)
    rv = orbit(t)

    # TODO: write unit tests!
