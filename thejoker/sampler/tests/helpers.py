# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np

# Package
from ...celestialmechanics import SimulatedRVOrbit
from ...data import RVData
from ..params import JokerParams, PolynomialVelocityTrend

class FakeData(object):

    def __init__(self, seed=42):
        np.random.seed(seed)

        EPOCH = np.random.uniform(55000., 57000)

        self.data = OrderedDict()
        self.joker_params = OrderedDict()
        self.truths = OrderedDict()

        mjd = np.linspace(0, 300., 128) + EPOCH

        # First just a binary
        truth = dict()
        truth['P'] = np.random.uniform(40, 80) * u.day
        truth['K'] = np.random.uniform(5, 15) * u.km/u.s
        truth['phi0'] = np.random.uniform(0., 2*np.pi) * u.radian
        truth['omega'] = np.random.uniform(0., 2*np.pi) * u.radian
        truth['ecc'] = np.random.uniform()
        v0 = np.random.uniform(-100, 100) * u.km/u.s
        truth['trends'] = PolynomialVelocityTrend(coeffs=[v0])

        orbit = SimulatedRVOrbit(**truth)
        rv = orbit.generate_rv_curve(mjd)
        err = np.full_like(rv.value, 0.01) * u.km/u.s
        noise = np.random.normal(0, err.value) * u.km/u.s
        rv += noise
        self.data['binary'] = RVData(mjd, rv, stddev=err, t_offset=0.)
        self.joker_params['binary'] = JokerParams(P_min=8*u.day, P_max=1024*u.day)
        self.truths['binary'] = truth.copy()

        # hierarchical triple - long term velocity trend
        v1 = np.random.uniform(-1, 1) * u.km/u.s/u.day
        truth['trends'] = PolynomialVelocityTrend(coeffs=[v0, v1])
        orbit = SimulatedRVOrbit(**truth)
        rv = orbit.generate_rv_curve(mjd)
        err = np.full_like(rv.value, 0.01) * u.km/u.s
        noise = np.random.normal(0, err.value) * u.km/u.s
        rv += noise
        self.data['triple'] = RVData(mjd, rv, stddev=err, t_offset=0.)
        self.joker_params['triple'] = JokerParams(P_min=8*u.day, P_max=1024*u.day,
                                                  trends=PolynomialVelocityTrend(n_terms=2))
        self.truths['triple'] = truth.copy()
