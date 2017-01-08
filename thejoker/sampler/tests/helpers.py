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
        self.v0 = np.random.uniform(-100, 100) * u.km/u.s

        orbit = SimulatedRVOrbit(**truth)
        rv = orbit.generate_rv_curve(mjd) + self.v0
        err = np.full_like(rv.value, 0.01) * u.km/u.s
        self.data['binary'] = RVData(mjd, rv, stddev=err)
        self.joker_params['binary'] = JokerParams(P_min=8*u.day, P_max=1024*u.day)
        self.truths['binary'] = truth.copy()

        # hierarchical triple - long term velocity trend
        self.v1 = np.random.uniform(-1, 1) * u.km/u.s/u.day
        orbit = SimulatedRVOrbit(**truth)
        rv = orbit.generate_rv_curve(mjd) + self.v0 + self.v1*(mjd-mjd.min())*u.day
        err = np.full_like(rv.value, 0.01) * u.km/u.s
        self.data['triple'] = RVData(mjd, rv, stddev=err, t_offset=mjd.min())
        self.joker_params['triple'] = JokerParams(P_min=8*u.day, P_max=1024*u.day,
                                                  trends=PolynomialVelocityTrend(n_terms=2))
        self.truths['triple'] = truth.copy()
