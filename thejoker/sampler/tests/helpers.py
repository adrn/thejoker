# Standard library
from collections import OrderedDict

# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np
from twobody import KeplerOrbit

# Package
from ...data import RVData
from ..params import JokerParams


class FakeData(object):

    def __init__(self, seed=42):
        np.random.seed(seed)

        EPOCH = np.random.uniform(0., 40)

        self.data = OrderedDict()
        self.joker_params = OrderedDict()
        self.truths = OrderedDict()

        P = np.random.uniform(40, 80) * u.day

        mjd = np.random.uniform(0, 300., 8)
        _genmjd = mjd + (EPOCH % P.value)

        # First just a binary
        truth = dict()
        truth['P'] = P
        truth['K'] = np.random.uniform(5, 15) * u.km/u.s
        truth['M0'] = np.random.uniform(0., 2*np.pi) * u.radian
        truth['omega'] = np.random.uniform(0., 2*np.pi) * u.radian
        truth['e'] = np.random.uniform()
        self.v0 = np.random.uniform(-100, 100) * u.km/u.s

        orbit = KeplerOrbit(P=truth['P'], e=truth['e'], omega=truth['omega'],
                            i=90*u.deg, Omega=0*u.deg, M0=truth['M0'],
                            t0=EPOCH)

        t = Time(mjd, format='mjd', scale='tcb')
        rv = truth['K'] * orbit.unscaled_radial_velocity(t) + self.v0
        err = np.full_like(rv.value, 0.01) * u.km/u.s
        data = RVData(mjd, rv, stddev=err)
        self.data['binary'] = data
        self.joker_params['binary'] = JokerParams(P_min=8*u.day, P_max=1024*u.day)
        self.truths['binary'] = truth.copy()

        # hierarchical triple - long term velocity trend
        # TODO: no longer works with TwoBody changes
        # self.v1 = np.random.uniform(-1, 1) * u.km/u.s/u.day
        # orbit = RVOrbit(**truth)
        # rv = orbit.generate_rv_curve(mjd) + self.v0 + self.v1*(mjd-mjd.min())*u.day
        # err = np.full_like(rv.value, 0.01) * u.km/u.s
        # data = RVData(mjd, rv, stddev=err, t_offset=mjd.min())
        # self.data['triple'] = data
        # self.joker_params['triple'] = JokerParams(P_min=8*u.day,
        #                                           P_max=1024*u.day)
        # self.truths['triple'] = truth.copy()
        # self.truths['triple']['M0'] = self.truths['triple']['M0'] - ((2*np.pi*data.t_offset/P.value))*u.radian

        # Binary on circular orbit
        truth = dict()
        truth['P'] = P
        truth['K'] = np.random.uniform(5, 15) * u.km/u.s
        truth['M0'] = np.random.uniform(0., 2*np.pi) * u.radian
        truth['omega'] = 0*u.radian
        truth['e'] = 0.

        orbit = KeplerOrbit(P=truth['P'], e=truth['e'], omega=truth['omega'],
                            i=90*u.deg, Omega=0*u.deg, M0=truth['M0'],
                            t0=EPOCH)
        _gent = Time(_genmjd, format='mjd', scale='tcb')
        rv = truth['K'] * orbit.unscaled_radial_velocity(_gent) + self.v0
        err = np.full_like(rv.value, 0.1) * u.km/u.s
        data = RVData(mjd+EPOCH, rv, stddev=err)
        self.data['circ_binary'] = data
        self.joker_params['circ_binary'] = JokerParams(P_min=8*u.day, P_max=1024*u.day)
        self.truths['circ_binary'] = truth.copy()
        self.truths['circ_binary']['M0'] = self.truths['circ_binary']['M0']
