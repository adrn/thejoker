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
        rnd = np.random.RandomState(seed=seed)

        EPOCH = Time('J2000') + rnd.uniform(0., 40) * u.day

        self.datasets = OrderedDict()
        self.params = OrderedDict()
        self.truths = OrderedDict()

        P = rnd.uniform(40, 80) * u.day

        t = Time('J2000') + rnd.uniform(0, 300., 8) * u.day

        ######################################################################
        # Binary with random parameters

        truth = dict()
        truth['P'] = P
        truth['M0'] = rnd.uniform(0., 2*np.pi) * u.radian
        truth['omega'] = rnd.uniform(0., 2*np.pi) * u.radian
        truth['e'] = rnd.uniform()
        truth['K'] = rnd.uniform(5, 15) * u.km/u.s
        truth['v0'] = rnd.uniform(-100, 100) * u.km/u.s

        orbit = KeplerOrbit(P=truth['P'], e=truth['e'], omega=truth['omega'],
                            M0=truth['M0'], t0=EPOCH,
                            i=90*u.deg, Omega=0*u.deg) # these don't matter

        rv = truth['K'] * orbit.unscaled_radial_velocity(t) + truth['v0']
        err = np.full_like(rv.value, 0.01) * u.km/u.s
        data = RVData(t, rv, stddev=err, t0=EPOCH)
        self.datasets['binary'] = data
        self.params['binary'] = JokerParams(P_min=8*u.day, P_max=1024*u.day)
        self.truths['binary'] = truth.copy()

        ######################################################################
        # Binary on circular orbit

        truth = dict()
        truth['P'] = P
        truth['K'] = rnd.uniform(5, 15) * u.km/u.s
        truth['M0'] = rnd.uniform(0., 2*np.pi) * u.radian
        truth['omega'] = 0*u.radian
        truth['e'] = 0.
        truth['v0'] = rnd.uniform(-100, 100) * u.km/u.s

        orbit = KeplerOrbit(P=truth['P'], e=truth['e'], omega=truth['omega'],
                            M0=truth['M0'], t0=EPOCH,
                            i=90*u.deg, Omega=0*u.deg)

        rv = truth['K'] * orbit.unscaled_radial_velocity(t) + truth['v0']
        err = np.full_like(rv.value, 0.1) * u.km/u.s
        data = RVData(t, rv, stddev=err, t0=EPOCH)
        self.datasets['circ_binary'] = data
        self.params['circ_binary'] = JokerParams(P_min=8*u.day,
                                                 P_max=1024*u.day)
        self.truths['circ_binary'] = truth.copy()

        ######################################################################
        # hierarchical triple - long term velocity trend
        # TODO: no longer works with TwoBody changes
        # self.v1 = rnd.uniform(-1, 1) * u.km/u.s/u.day
        # orbit = RVOrbit(**truth)
        # rv = orbit.generate_rv_curve(mjd) + self.v0 + self.v1*(mjd-mjd.min())*u.day
        # err = np.full_like(rv.value, 0.01) * u.km/u.s
        # data = RVData(mjd, rv, stddev=err, t_offset=mjd.min())
        # self.data['triple'] = data
        # self.joker_params['triple'] = JokerParams(P_min=8*u.day,
        #                                           P_max=1024*u.day)
        # self.truths['triple'] = truth.copy()
        # self.truths['triple']['M0'] = self.truths['triple']['M0'] - ((2*np.pi*data.t_offset/P.value))*u.radian
