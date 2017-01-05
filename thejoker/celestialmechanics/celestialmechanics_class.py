from __future__ import division, print_function

# Third-party
import astropy.time as at
import astropy.units as u
import numpy as np

# Project
from .celestialmechanics import rv_from_elements
from ..util import find_t0

__all__ = ['SimulatedRVOrbit', 'EPOCH']

EPOCH = 55555. # Magic Number: used for un-modding phi0

class SimulatedRVOrbit(object):
    """

    Parameters
    ----------
    pars : `thejoker.OrbitalParams`

    """
    def __init__(self, pars):
        self.pars = pars

    @property
    def _t0(self):
        return find_t0(self.pars.phi0.to(u.radian).value,
                       self.P.to(u.day).value, EPOCH)

    @property
    def t0(self):
        return at.Time(self._t0, scale='tcb', format='mjd')

    def _generate_rv_curve(self, t):
        """
        Parameters
        ----------
        t : array_like, `~astropy.time.Time`
            Array of times. Either in BJD or as an Astropy time.

        Returns
        -------
        rv : numpy.ndarray
        """

        if isinstance(t, at.Time):
            _t = t.tcb.mjd
        else:
            _t = t

        rv = rv_from_elements(times=_t,
                              P=self.pars.P.to(u.day).value,
                              K=self.pars.K.to(u.m/u.s).value,
                              e=self.pars.ecc.value,
                              omega=self.pars.omega.to(u.radian).value,
                              phi0=self.pars.phi0.to(u.radian).value)
        rv += self.pars.v0.to(u.m/u.s).value
        return rv

    def generate_rv_curve(self, t):
        """
        Parameters
        ----------
        t : array_like, `~astropy.time.Time`
            Array of times. Either in BJD or as an Astropy time.

        Returns
        -------
        rv : astropy.units.Quantity [km/s]
        """
        rv = self._generate_rv_curve(t)
        return (rv*u.m/u.s).to(u.km/u.s)

    def __call__(self, t):
        return self.generate_rv_curve(t)

    def plot(self, t=None, ax=None, **kwargs):
        """
        needs t or ax
        """
        if t is None and ax is None:
            raise ValueError("You must pass a time array (t) or axes "
                             "instance (ax)")

        if ax is None:
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots(1,1)

        if t is None:
            t = np.linspace(*ax.get_xlim(), num=1024)

        style = kwargs.copy()
        style.setdefault('linestyle', '-')
        style.setdefault('alpha', 0.5)
        style.setdefault('marker', None)
        style.setdefault('color', '#de2d26')

        rv = self.generate_rv_curve(t).to(u.km/u.s).value
        ax.plot(t, rv, **style)

        return ax
