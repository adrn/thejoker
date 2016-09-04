""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import copy

# Third-party
from astropy.constants import G
import astropy.time as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from gala.units import UnitSystem

# Project
from .celestialmechanics import rv_from_elements
from .util import find_t0
from .units import usys

__all__ = ['RVOrbit', 'SimulatedRVOrbit']

_G = G.decompose(usys).value
EPOCH = 55555. # Magic Number: used for un-modding phi0

class RVOrbit(object):
    """

    Parameters
    ----------
    P : `~astropy.units.Quantity` [time]
        Orbital period.
    a_sin_i : `~astropy.units.Quantity` [length]
        Semi-major axis times sine of inclination angle.
    ecc : numeric
        Eccentricity.
    omega : `~astropy.units.Quantity` [angle]
        Argument of perihelion.
    phi0 : `~astropy.units.Quantity` [angle]
        Phase of pericenter.
    v0 : `~astropy.units.Quantity` [speed]
        Systemic velocity
    """
    @u.quantity_input(P=u.yr, a_sin_i=u.au, omega=u.radian,
                      phi0=u.radian, v0=u.km/u.s)
    def __init__(self, P, a_sin_i, ecc, omega, phi0, v0=0*u.km/u.s):
        # store unitful things without units for speed
        self._P = P.decompose(usys).value
        self._a_sin_i = a_sin_i.decompose(usys).value
        self._omega = omega.decompose(usys).value
        self._phi0 = phi0.decompose(usys).value
        self._v0 = v0.decompose(usys).value
        self.ecc = ecc

    # Computed Quantities
    @property
    def _K(self):
        return 2*np.pi / (self._P * np.sqrt(1-self.ecc**2)) * self._a_sin_i

    @property
    def _mf(self):
        return self._P * self._K**3 / (2*np.pi*_G) * (1 - self.ecc**2)**(3/2.)

    @property
    def _t0(self):
        return find_t0(self._phi0, self._P, EPOCH)

    # Unit-ful properties
    @property
    def P(self):
        return self._P * usys['time']

    @property
    def a_sin_i(self):
        return self._a_sin_i * usys['length']

    @property
    def omega(self):
        return self._omega * usys['angle']

    @property
    def t0(self):
        return at.Time(self._t0, scale='tcb', format='mjd')

    @property
    def v0(self):
        return self._v0 * usys['length'] / usys['time']

    @property
    def K(self):
        return self._K * usys['length'] / usys['time']

    @property
    def mf(self):
        return self._mf * usys['mass']

    def copy(self):
        return copy.copy(self)

    # convenience methods
    @staticmethod
    def mf_asini_ecc_to_P_K(mf, asini, ecc):
         P = 2*np.pi * asini**(3./2) / np.sqrt(G * mf)
         K = 2*np.pi * asini / (P * np.sqrt(1-ecc**2))
         return P, K

    @staticmethod
    def P_K_ecc_to_mf_asini_ecc(P, K, ecc):
        asini = K*P / (2*np.pi) * np.sqrt(1-ecc**2)
        mf = P * K**3 / (2*np.pi*G) * (1 - ecc**2)**(3/2.)
        return mf, asini

class SimulatedRVOrbit(RVOrbit):

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

        rv = rv_from_elements(_t, self._P, self._a_sin_i,
                              self.ecc, self._omega, self._phi0,
                              self._v0)
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
        return (rv*usys['length']/usys['time']).to(u.km/u.s)

    def plot(self, t=None, ax=None, **kwargs):
        """
        needs t or ax
        """
        if t is None and ax is None:
            raise ValueError("You must pass a time array (t) or axes "
                             "instance (ax)")

        if ax is None:
            fig,ax = plt.subplots(1,1)

        if t is None:
            t = np.linspace(*ax.get_xlim(), 1024)

        style = kwargs.copy()
        style.setdefault('linestyle', '-')
        style.setdefault('alpha', 0.5)
        style.setdefault('marker', None)
        style.setdefault('color', '#de2d26')

        rv = self.generate_rv_curve(t).to(u.km/u.s).value
        ax.plot(t, rv, **style)

        return ax
