# Standard library
import sys

# Third-party
import astropy.units as u
from astropy.time import Time
from astropy.constants import G
import numpy as np

__all__ = ['get_t0', 'get_phi0', 'a1_sini', 'mf', 'mf_a1_sini_ecc_to_P_K']

@u.quantity_input(P=u.day)
def get_t0(phi0, P, ref_mjd, max_iter=16):
    r"""
    Compute the time at pericenter closest to the input reference
    time, ``ref_mjd``. This assumes that the epoch is at MJD=0, i.e.
    that the input phase was computed by doing:

    .. math::

        \phi_0 = \frac{2\pi \, t_0}{P} \bmod 2\pi

    This is carefully written to not subtract large numbers, but might
    look like magic.

    Parameters
    ----------
    phi0 : `~astropy.units.Quantity`, numeric [angle]
        The phase at pericenter, either as an Astropy Quantity with angle units
        or as a number, assumed to be in radians.
    P : `~astropy.units.Quantity` [time]
        The period.
    ref_mjd : numeric
        Reference MJD.

    Returns
    -------
    t0 : `~astropy.time.Time`

    """

    if hasattr(phi0, 'unit'):
        phi0 = phi0.to(u.radian).value

    P = P.to(u.day).value

    phi0 = np.arctan2(np.sin(phi0), np.cos(phi0)) # HACK
    epoch_phi = (2 * np.pi * ref_mjd / P) % (2. * np.pi)

    delta_phi = np.inf
    n_iter = 0
    guess = 0.
    while np.abs(delta_phi) > 8*sys.float_info.epsilon and n_iter < max_iter:
        delta_phi = (2*np.pi*guess/P) % (2*np.pi) - (phi0 - epoch_phi)
        delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi)) # HACK
        guess -= delta_phi / (2*np.pi) * P
        n_iter += 1

    return Time(ref_mjd + guess, format='mjd', scale='tcb')

@u.quantity_input(P=u.day)
def get_phi0(t0, P, ref_mjd=0.):
    r"""
    Compute the phase at pericenter relative to the reference MJD
    given a time of pericenter. The phase is computed with the
    time in Barycentric MJD.

    .. math::

        \phi_0 = \frac{2\pi \, (t_0-t_{\rm ref})}{P} \bmod 2\pi

    This is carefully written to not subtract large numbers, but might
    look like magic.

    Parameters
    ----------
    t0 : `~astropy.time.Time`, numeric
        The time at pericenter, either as a number, assumed to be in
        Barycentric MJD, or an Astropy time object.
    P : `~astropy.units.Quantity` [time]
        The period.
    ref_mjd :

    Returns
    -------
    phi0 : `~astropy.units.Quantity`

    """
    if isinstance(t0, Time):
        t0 = t0.tcb.mjd

    t0 = t0 - ref_mjd
    return ((2*np.pi*t0 / P.to(u.day).value) % (2*np.pi)) * u.radian

def a1_sini(P, K, ecc):
    """
    Compute the projected semi-major axis of the primary given the
    period, velocity semi-amplitude, and eccentricity.

    Parameters
    ----------
    P : `~astropy.units.Quantity` [time]
        Period.
    K : `~astropy.units.Quantity` [speed]
        Velocity semi-amplitude - this is really :math:`K_1`.
    ecc : numeric
        Eccentricity.
    """
    return K/(2*np.pi) * (P * np.sqrt(1-ecc**2))

def mf(P, K, ecc):
    """
    Compute the mass function given the period, velocity semi-amplitude,
    and eccentricity.

    Parameters
    ----------
    P : `~astropy.units.Quantity` [time]
        Period.
    K : `~astropy.units.Quantity` [speed]
        Velocity semi-amplitude - this is really :math:`K_1`.
    ecc : numeric
        Eccentricity.
    """
    mf = P * K**3 / (2*np.pi*G) * (1 - ecc**2)**(3/2.)
    return mf

def mf_a1_sini_ecc_to_P_K(mf, a1_sini, ecc):
    """
    Convert from mass function and projected semi-major axis of the primary
    to period and semi-amplitude of the primary :math:`K_1` (here, ``K``).

    Parameters
    ----------
    mf : `~astropy.units.Quantity` [mass]
        Mass function.
    a1_sini : `~astropy.units.Quantity` [length]
        Projected semi-major axis of the primary.
    ecc : numeric
        Eccentricity.
    """
    P = 2*np.pi * a1_sini**(3./2) / np.sqrt(G * mf)
    K = 2*np.pi * a1_sini / (P * np.sqrt(1-ecc**2))
    return P, K
