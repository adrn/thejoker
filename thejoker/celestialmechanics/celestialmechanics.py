"""
Celestial mechanics for the Ebak project.

General comments
----------------
- Parameterization comes from Winn http://arxiv.org/abs/1001.2010
- Mean, eccentric, and true anomaly formulae from Wikipedia
  https://en.wikipedia.org/wiki/Eccentric_anomaly

Issues
------
- Should I permit inputs of sin and cos instead of just angles?

"""
from __future__ import division

__author__ = "David W. Hogg <david.hogg@nyu.edu>"

# Standard-library
import warnings

# Third-party
import numpy as np

__all__ = ['mean_anomaly_from_eccentric_anomaly',
           'eccentric_anomaly_from_mean_anomaly',
           'true_anomaly_from_eccentric_anomaly',
           'd_eccentric_anomaly_d_mean_anomaly',
           'd_true_anomaly_d_eccentric_anomaly',
           'Z_from_elements', 'rv_from_elements']

def mean_anomaly_from_eccentric_anomaly(Es, e):
    """
    Parameters
    ----------
    Es : numeric, array_like [radian]
        Eccentric anomaly.
    e : numeric, array_like
        Eccentricity.

    Returns
    -------
    Ms : numeric, array_like [radian]
        Mean anomaly.
    """
    return Es - e * np.sin(Es)

def eccentric_anomaly_from_mean_anomaly(Ms, e, tol=1E-13, maxiter=128):
    """
    Parameters
    ----------
    Ms : numeric, array_like [radian]
        Mean anomaly.
    e : numeric
        Eccentricity.
    tol : numeric (optional)
        Numerical tolerance used in iteratively solving for eccentric anomaly.
    maxiter : int (optional)
        Maximum number of iterations when iteratively solving for
        eccentric anomaly.

    Returns
    -------
    Es : numeric [radian]
        Eccentric anomaly.

    Issues
    ------
    - Magic numbers ``tol`` and ``maxiter``
    """
    Ms = np.atleast_1d(Ms)

    if Ms.ndim > 1:
        raise ValueError("Input must have <= 1 dim.")

    Es = Ms + e * np.sin(Ms)

    # how APW thinks this should be done:
    # for i in range(Ms.shape[0]):
    #     for _ in range(maxiter):
    #         deltaMs = Ms[i] - mean_anomaly_from_eccentric_anomaly(Es[i], e)
    #         Es[i] = Es[i] + deltaMs / (1. - e * np.cos(Es[i]))

    #         if np.all(np.abs(deltaMs) < tol):
    #             break

    #     else:
    #         warnings.warn("eccentric_anomaly_from_mean_anomaly() reached maximum "
    #                       "number of iterations ({})".format(maxiter), RuntimeWarning)

    # how Hogg wrote this originally:
    for _ in range(maxiter):
        deltaMs = (Ms - mean_anomaly_from_eccentric_anomaly(Es, e))
        Es = Es + deltaMs / (1. - e * np.cos(Es))

        if np.all(np.abs(deltaMs) < tol):
            break

    else:
        warnings.warn("eccentric_anomaly_from_mean_anomaly() reached maximum "
                      "number of iterations ({})".format(maxiter), RuntimeWarning)

    return Es

def true_anomaly_from_eccentric_anomaly(Es, e):
    """
    Parameters
    ----------
    Es : numeric, array_like [radian]
        Eccentric anomaly.
    e : numeric, array_like
        Eccentricity.

    Returns
    -------
    fs : numeric [radian]
        True anomaly.
    """
    cEs, sEs = np.cos(Es), np.sin(Es)
    fs = np.arccos((cEs - e) / (1.0 - e * cEs))
    return fs * np.sign(np.sin(fs)) * np.sign(sEs)

def d_eccentric_anomaly_d_mean_anomaly(Es, e):
    """
    Parameters
    ----------
    Es : numeric, array_like [radian]
        Eccentric anomaly.
    e : numeric, array_like
        Eccentricity.

    Returns
    -------
    dE_dM : numeric
        Derivatives of one anomaly w.r.t. the other.
    """
    return 1. / (1. - e * np.cos(Es))

def d_true_anomaly_d_eccentric_anomaly(Es, fs, e):
    """
    Parameters
    ----------
    Es : numeric, array_like [radian]
        Eccentric anomaly.
    fs : numeric, array_like [radian]
        True anomaly.
    e : numeric, array_like
        Eccentricity.

    Returns
    -------
    df_dE : numeric
        Derivatives of one anomaly w.r.t. the other.

    Issues
    ------
    - Insane assert statement.
    """
    cfs, sfs = np.cos(fs), np.sin(fs)
    cEs, sEs = np.cos(Es), np.sin(Es)
    assert np.allclose(cEs, (e + cfs) / (1. + e * cfs))
    return (sEs / sfs) * (1. + e * cfs) / (1. - e * cEs)

def Z_from_elements(times, P, asini, e, omega, time0):
    """
    Z points towards the observer.

    Parameters
    ----------
    times : array_like [day]
        BJD of observations.
    p : numeric [day]
        Period.
    asini : numeric [AU]
        Semi-major axis times sine of inclination for star from system
        barycenter (will be negative for one of the stars?)
    e : numeric
        Eccentricity.
    omega : numeric [radian]
        Perihelion argument parameter from Winn.
    time0 : numeric [day]
        Time of "zeroth" pericenter.

    Returns
    -------
    Z : numeric [AU]
        Line-of-sight position.

    Issues
    ------
    - doesn't include system Z value (Z offset or Z zeropoint)
    - could be made more efficient (there are lots of re-dos of trig calls)

    """
    times = np.array(times)

    dMdt = 2. * np.pi / P
    Ms = (times - time0) * dMdt

    Es = eccentric_anomaly_from_mean_anomaly(Ms, e)
    fs = true_anomaly_from_eccentric_anomaly(Es, e)

    rs = asini * (1. - e * np.cos(Es))
    # this is equivalent to:
    # rs = asini * (1. - e**2) / (1 + e*np.cos(fs))

    return rs * np.sin(omega + fs)

def rv_from_elements(times, P, asini, e, omega, phi0, rv0):
    """
    Parameters
    ----------
    times : array_like [day]
        BJD of observations.
    p : numeric [day]
        Period.
    asini : numeric [AU]
        Semi-major axis times sine of inclination for star from system
        barycenter (will be negative for one of the stars?)
    e : numeric
        Eccentricity.
    omega : numeric [radian]
        Perihelion argument parameter from Winn.
    phi0 : numeric [radian]
        Phase at pericenter.
    rv0 : numeric [AU/day]
        Systemic velocity.

    Returns
    -------
    rv : numeric [AU/day]
        Radial velocity.

    Issues
    ------
    - could be made more efficient (there are lots of re-dos of trig calls)
    """
    times = np.array(times)
    phase = 2 * np.pi * times / P

    dMdt = 2. * np.pi / P
    # Ms = (times - time0) * dMdt
    Ms = phase - phi0

    Es = eccentric_anomaly_from_mean_anomaly(Ms, e)
    fs = true_anomaly_from_eccentric_anomaly(Es, e)

    dEdts = d_eccentric_anomaly_d_mean_anomaly(Es, e) * dMdt
    dfdts = d_true_anomaly_d_eccentric_anomaly(Es, fs, e) * dEdts

    rs = asini * (1. - e * np.cos(Es))
    drdts = asini * e * np.sin(Es) * dEdts

    rvs = rs * np.cos(omega + fs) * dfdts + np.sin(omega + fs) * drdts

    return rvs + rv0
