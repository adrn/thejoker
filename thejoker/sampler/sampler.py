# Third-party
from astropy import log as logger
import numpy as np
from scipy.interpolate import interp1d

from .. import EPOCH
from ..util import find_t0
from ..celestialmechanics import rv_from_elements

__all__ = ['design_matrix', 'tensor_vector_scalar', 'marginal_ln_likelihood',
           'period_grid']

def design_matrix(nonlinear_p, t):
    """

    Parameters
    ----------
    nonlinear_p : array_like
        Array of non-linear parameter values: P (period),
        phi0 (phase at pericenter), ecc (eccentricity),
        omega (argument of perihelion).
    t : array_like [day]
        Array of times in days.

    Returns
    -------
    A : `numpy.ndarray`

    """
    t = np.atleast_1d(t)
    P, phi0, ecc, omega = nonlinear_p

    a = np.ones_like(t)
    x1 = rv_from_elements(t, P, 1., ecc, omega, phi0, 0.)
    A = np.vstack((a, x1)).T
    return A

def tensor_vector_scalar(nonlinear_p, data):
    """

    Parameters
    ----------
    nonlinear_p : array_like
        Array of non-linear parameter values: P (period),
        phi0 (phase at pericenter), ecc (eccentricity),
        omega (argument of perihelion).
    data : `ebak.singleline.RVData`
        Instance of `RVData` containing the data to fit.

    Returns
    -------
    ATA : `numpy.ndarray`
        Value of A^T C^-1 A -- inverse of the covariance matrix
        of the linear parameters.
    p : `numpy.ndarray`
        Optimal values of linear parameters.
    chi2 : float
        Chi-squared value.

    """
    A = design_matrix(nonlinear_p, data._t)
    ATCinv = (A.T * data._ivar[None])
    ATA = ATCinv.dot(A)

    # Note: this is unstable! if cond num is high, could do:
    # p,*_ = np.linalg.lstsq(A, y)
    p = np.linalg.solve(ATA, ATCinv.dot(data._rv))

    dy = A.dot(p) - data._rv
    chi2 = np.sum(dy**2 * data._ivar)

    return ATA, p, chi2

def marginal_ln_likelihood(ATA, chi2):
    """

    Parameters
    ----------
    ATA : array_like
        Should have shape `(N, M, M)` or `(M, M)` where `M`
        is the number of linear parameters in the model.
    chi2 : numeric, array_like
        Chi-squared value(s).

    Returns
    -------
    marg_ln_like : `numpy.ndarray`
        Marginal log-likelihood values.

    """
    sign,logdet = np.linalg.slogdet(ATA)
    if not np.all(sign == 1.):
        logger.warning('logdet sign < 0')
    return -0.5*np.atleast_1d(chi2) + 0.5*logdet

def period_grid(data, P_min=1, P_max=1E4, resolution=2):
    """

    Parameters
    ----------
    data : `ebak.singleline.RVData`
        Instance of `RVData` containing the data to fit.
    P_min : numeric (optional)
        Minimum period value for the grid.
    P_max : numeric (optional)
        Maximum period value for the grid.
    resolution : numeric (optional)
        Extra factor used in computing the grid spacing.

    Returns
    -------
    P_grid : `numpy.ndarray` [day]
        Grid of periods in days.
    dP_grid : `numpy.ndarray` [day]
        Grid of period spacings in days.
    """
    T_max = data._t.max() - data._t.min()

    def _grid_element(P):
        return P**2 / (2*np.pi*T_max) / resolution

    P_grid = [P_min]
    while np.max(P_grid) < P_max:
        dP = _grid_element(P_grid[-1])
        P_grid.append(P_grid[-1] + dP)

    P_grid = np.array(P_grid)
    dP_grid = _grid_element(P_grid)

    return P_grid, dP_grid
