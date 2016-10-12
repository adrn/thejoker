# Third-party
import astropy.units as u
from astropy import log as logger
import numpy as np

from ..config import defaults
from ..celestialmechanics import rv_from_elements

__all__ = ['design_matrix', 'tensor_vector_scalar', 'marginal_ln_likelihood',
           'sample_prior', 'period_grid']

def design_matrix(nonlinear_p, t, t_offset):
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
    P, phi0, ecc, omega, _ = nonlinear_p

    a = np.ones_like(t)
    x1 = rv_from_elements(times=t, P=P, K=1., e=ecc, omega=omega,
                          phi0=phi0-2*np.pi*((t_offset/P) % 1.), rv0=0.)
    A = np.vstack((a, x1)).T
    return A

def tensor_vector_scalar(A, ivar, y):
    """

    Parameters
    ----------
    nonlinear_p : array_like
        Array of non-linear parameter values: P (period),
        phi0 (phase at pericenter), ecc (eccentricity),
        omega (argument of perihelion), log_s2 (ln of the jitter squared).
    data : `thejoker.sampler.RVData`
        Instance of `RVData` containing the data to fit.

    Returns
    -------
    ATCinvA : `numpy.ndarray`
        Value of A^T C^-1 A -- inverse of the covariance matrix
        of the linear parameters.
    p : `numpy.ndarray`
        Optimal values of linear parameters.
    chi2 : float
        Chi-squared value.

    """
    ATCinv = (A.T * ivar[None])
    ATCinvA = ATCinv.dot(A)

    # Note: this is unstable! if cond num is high, could do:
    # p,*_ = np.linalg.lstsq(A, y)
    p = np.linalg.solve(ATCinvA, ATCinv.dot(y))

    dy = A.dot(p) - y
    chi2 = np.sum(dy**2 * ivar) # don't need log term for the jitter b.c. in likelihood below

    return ATCinvA, p, chi2

def marginal_ln_likelihood(nonlinear_p, data):
    """

    Parameters
    ----------
    ATCinvA : array_like
        Should have shape `(N, M, M)` or `(M, M)` where `M`
        is the number of linear parameters in the model.
    chi2 : numeric, array_like
        Chi-squared value(s).

    Returns
    -------
    marg_ln_like : `numpy.ndarray`
        Marginal log-likelihood values.

    """
    A = design_matrix(nonlinear_p, data._t, data.t_offset)

    s2 = nonlinear_p[4]
    ivar = data.get_ivar(s2)

    ATCinvA,_,chi2 = tensor_vector_scalar(A, ivar, data._rv)

    sign,logdet = np.linalg.slogdet(ATCinvA)
    if not np.all(sign == 1.):
        logger.debug('logdet sign < 0')
        return np.nan

    logdet += np.sum(np.log(ivar/(2*np.pi))) # TODO: this needs a final audit, and is inconsistent with math in the paper

    return 0.5*logdet - 0.5*np.atleast_1d(chi2)

u.quantity_input(P_min=u.day, P_max=u.day)
def sample_prior(n=1, P_min=defaults['P_min'], P_max=defaults['P_min'],
                 log_jitter2_mean=0., log_jitter2_std=1.):
    """
    Generate samples from the prior. Logarithmic in period, uniform in
    phase and argument of pericenter, Beta distribution in eccentricity.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    P_min : `astropy.units.Quantity`
        Minimum period.
    P_max : `astropy.units.Quantity`
        Maximum period.

    Returns
    -------
    prior_samples : dict
        Keys: `['P', 'phi0', 'ecc', 'omega']`, each as
        `astropy.units.Quantity` objects (i.e. with units).

    """

    # sample from priors in nonlinear parameters
    P = np.exp(np.random.uniform(np.log(P_min.to(u.day).value),
                                 np.log(P_max.to(u.day).value),
                                 size=n)) * u.day
    phi0 = np.random.uniform(0, 2*np.pi, size=n) * u.radian

    # MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
    ecc = np.random.beta(a=0.867, b=3.03, size=n)
    omega = np.random.uniform(0, 2*np.pi, size=n) * u.radian

    # DFM's idea: wide, Gaussian prior in log(s^2)
    s2 = np.exp(np.random.normal(log_jitter2_mean, log_jitter2_std, size=n)) * (u.m/u.s)**2

    return dict(P=P, phi0=phi0, ecc=ecc, omega=omega, jitter2=s2)

# ----------------------------------------------------------------------------

def period_grid(data, P_min=1, P_max=1E4, resolution=2):
    """
    DEPRECATED!

    .. note::

        This function is not used anymore.

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
