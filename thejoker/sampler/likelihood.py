# Third-party
import numpy as np
from twobody.celestial import rv_from_elements

# Package
from .utils import get_ivar
from ..log import log as logger

__all__ = ['design_matrix', 'tensor_vector_scalar', 'marginal_ln_likelihood']

def design_matrix(nonlinear_p, data, joker_params):
    """

    Parameters
    ----------
    nonlinear_p : array_like
        Array of non-linear parameter values. For the default case,
        these are P (period, day), phi0 (phase at pericenter, rad),
        ecc (eccentricity), omega (argument of perihelion, rad).
        May also contain log(jitter^2) as the last index.
    data : `~thejoker.data.RVData`
        The observations.
    joker_params : `~thejoker.sampler.params.JokerParams`
        The specification of parameters to infer with The Joker.

    Returns
    -------
    A : `numpy.ndarray`
        The design matrix with shape ``(n_times, n_params)``.

    """
    P, phi0, ecc, omega = nonlinear_p[:4] # we don't need the jitter here

    # phi0 now is implicitly relative to data.t_offset, not mjd=0
    t = data._t_bmjd
    zdot = rv_from_elements(times=t, P=P, K=1., e=ecc,
                            omega=omega, phi0=phi0,
                            anomaly_tol=joker_params.anomaly_tol)

    # TODO: right now, we only support a single, global velocity trend!
    A1 = np.vander(t, N=joker_params._n_trend, increasing=True)
    A = np.hstack((zdot[:,None], A1))

    return A

def tensor_vector_scalar(A, ivar, y):
    """
    Internal function used to construct linear algebra objects
    used to compute the marginal log-likelihood.

    Parameters
    ----------
    A : `~numpy.ndarray`
        Design matrix.
    ivar : `~numpy.ndarray`
        Inverse-variance matrix.
    y : `~numpy.ndarray`
        Data (in this case, radial velocities).

    Returns
    -------
    ATCinvA : `numpy.ndarray`
        Value of A^T C^-1 A -- inverse of the covariance matrix
        of the linear parameters.
    p : `numpy.ndarray`
        Optimal values of linear parameters.
    chi2 : float
        Chi-squared value.

    Notes
    -----
    The linear parameter vector returned here (``p``) may have a negative
    velocity semi-amplitude. I don't think there is anything we can do
    about this if we want to preserve the simple linear algebra below and
    it means that optimizing over the marginal likelihood below won't
    work -- in the argument of periastron, there will be two peaks of
    similar height, and so the highest marginal likelihood period might be
    shifted from the truth.

    """
    ATCinv = (A.T * ivar[None])
    ATCinvA = ATCinv.dot(A)

    # Note: this is unstable! if cond num is high, could do:
    # p,*_ = np.linalg.lstsq(A, y)
    p = np.linalg.solve(ATCinvA, ATCinv.dot(y))
    dy = A.dot(p) - y

    chi2 = np.sum(dy**2 * ivar) # don't need log term for the jitter b.c. in likelihood below

    return ATCinvA, p, chi2

def marginal_ln_likelihood(nonlinear_p, data, joker_params, tvsi=None):
    """
    Internal function used to compute the likelihood marginalized
    over the linear parameters.

    This returns Eq. 11 arxiv:1610.07602v2

    Parameters
    ----------
    nonlinear_p : array_like
        Array of non-linear parameter values. For the default case,
        these are P (period, day), phi0 (phase at pericenter, rad),
        ecc (eccentricity), omega (argument of perihelion, rad).
        May also contain jitter as the last index.
    data : `~thejoker.data.RVData`
        The observations.
    joker_params : `~thejoker.sampler.params.JokerParams`
        The specificationof parameters to infer with The Joker.
    tvsi : iterable (optional)
        Optionally pass in the tensor, vector, scalar, ivar values so they
        aren't re-computed.

    Returns
    -------
    marg_ln_like : `numpy.ndarray`
        Marginal log-likelihood values.

    """
    if tvsi is None:
        A = design_matrix(nonlinear_p, data, joker_params)

        # jitter must be in same units as the data RV's / ivar!
        s = nonlinear_p[4]
        ivar = get_ivar(data, s)
        ATCinvA, p, chi2 = tensor_vector_scalar(A, ivar, data.rv.value)

    else:
        ATCinvA, p, chi2, ivar = tvsi

    # This is -logdet(2πC_j)
    sign, logdet = np.linalg.slogdet(ATCinvA / (2*np.pi))
    if not np.all(sign == 1.):
        logger.debug('logdet sign < 0')
        return np.nan

    logdet += np.sum(np.log(ivar / (2*np.pi)))

    return 0.5*logdet - 0.5*np.atleast_1d(chi2)
