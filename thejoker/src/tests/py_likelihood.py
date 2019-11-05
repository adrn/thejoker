"""
NOTE: this is only used for testing the cython / c implementation.
"""

# Third-party
import astropy.units as u
import numpy as np
from scipy.stats import multivariate_normal

__all__ = ['get_ivar', 'likelihood_worker', 'marginal_ln_likelihood']


def get_ivar(data, s):
    """Return a copy of the inverse variance array with jitter included.

    This is safe for zero'd out inverse variances.

    Parameters
    ----------
    data : `~thejoker.data.RVData`
    s : numeric
        Jitter in the same units as the RV data.

    """
    return data.ivar.value / (1 + s**2 * data.ivar.value)


def likelihood_worker(y, ivar, M, mu, Lambda, make_aAinv=False):
    """
    Internal function used to construct linear algebra objects
    used to compute the marginal log-likelihood.

    Parameters
    ----------
    M : array_like
        Design matrix.
    ivar : array_like
        Inverse-variance matrix.
    y : array_like
        Data (in this case, radial velocities).
    mu : array_like
        Prior mean for linear parameters.
    Lambda : array_like
        Prior variance matrix for linear parameters.

    Returns
    -------
    B : `numpy.ndarray`
        C + M Λ M^T - Variance of data gaussian.
    b : `numpy.ndarray`
        M µ - Optimal values of linear parameters.
    chi2 : float
        Chi-squared value.

    Notes
    -----
    The linear parameter vector returned here (``b``) may have a negative
    velocity semi-amplitude. I don't think there is anything we can do
    about this if we want to preserve the simple linear algebra below and
    it means that optimizing over the marginal likelihood below won't
    work -- in the argument of periastron, there will be two peaks of
    similar height, and so the highest marginal likelihood period might be
    shifted from the truth.

    """
    Λ = Lambda
    Λinv = np.linalg.inv(Λ)
    µ = mu

    Cinv = np.diag(ivar)
    C = np.diag(1 / ivar)

    b = M @ µ
    B = C + M @ Λ @ M.T
    marg_ll = multivariate_normal.logpdf(y, b, B)

    if make_aAinv:
        Ainv = Λinv + M.T @ Cinv @ M
        # Note: this is unstable! if cond num is high, could do:
        # p, *_ = np.linalg.lstsq(A, y)
        a = np.linalg.solve(Ainv, Λinv @ mu + M.T @ Cinv @ y)
        return marg_ll, b, B, a, Ainv

    else:
        return marg_ll, b, B


def marginal_ln_likelihood(nonlinear_p, data, joker_params, make_aAinv=False):
    """
    Internal function used to compute the likelihood marginalized
    over the linear parameters.

    Parameters
    ----------
    nonlinear_p : array_like
        Array of non-linear parameter values. For the default case,
        these are P (period, day), M0 (phase at pericenter, rad),
        ecc (eccentricity), omega (argument of perihelion, rad).
        May also contain jitter as the last index.
    data : `~thejoker.data.RVData`
        The observations.
    joker_params : `~thejoker.sampler.params.JokerParams`
        The specification of parameters to infer with The Joker.
    tvsi : iterable (optional)
        Optionally pass in the tensor, vector, scalar, ivar values so they
        aren't re-computed.

    Returns
    -------
    marg_ln_like : `numpy.ndarray`
        Marginal log-likelihood values.

    """

    M = design_matrix(nonlinear_p, data, joker_params)

    if joker_params.scale_K_prior_with_P:
        # TODO: allow customizing K0!
        K0 = (25 * u.km/u.s).to_value(joker_params._jitter_unit)
        Lambda = np.zeros((joker_params._n_linear, joker_params._n_linear))
        P = nonlinear_p[0]
        e = nonlinear_p[2]
        Lambda[0, 0] = K0**2 / (1 - e**2) * (P / 365.)**(-2/3)
        Lambda[1:, 1:] = joker_params.linear_par_Lambda
    else:
        Lambda = joker_params.linear_par_Lambda

    # jitter must be in same units as the data RV's / ivar!
    s = nonlinear_p[4]
    ivar = get_ivar(data, s)
    marg_ll, *_ = likelihood_worker(data.rv.value, ivar, M,
                                    joker_params.linear_par_mu,
                                    Lambda,
                                    make_aAinv=make_aAinv)

    return marg_ll
