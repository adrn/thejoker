"""
NOTE: this is only used for testing the cython / c implementation.
"""

# Third-party
import astropy.units as u
import numpy as np
from twobody.wrap import cy_rv_from_elements
from scipy.stats import multivariate_normal

# Package
from ..stats import beta_logpdf, norm_logpdf

__all__ = ['ln_prior', 'get_ivar', 'design_matrix',
           'likelihood_worker', 'marginal_ln_likelihood']


def ln_prior(samples, joker_params):
    """
    Evaluate The Joker prior for the nonlinear parameters.
    """
    size = len(samples)
    ln_prior_val = np.zeros(size)
    a, b = (np.log(joker_params.P_min.to(u.day).value),
            np.log(joker_params.P_max.to(u.day).value))

    # P
    ln_prior_val += -np.log(b - a) - np.log(samples['P'].to(u.day).value)

    # M0
    ln_prior_val += -np.log(2 * np.pi)

    # e - MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
    ln_prior_val += beta_logpdf(np.array(samples['e']), 0.867, 3.03)

    # omega
    ln_prior_val += -np.log(2 * np.pi)

    # jitter
    if not joker_params._fixed_jitter:
        Jac = np.log(2 / samples['jitter'].value)  # Jacobian
        log_s2 = np.log(samples['jitter'].value ** 2)
        ln_prior_val += norm_logpdf(log_s2,
                                    joker_params.jitter[0],
                                    joker_params.jitter[1]) + Jac

    return ln_prior_val


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


def design_matrix(nonlinear_p, data, joker_params):
    """Compute the design matrix, M.

    Parameters
    ----------
    nonlinear_p : array_like
        Array of non-linear parameter values. For the default case,
        these are P (period, day), M0 (phase at pericenter, rad),
        ecc (eccentricity), omega (argument of perihelion, rad).
        May also contain log(jitter^2) as the last index.
    data : `~thejoker.data.RVData`
        The observations.
    joker_params : `~thejoker.sampler.params.JokerParams`
        The specification of parameters to infer with The Joker.

    Returns
    -------
    M : `numpy.ndarray`
        The design matrix with shape ``(n_times, n_params)``.

    """
    P, M0, ecc, omega = nonlinear_p[:4]  # we don't need the jitter here

    t = data._t_bmjd
    t0 = data._t0_bmjd
    zdot = cy_rv_from_elements(t, P, 1., ecc, omega, M0, t0,
                               joker_params.anomaly_tol,
                               joker_params.anomaly_maxiter)

    M1 = np.vander(t - t0, N=joker_params.poly_trend, increasing=True)
    M = np.hstack((zdot[:, None], M1))

    return M


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
