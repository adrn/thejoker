"""
NOTE: this is only used for testing the cython / c implementation.
"""

# Third-party
import astropy.units as u
import numpy as np
from twobody.wrap import cy_rv_from_elements
from astroML.utils import log_multivariate_gaussian
# from scipy.stats import multivariate_normal

# Project
from ...samples import JokerSamples
from ...distributions import FixedCompanionMass
from ...utils import DEFAULT_RNG

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


def likelihood_worker(y, ivar, M, mu, Lambda, make_aA=False):
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

    # Old implementation:
    # old_marg_ll = multivariate_normal.logpdf(y, b, B)

    # if make_aAinv:
    #     Ainv = Λinv + M.T @ Cinv @ M
    #     # Note: this is unstable! if cond num is high, could do:
    #     # p, *_ = np.linalg.lstsq(A, y)
    #     a = np.linalg.solve(Ainv, Λinv @ mu + M.T @ Cinv @ y)
    #     return marg_ll, b, B, a, Ainv

    # else:
    #     return marg_ll, b, B

    # New implementation:
    Ainv = Λinv + M.T @ Cinv @ M
    A = np.linalg.inv(Ainv)
    Binv = Cinv - Cinv @ M @ A @ M.T @ Cinv

    marg_ll = log_multivariate_gaussian(y, b, B, Vinv=Binv)

    if make_aA:
        # Note: this is unstable! if cond num is high, could do:
        # p, *_ = np.linalg.lstsq(A, y)
        a = np.linalg.solve(Ainv, Λinv @ mu + M.T @ Cinv @ y)
        return marg_ll, b, B, a, A

    else:
        return marg_ll, b, B


def design_matrix(nonlinear_p, data, prior):
    """Compute the design matrix, M.
    Parameters
    ----------
    nonlinear_p : array_like
    data : `~thejoker.RVData`
    prior : `~thejoker.JokerPrior`
    Returns
    -------
    M : `numpy.ndarray`
        The design matrix with shape ``(n_times, n_params)``.
    """
    P, ecc, omega, M0 = nonlinear_p[:4]  # we don't need the jitter here

    t = data._t_bmjd
    t0 = data._t_ref_bmjd
    zdot = cy_rv_from_elements(t, P, 1., ecc, omega, M0, t0, 1e-8, 128)

    M1 = np.vander(t - t0, N=prior.poly_trend, increasing=True)
    M = np.hstack((zdot[:, None], M1))

    return M


def get_M_Lambda_ivar(samples, prior, data):
    v_unit = data.rv.unit
    units = {'K': v_unit, 's': v_unit}
    for i, k in enumerate(list(prior._linear_equiv_units.keys())[1:]):  # skip K
        units[k] = v_unit / u.day**i
    packed_samples, _ = samples.pack(units=units)

    n_samples = packed_samples.shape[0]
    n_linear = len(prior._linear_equiv_units)

    Lambda = np.zeros(n_linear)
    for i, k in enumerate(prior._linear_equiv_units.keys()):
        if k == 'K':
            continue  # set below
        Lambda[i] = prior.pars[k].distribution.sd.eval() ** 2

    K_dist = prior.pars['K'].distribution
    if isinstance(K_dist, FixedCompanionMass):
        sigma_K0 = K_dist._sigma_K0.to_value(v_unit)
        P0 = K_dist._P0.to_value(samples['P'].unit)
        max_K2 = K_dist._max_K.to_value(v_unit) ** 2
    else:
        Lambda[0] = K_dist.sd.eval() ** 2

    for n in range(n_samples):
        M = design_matrix(packed_samples[n], data, prior)
        if isinstance(K_dist, FixedCompanionMass):
            P = samples['P'][n].value
            e = samples['e'][n]
            Lambda[0] = sigma_K0**2 / (1 - e**2) * (P / P0)**(-2/3)
            Lambda[0] = min(max_K2, Lambda[0])

        # jitter must be in same units as the data RV's / ivar!
        s = packed_samples[n, 4]
        ivar = get_ivar(data, s)

        yield n, M, Lambda, ivar, packed_samples[n], units


def marginal_ln_likelihood(samples, prior, data):
    """
    Internal function used to compute the likelihood marginalized
    over the linear parameters.

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`
    prior : `~thejoker.JokerPrior`
    data : `~thejoker.RVData`

    Returns
    -------
    marg_ln_like : `numpy.ndarray`
        Marginal log-likelihood values.

    """
    n_samples = len(samples)
    n_linear = len(prior._linear_equiv_units)
    mu = np.zeros(n_linear)

    marg_ll = np.zeros(n_samples)
    for n, M, Lambda, ivar, *_ in get_M_Lambda_ivar(samples, prior, data):
        try:
            marg_ll[n], *_ = likelihood_worker(data.rv.value, ivar, M,
                                               mu, np.diag(Lambda),
                                               make_aA=False)
        except np.linalg.LinAlgError as e:
            raise e

    return marg_ll


def rejection_sample(samples, prior, data, rnd=None):
    """

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`
    prior : `~thejoker.JokerPrior`
    data : `~thejoker.RVData`
    """

    n_linear = len(prior._linear_equiv_units)
    mu = np.zeros(n_linear)

    if rnd is None:
        rnd = DEFAULT_RNG()

    ll = marginal_ln_likelihood(samples, prior, data)
    uu = rnd.uniform(size=len(ll))
    mask = np.exp(ll - ll.max()) > uu

    n_good_samples = mask.sum()
    good_samples = samples[mask]

    all_packed = np.zeros((n_good_samples, len(prior.par_names)))
    for n, M, Lambda, ivar, packed_nonlinear, units in get_M_Lambda_ivar(
            good_samples, prior, data):
        try:
            _, b, B, a, A = likelihood_worker(data.rv.value, ivar, M,
                                              mu, np.diag(Lambda),
                                              make_aA=True)
        except np.linalg.LinAlgError as e:
            raise e

        linear_pars = rnd.multivariate_normal(a, A)
        all_packed[n] = np.concatenate((packed_nonlinear, linear_pars))

    unpack_units = dict()
    for k in prior.par_names:
        if k in units:
            unpack_units[k] = units[k]
        else:
            unpack_units[k] = samples[k].unit

    return JokerSamples.unpack(all_packed, unpack_units, prior.poly_trend,
                               data.t_ref)


def get_aAbB(samples, prior, data):
    """
    For testing Cython against

    Parameters
    ----------
    samples : `~thejoker.JokerSamples`
    prior : `~thejoker.JokerPrior`
    data : `~thejoker.RVData`
    """

    n_samples = len(samples)
    n_linear = len(prior._linear_equiv_units)
    n_times = len(data)
    mu = np.zeros(n_linear)

    out = {'a': np.zeros((n_samples, n_linear)),
           'A': np.zeros((n_samples, n_linear, n_linear)),
           'b': np.zeros((n_samples, n_times)),
           'B': np.zeros((n_samples, n_times, n_times))}

    for n, M, Lambda, ivar, *_ in get_M_Lambda_ivar(samples, prior, data):
        try:
            _, b, B, a, A = likelihood_worker(data.rv.value, ivar, M,
                                              mu, np.diag(Lambda),
                                              make_aA=True)
        except np.linalg.LinAlgError as e:
            raise e

        out['a'][n] = a
        out['A'][n] = A
        out['b'][n] = b
        out['B'][n] = B

    return out
