# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False
# cython: language_level=3

# Third-party
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython
cimport scipy.linalg.cython_lapack as lapack

# from libc.stdio cimport printf
from libc.math cimport pow, log, fabs, pi

cdef extern from "src/twobody.h":
    void c_rv_from_elements(double *t, double *rv, int N_t,
                            double P, double K, double e, double omega,
                            double phi0, double t0, double tol, int maxiter)

# Log of 2π
cdef:
    double LN_2PI = 1.8378770664093453
    double INF = float('inf')
    double anomaly_tol = 1E-10 # passed to c_rv_from_elements
    int anomaly_maxiter = 128 # passed to c_rv_from_elements

cdef void design_matrix(double P, double phi0, double ecc, double omega,
                        double[::1] t, double t0,
                        double[:, ::1] M_T,
                        int n_trend):
    """Construct the elements of the design matrix transpose, M_T.

    Parameters
    ----------
    P : double
        Period [day].
    phi0 : double
        Phase [radian].
    ecc : double
        Eccentricity
    omega : double
        Argument of pericenter [radian].
    t : `numpy.ndarray`
        Data time array.
    t0 : double
        Reference time.
    n_trend : int
        Number of terms in the long-term velocity trend.

    Outputs
    -------
    M_T : `numpy.ndarray`
        The transpose of the design matrix, to be filled by this function.
        Should have shape: (number of linear parameters, number of data points).

    """
    cdef:
        int i, j
        int n_times = t.shape[0]

    c_rv_from_elements(&t[0], &M_T[0,0], n_times,
                       P, 1., ecc, omega, phi0, t0,
                       anomaly_tol, anomaly_maxiter)

    if n_trend > 0: # only needed if we are fitting for a constant trend
        for j in range(n_times):
            M_T[1, j] = 1.

    if n_trend > 1: # only needed if more than constant trend
        for i in range(1, n_trend):
            for j in range(n_times):
                M_T[1+i, j] = pow(t[j] - t0, i)


cdef void get_ivar(double[::1] ivar, double s, double[::1] new_ivar):
    """Return new ivar values with the jitter incorporated.

    This is safe for zero'd out inverse variances.

    Parameters
    ----------
    ivar : `numpy.ndarray`
        Inverse-variance array.
    s : numeric
        Jitter in the same units as the RV data.
    new_ivar : `numpy.ndarray`
        The output array.

    """
    cdef:
        int n = ivar.shape[0]
        int i

    for i in range(n):
        new_ivar[i] = ivar[i] / (1 + s*s * ivar[i])


cdef double likelihood_worker(double[::1] y, double[::1] ivar,  # Data
                              double[:,::1] M_T,  # Design matrix transposed
                              double[::1] mu, double[:, ::1] Lambda,
                              double[:, ::1] Lambda_inv, # Prior
                              int make_aAinv,  # Controls whether to make A, a
                              # Output:
                              double[:, ::1] B, double[::1] b,
                              double[:, ::1] Ainv, double[::1] a,
                              # Helper arrays:
                              double[:, ::1] Btmp, double[:, ::1] Atmp,
                              int[::1] npar_ipiv, int[::1] ntime_ipiv,
                              double[::1] npar_work, double[::1] ntime_work):
    cdef:
        int i, j, n, m  # i,j used below for n_pars, n,m used for n_times
        int n_times = M_T.shape[1]
        int n_pars = M_T.shape[0]

        # Needed to LAPACK dsysv
        char* uplo = 'U'  # store the upper triangle
        int nrhs = 1  # number of columns in b
        int info = 0  # if 0, success, otherwise some shit happened
        int lwork = n_times

        # Temp. variables needed for computation below
        double dy
        double var
        double chi2

    # We always produce B and b:
    # B = C + M @ Λ @ M.T
    # b = M @ µ

    # First zero-out arrays
    for n in range(n_times):
        b[n] = 0.
        for m in range(n_times):
            B[n, m] = 0.

    # Make the vector b:
    for n in range(n_times):
        for i in range(n_pars):
            b[n] += M_T[i, n] * mu[i]

    for n in range(n_times):
        B[n, n] = 1 / ivar[n]  # Assumes diagonal data covariance
        for m in range(n_times):
            for i in range(n_pars):
                for j in range(n_pars):
                    B[n, m] += M_T[j, n] * Lambda[i, j] * M_T[i, m]
            Btmp[n, m] = B[n, m]  # Make a copy because in-place shit below

    # LU factorization of B, used for determinant and inverse:
    lapack.dgetrf(&n_times, &n_times, &Btmp[0, 0], &n_times,
                  &ntime_ipiv[0], &info)
    if info != 0:
        return INF

    # Compute log-determinant:
    log_det_val = 0.
    for i in range(n_times):
        log_det_val += log(2*pi * fabs(Btmp[i, i]))
    # print(np.allclose(log_det_val, np.linalg.slogdet(2*np.pi*np.array(B))[1]))

    # Compute Binv - Btmp is now Binv:
    lapack.dgetri(&n_times, &Btmp[0, 0], &n_times,
                  &ntime_ipiv[0], &ntime_work[0], &lwork, &info)
    if info != 0:
        return INF
    # print(np.allclose(np.array(Btmp).ravel(),
    #                   np.linalg.inv(np.array(B)).ravel()))

    # Compute the chi2 term of the marg. likelihood
    chi2 = 0.
    for n in range(n_times):
        for m in range(n_times):
            chi2 += (b[m] - y[m]) * Btmp[n, m] * (b[n] - y[n])

    if make_aAinv == 1:
        # Ainv = Λinv + M.T @ Cinv @ M
        # First construct Ainv using the temp 2D array:
        for i in range(n_pars):
            for j in range(n_pars):
                Ainv[i, j] = Lambda_inv[i, j]
                for n in range(n_times):
                    Ainv[i, j] += M_T[j, n] * ivar[n] * M_T[i, n]
                Atmp[i, j] = Ainv[i, j]

        # Construct the RHS using the variable `a`:
        for i in range(n_pars):
            a[i] = 0.

        for n in range(n_times):
            for i in range(n_pars):
                a[i] += M_T[i, n] * ivar[n] * y[n]

        for i in range(n_pars):
            for j in range(n_pars):
                a[i] += Lambda_inv[i, j] * mu[j]

        # `a` on input is actually the RHS (e.g., b in Ax=b), but on output
        # is the vector we want!
        lapack.dsysv(uplo, &n_pars, &nrhs,
                     &Atmp[0, 0], &n_pars, # lda = same as n
                     &npar_ipiv[0], &a[0], &n_pars, # ldb = same as n
                     &npar_work[0], &lwork,
                     &info)

        if info != 0:
            return INF

    return -0.5 * (chi2 + log_det_val)


cpdef test_likelihood_worker(y, ivar, M, mu, Lambda, make_aAinv):
    """Used for testing the likelihood_worker function"""

    cdef:
        int n_times = len(ivar)
        int n_pars = len(mu)

        double[:, ::1] B = np.zeros((n_times, n_times), dtype=np.float64)
        double[::1] b = np.zeros(n_times, dtype=np.float64)
        double[:, ::1] Ainv = np.zeros((n_pars, n_pars), dtype=np.float64)
        double[::1] a = np.zeros(n_pars, dtype=np.float64)

        double[::1] _ivar = np.ascontiguousarray(ivar)
        double[::1] _y = np.ascontiguousarray(y)
        double[:, ::1] _M_T = np.ascontiguousarray(M.T)
        double[::1] _mu = np.ascontiguousarray(mu)
        double[:, ::1] _Lambda = np.ascontiguousarray(Lambda)

        # Needed for temporary storage in likelihood_worker:
        double[:, ::1] Btmp = np.zeros((n_times, n_times), dtype=np.float64)
        double[:, ::1] Atmp = np.zeros((n_pars, n_pars), dtype=np.float64)
        double[:, ::1] Linv = np.linalg.inv(Lambda)
        int[::1] npar_ipiv = np.zeros(n_pars, dtype=np.int32)
        int[::1] ntime_ipiv = np.zeros(n_times, dtype=np.int32)
        double[::1] npar_work = np.zeros(n_pars, dtype=np.float64)
        double[::1] ntime_work = np.zeros(n_times, dtype=np.float64)

    likelihood_worker(_y, _ivar, _M_T, _mu, _Lambda, Linv,
                      int(make_aAinv), B, b, Ainv, a,
                      Btmp, Atmp,
                      npar_ipiv, ntime_ipiv, npar_work, ntime_work)

    return np.array(b), np.array(B), np.array(a), np.array(Ainv)


cpdef batch_marginal_ln_likelihood(double[:,::1] chunk,
                                   data, joker_params):
    """Compute the marginal log-likelihood for a batch of prior samples.

    Parameters
    ----------
    chunk : numpy.ndarray
        A chunk of nonlinear parameter prior samples. For the default case,
        these are P (period, day), phi0 (phase at pericenter, rad), ecc
        (eccentricity), omega (argument of perihelion, rad). May also contain
        jitter as the last index.
    data : `~thejoker.data.RVData`
        The radial velocity data.
    joker_params : `~thejoker.sampler.params.JokerParams`
        The specification of parameters to infer with The Joker.
    """

    cdef:
        int n
        int n_samples = chunk.shape[0]
        int n_times = len(data)
        int n_poly = joker_params.poly_trend  # polynomial trend terms
        int n_pars = 1 + n_poly  # always have K, v trend terms

        double[::1] t = np.ascontiguousarray(data._t_bmjd, dtype='f8')
        double[::1] rv = np.ascontiguousarray(data.rv.value, dtype='f8')
        double[::1] ivar = np.ascontiguousarray(data.ivar.value, dtype='f8')

        # inverse variance array with jitter included
        double[::1] jitter_ivar = np.zeros(data.ivar.value.shape)
        double[:, ::1] Lambda = np.zeros((n_pars, n_pars))
        double[::1] mu = np.ascontiguousarray(joker_params.linear_par_mu)

        # design matrix
        double[:, ::1] M_T = np.zeros((n_pars, n_times))
        double logdet

        # likelihoodz
        double[::1] ll = np.full(n_samples, np.nan)

        # jitter stuff
        double t0 = data._t0_bmjd
        int _fixed_jitter
        double jitter

        # Needed for likelihood_worker:
        double[:, ::1] B = np.zeros((n_times, n_times), dtype=np.float64)
        double[::1] b = np.zeros(n_times, dtype=np.float64)
        double[:, ::1] Ainv = np.zeros((n_pars, n_pars), dtype=np.float64)
        double[::1] a = np.zeros(n_pars, dtype=np.float64)

        # Needed for temporary storage in likelihood_worker:
        double[:, ::1] Btmp = np.zeros((n_times, n_times), dtype=np.float64)
        double[:, ::1] Atmp = np.zeros((n_pars, n_pars), dtype=np.float64)
        double[:, ::1] Linv = np.zeros((n_pars, n_pars), dtype=np.float64)
        int[::1] npar_ipiv = np.zeros(n_pars, dtype=np.int32)
        int[::1] ntime_ipiv = np.zeros(n_times, dtype=np.int32)
        double[::1] npar_work = np.zeros(n_pars, dtype=np.float64)
        double[::1] ntime_work = np.zeros(n_times, dtype=np.float64)

        double K0
        int scale_K_prior_with_P = joker_params.scale_K_prior_with_P

    if scale_K_prior_with_P > 0:
        # TODO: allow customizing K0!
        K0 = (100 * u.m/u.s).to_value(joker_params._jitter_unit)
        Lambda[1:, 1:] = np.array(joker_params.linear_par_Lambda)
        Linv[1:, 1:] = np.linalg.inv(joker_params.linear_par_Lambda)
    else:
        Lambda = np.ascontiguousarray(joker_params.linear_par_Lambda)
        Linv = np.linalg.inv(Lambda)

    if joker_params._fixed_jitter:
        _fixed_jitter = 1
        jitter = joker_params.jitter.to(data.rv.unit).value

    else:
        _fixed_jitter = 0

    for n in range(n_samples):
        if _fixed_jitter == 0:
            jitter = chunk[n, 4]

        design_matrix(chunk[n, 0], chunk[n, 1], chunk[n, 2], chunk[n, 3],
                      t, t0, M_T, n_poly)

        # Note: jitter must be in same units as the data RV's / ivar!
        get_ivar(ivar, jitter, jitter_ivar)

        if scale_K_prior_with_P > 0:
            Lambda[0, 0] = K0**2 / (1 - chunk[n, 2]**2) * (chunk[n, 0] / 365.)**(-2/3.)
            Linv[0, 0] = 1 / Lambda[0, 0]

        # compute things needed for the ln(likelihood)
        ll[n] = likelihood_worker(rv, jitter_ivar, M_T, mu, Lambda, Linv,
                                  0, B, b, Ainv, a, Btmp, Atmp,
                                  npar_ipiv, ntime_ipiv, npar_work, ntime_work)

    return ll


cpdef batch_get_posterior_samples(double[:,::1] chunk,
                                  data, joker_params, rnd, return_logprobs):
    """TODO:

    Parameters
    ----------
    chunk : numpy.ndarray
        A chunk of nonlinear parameter prior samples. For the default case,
        these are P (period, day), phi0 (phase at pericenter, rad), ecc
        (eccentricity), omega (argument of perihelion, rad). May also contain
        jitter as the last index.
    data : `~thejoker.data.RVData`
        The radial velocity data.
    joker_params : `~thejoker.sampler.params.JokerParams`
        The specification of parameters to infer with The Joker.
    """

    cdef:
        int n, j
        int n_samples = chunk.shape[0]
        int n_times = len(data)
        int n_poly = joker_params.poly_trend # polynomial trend terms
        int n_pars = 1 + n_poly # always have K, v trend terms

        double[::1] t = np.ascontiguousarray(data._t_bmjd, dtype='f8')
        double[::1] rv = np.ascontiguousarray(data.rv.value, dtype='f8')
        double[::1] ivar = np.ascontiguousarray(data.ivar.value, dtype='f8')

        # inverse variance array with jitter included
        double[::1] jitter_ivar = np.zeros(data.ivar.value.shape)
        double[:, ::1] Lambda = np.zeros((n_pars, n_pars))
        double[::1] mu = np.ascontiguousarray(joker_params.linear_par_mu)

        # transpose of design matrix
        double[:,::1] M_T = np.zeros((n_pars, n_times))
        double ll

        # jitter
        double t0 = data._t0_bmjd
        int _fixed_jitter
        double jitter

        # Needed for likelihood_worker:
        double[:, ::1] B = np.zeros((n_times, n_times), dtype=np.float64)
        double[::1] b = np.zeros(n_times, dtype=np.float64)
        double[:, ::1] Ainv = np.zeros((n_pars, n_pars), dtype=np.float64)
        double[::1] a = np.zeros(n_pars, dtype=np.float64)

        # Needed for temporary storage in likelihood_worker:
        double[:, ::1] Btmp = np.zeros((n_times, n_times), dtype=np.float64)
        double[:, ::1] Atmp = np.zeros((n_pars, n_pars), dtype=np.float64)
        double[:, ::1] Linv = np.zeros((n_pars, n_pars), dtype=np.float64)
        int[::1] npar_ipiv = np.zeros(n_pars, dtype=np.int32)
        int[::1] ntime_ipiv = np.zeros(n_times, dtype=np.int32)
        double[::1] npar_work = np.zeros(n_pars, dtype=np.float64)
        double[::1] ntime_work = np.zeros(n_times, dtype=np.float64)

        double[:,::1] pars = np.zeros((n_samples,
            joker_params.num_params + int(return_logprobs)))
        double[::1] linear_pars

        double K0
        int scale_K_prior_with_P = joker_params.scale_K_prior_with_P

    if scale_K_prior_with_P > 0:
        # TODO: allow customizing K0!
        K0 = (100 * u.m/u.s).to_value(joker_params._jitter_unit)
        Lambda[1:, 1:] = np.array(joker_params.linear_par_Lambda)
        Linv[1:, 1:] = np.linalg.inv(joker_params.linear_par_Lambda)
    else:
        Lambda = np.ascontiguousarray(joker_params.linear_par_Lambda)
        Linv = np.linalg.inv(Lambda)

    for n in range(n_samples):
        pars[n, 0] = chunk[n, 0] # P
        pars[n, 1] = chunk[n, 1] # M0
        pars[n, 2] = chunk[n, 2] # e
        pars[n, 3] = chunk[n, 3] # omega
        pars[n, 4] = chunk[n, 4] # jitter

        design_matrix(chunk[n, 0], chunk[n, 1], chunk[n, 2], chunk[n, 3],
                      t, t0, M_T, n_poly)

        # jitter must be in same units as the data RV's / ivar!
        get_ivar(ivar, chunk[n, 4], jitter_ivar)

        if scale_K_prior_with_P > 0:
            Lambda[0, 0] = K0**2 / (1 - chunk[n, 2]**2) * (chunk[n, 0] / 365.)**(-2/3)
            Linv[0, 0] = 1 / Lambda[0, 0]

        # compute things needed for the ln(likelihood)
        ll = likelihood_worker(rv, jitter_ivar, M_T, mu, Lambda, Linv,
                               1, B, b, Ainv, a, Btmp, Atmp,
                               npar_ipiv, ntime_ipiv, npar_work, ntime_work)

        linear_pars = rnd.multivariate_normal(a, np.linalg.inv(Ainv))
        if linear_pars[0] < 0:
            # Swapping sign of K! TODO: do we want to do this?
            linear_pars[0] = fabs(linear_pars[0])
            pars[n, 3] += pi
            pars[n, 3] = pars[n, 3] % (2*pi) # HACK: I think this is safe

        pars[n, 5] = linear_pars[0]
        for j in range(n_poly):
            pars[n, 6+j] = linear_pars[1+j]

        if return_logprobs:
            pars[n, 6+n_poly] = ll

    return np.array(pars)
