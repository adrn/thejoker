# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

# Third-party
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython
cimport scipy.linalg.cython_lapack as lapack

# from libc.stdio cimport printf
from libc.math cimport pow, log, fabs

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
                        double[:,::1] A_T,
                        int n_trend):
    """Construct the elements of the design matrix.

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
    A_T : `numpy.ndarray`
        The transpose of the design matrix, to be filled by this function.
        Should have shape: (number of linear parameters, number of data points).

    """
    cdef:
        int i, j
        int n_times = t.shape[0]

    c_rv_from_elements(&t[0], &A_T[0,0], n_times,
                       P, 1., ecc, omega, phi0, t0,
                       anomaly_tol, anomaly_maxiter)

    if n_trend > 0: # only needed if we are fitting for a constant trend
        for j in range(n_times):
            A_T[1, j] = 1.

    if n_trend > 1: # only needed if more than constant trend
        for i in range(1, n_trend):
            for j in range(n_times):
                A_T[1+i, j] = pow(t[j] - t0, i)


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


cdef double tensor_vector_scalar(double[:,::1] A_T, double[::1] ivar,
                                 double[::1] y, double[:, ::1] Lambda,
                                 double[::1] linear_pars,
                                 int sample_linear_pars,
                                 double[:, ::1] ATCinvA, double[:, ::1] _A,
                                 int[:,::1] ipiv,
                                 double[:,::1] work, int[:,::1] lwork,
                                 double[::1] p):
    """Construct objects used to compute the marginal log-likelihood.

    Parameters
    ----------
    A_T : `numpy.ndarray`
        Transpose of the design matrix.
    ivar : `numpy.ndarray`
        Inverse-variance matrix.
    y : `numpy.ndarray`
        Data (in this case, radial velocities).
    Lambda : `numpy.ndarray`
        Inverse variance matrix for a Gaussian prior applied to the linear
        parameters.

    Outputs
    -------
    linear_pars : `numpy.ndarray`
        If ``sample_linear_pars==1``, this is the output array that contains the
        linear (velocity amplitude and trend) parameters.
    sample_linear_pars : int
        Controls whether or not to generate the linear parameters (set to 1
        for True).

    Other parameters
    ----------------


    Returns
    -------
    chi2 : float
        Chi-squared value.

    """

    cdef:
        int i, j, k
        int n_times = A_T.shape[1]
        int n_pars = A_T.shape[0]

        double chi2 = 0. # chi-squared
        double model_y, dy

        # Needed to LAPACK dsysv
        char* uplo = 'U' # store the upper triangle
        int nrhs = 1 # number of columns in b
        int info = 0 # if 0, success, otherwise some whack shit happened

    # first zero-out arrays
    for i in range(n_pars):
        p[i] = 0.
        for j in range(n_pars):
            ATCinvA[i, j] = 0.

    for k in range(n_times):
        for i in range(n_pars):
            p[i] += A_T[i, k] * ivar[k] * y[k]
            for j in range(n_pars):
                # implicit transpose of first A_T in the below
                ATCinvA[i, j] += A_T[i, k] * ivar[k] * A_T[j, k]
                ATCinvA[i, j] += Lambda[i, j]

    _A[:, :] = ATCinvA

    # p = np.linalg.solve(ATCinvA, ATCinv.dot(y))
    lapack.dsysv(uplo, &n_pars, &nrhs,
                 &_A[0,0], &n_pars, # lda = same as n
                 &ipiv[0,0], &p[0], &n_pars, # ldb = same as n
                 &work[0,0], &lwork[0,0],
                 &info)

    if info != 0:
        return INF

    for k in range(n_times):
        # compute predicted RV at this timestep
        model_y = 0.
        for i in range(n_pars):
            model_y += A_T[i, k] * p[i]

        # don't need log term for the jitter b.c. in likelihood func
        dy = model_y - y[k]
        chi2 += dy*dy * ivar[k]

    if chi2 == INF:
        return -INF

    logdet = logdet_term(ATCinvA, ivar, _A, ipiv)

    if sample_linear_pars == 1:
        cov = np.linalg.inv(ATCinvA)
        # TODO: set np.random.set_state(state) outside of here to make deterministic
        v_terms = np.random.multivariate_normal(p, cov)
        for k in range(n_pars):
            linear_pars[k] = v_terms[k]

    return 0.5*logdet - 0.5*chi2


cdef double logdet(double[:,::1] A, double[:,::1] B, int[:,::1] ipiv):
    """Compute the log-determinant of the (assumed) square, symmetric matrix A.

    Parameters
    ----------
    A : `numpy.ndarray`
        The input matrix.

    Returns
    -------
    log_det : float
        Log-determinant value.

    """
    cdef:
        int i
        int n_pars = A.shape[0] # A is assumed symmetric
        int info = 0 # if 0, success, otherwise some whack shit hapnd
        # int[:,::1] ipiv = np.ones((n_pars, n_pars), dtype=np.int32)

        double log_det = 0.

        # double[:,::1] B = np.zeros((A.shape[0], A.shape[1]))

    # Copy because factorization is done in place
    B[:,:] = A

    # LU factorization of B
    lapack.dgetrf(&n_pars, &n_pars, &B[0,0], &n_pars, &ipiv[0,0], &info)

    # Compute determinant
    for i in range(n_pars):
        log_det += log(fabs(B[i,i]))

    if info != 0:
        return INF

    return log_det


cdef double logdet_term(double[:,::1] ATCinvA, double[::1] ivar,
                        double[:,::1] B, int[:,::1] ipiv):
    """Compute the log-determinant term of the log-likelihood."""
    cdef:
        int i
        int n_pars = ATCinvA.shape[0] # symmetric
        int n_times = ivar.shape[0]
        double ld

    ld = logdet(ATCinvA, B, ipiv)
    ld -= n_pars * LN_2PI

    for i in range(n_times):
        ld = ld + log(ivar[i])
    ld = ld - n_times * LN_2PI

    return ld


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
        int n_poly = joker_params.poly_trend # polynomial trend terms
        int n_pars = 1 + n_poly # always have K, v trend terms

        double[::1] t = np.ascontiguousarray(data._t_bmjd, dtype='f8')
        double[::1] rv = np.ascontiguousarray(data.rv.value, dtype='f8')
        double[::1] ivar = np.ascontiguousarray(data.ivar.value, dtype='f8')

        # inverse variance array with jitter included
        double[::1] jitter_ivar = np.zeros(data.ivar.value.shape)
        double[:, ::1] Lambda = np.ascontiguousarray(joker_params.linear_par_Vinv)

        # transpose of design matrix
        double[:,::1] A_T = np.zeros((n_pars, n_times))
        double logdet

        # likelihoodz
        double[::1] ll = np.full(n_samples, np.nan)

        # lol
        double t0 = data._t0_bmjd
        int _fixed_jitter
        double jitter

        double[::1] linear_pars = np.zeros(n_pars)

        # needed for temporary storage in tensor_vector_scalar
        double[:, ::1] ATCinvA = np.zeros((n_pars, n_pars))
        double[:, ::1] _A = np.zeros((n_pars, n_pars))
        double[::1] p = np.zeros(n_pars)
        int[:,::1] ipiv = np.zeros((n_pars, n_pars), dtype=np.int32) # ??
        double[:,::1] work = np.zeros((n_pars, n_pars), dtype=np.float64) # ??
        int[:,::1] lwork = np.ones((n_pars, n_pars), dtype=np.int32) # ??

    if joker_params._fixed_jitter:
        _fixed_jitter = 1
        jitter = joker_params.jitter.to(data.rv.unit).value

    else:
        _fixed_jitter = 0

    for n in range(n_samples):
        if _fixed_jitter == 0:
            jitter = chunk[n, 4]

        design_matrix(chunk[n, 0], chunk[n, 1], chunk[n, 2], chunk[n, 3],
                      t, t0, A_T, n_poly)

        # Note: jitter must be in same units as the data RV's / ivar!
        get_ivar(ivar, jitter, jitter_ivar)

        # compute things needed for the ln(likelihood)
        ll[n] = tensor_vector_scalar(A_T, jitter_ivar, rv, Lambda,
                                     linear_pars, 0, # IGNORED PLACEHOLDER
                                     ATCinvA, _A, ipiv, work, lwork, p)

    return ll


cpdef batch_get_posterior_samples(double[:,::1] chunk,
                                  data, joker_params, rnd, return_logprobs):
    """TODO"

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
        double[:, ::1] Lambda = np.ascontiguousarray(joker_params.linear_par_Vinv)

        # transpose of design matrix
        double[:,::1] A_T = np.zeros((n_pars, n_times))
        double ll

        # lol
        double t0 = data._t0_bmjd
        int _fixed_jitter
        double jitter

        double[::1] linear_pars = np.zeros(n_pars)

        # needed for temporary storage in tensor_vector_scalar
        double[:, ::1] ATCinvA = np.zeros((n_pars, n_pars))
        double[:, ::1] _A = np.zeros((n_pars, n_pars))
        double[::1] p = np.zeros(n_pars)
        int[:,::1] ipiv = np.zeros((n_pars, n_pars), dtype=np.int32) # ??
        double[:,::1] work = np.zeros((n_pars, n_pars), dtype=np.float64) # ??
        int[:,::1] lwork = np.ones((n_pars, n_pars), dtype=np.int32) # ??

        double[:,::1] pars = np.zeros((n_samples,
            joker_params.num_params + int(return_logprobs)))

    for n in range(n_samples):
        pars[n, 0] = chunk[n, 0] # P
        pars[n, 1] = chunk[n, 1] # M0
        pars[n, 2] = chunk[n, 2] # e
        pars[n, 3] = chunk[n, 3] # omega
        pars[n, 4] = chunk[n, 4] # jitter

        design_matrix(chunk[n, 0], chunk[n, 1], chunk[n, 2], chunk[n, 3],
                      t, t0, A_T, n_poly)

        # jitter must be in same units as the data RV's / ivar!
        get_ivar(ivar, chunk[n, 4], jitter_ivar)

        # compute things needed for the ln(likelihood)
        ll = tensor_vector_scalar(A_T, jitter_ivar, rv, Lambda,
                                  linear_pars, 1, # do make samples (1 = True)
                                  ATCinvA, _A, ipiv, work, lwork, p)

        K = linear_pars[0]
        if K < 0:
            # Swapping sign of K!
            K = np.abs(K)
            pars[n, 3] += np.pi
            pars[n, 3] = pars[n, 3] % (2*np.pi) # HACK: I think this is safe

        pars[n, 5] = K
        for j in range(n_poly):
            pars[n, 6+j] = linear_pars[1+j]

        if return_logprobs:
            pars[n, 6+n_poly] = ll

    return np.array(pars)
