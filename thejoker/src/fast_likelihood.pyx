# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False
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
import theano.tensor as tt

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
    # TODO: these should be pulled from the instance of TheJoker!
    double anomaly_tol = 1E-10  # passed to c_rv_from_elements
    int anomaly_maxiter = 128   # passed to c_rv_from_elements


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
        int i

    for i in range(ivar.shape[0]):
        new_ivar[i] = ivar[i] / (1 + s*s * ivar[i])


cdef class CJokerHelper:
    cdef:
        # Counts:
        int n_times
        int n_poly
        int n_linear
        int n_pars

        # Data:
        double t0
        double[::1] t
        double[::1] rv
        double[::1] ivar

        # Needed for runtime:
        double[::1] s_ivar
        double[:, ::1] trend_M
        double[:,::1] M_T

        # Prior on linear parameters:
        # TODO: Lambda should be a matrix, but we currently only support
        # diagonal variance
        double[::1] mu
        double[::1] Lambda
        int fixed_K_prior  # TODO: total HACK
        double sigma_K0  # TODO: total HACK

        # Needed for temporary storage in likelihood_worker:
        double[:, ::1] Btmp
        double[:, ::1] Atmp
        int[::1] npar_ipiv
        int[::1] ntime_ipiv
        double[::1] npar_work
        double[::1] ntime_work

        # Needed for internal work / output from likelihood_worker:
        public double[:, ::1] B
        public double[::1] b
        public double[:, ::1] Ainv
        public double[::1] a


    def __init__(self, data, prior, double[:, ::1] trend_M):
        cdef int i, n

        # Counting:
        self.n_times = len(data)  # number of data pints
        self.n_poly = prior.poly_trend  # polynomial trend terms
        self.n_linear = 1 + self.n_poly  # K, trend - TODO: v0_offsets
        self.n_pars = len(prior.par_names)

        # Data:
        self.t0 = data._t0_bmjd
        self.t = np.ascontiguousarray(data._t_bmjd, dtype='f8')
        self.rv = np.ascontiguousarray(data.rv.value, dtype='f8')
        self.ivar = np.ascontiguousarray(data.ivar.value, dtype='f8')
        self.trend_M = trend_M

        # ivar with jitter included
        self.s_ivar = np.zeros(self.n_times, dtype='f8')

        # Transpose of design matrix: Fill the columns for the linear part of M
        if (trend_M.shape[0] != self.n_times
                or trend_M.shape[1] != self.n_linear):
            raise ValueError("Invalid design matrix shape: {}, expected: {}"
                             .format(trend_M.shape,
                                     (self.n_times, self.n_linear)))

        self.M_T = np.zeros((self.n_linear, self.n_times))
        for n in range(self.n_times):
            for i in range(self.n_linear-1):
                self.M_T[1 + i, n] = trend_M[n, i]

        # Needed for temporary storage in likelihood_worker:
        self.Btmp = np.zeros((self.n_times, self.n_times), dtype=np.float64)
        self.Atmp = np.zeros((self.n_linear, self.n_linear), dtype=np.float64)
        self.npar_ipiv = np.zeros(self.n_linear, dtype=np.int32)
        self.ntime_ipiv = np.zeros(self.n_times, dtype=np.int32)
        self.npar_work = np.zeros(self.n_linear, dtype=np.float64)
        self.ntime_work = np.zeros(self.n_times, dtype=np.float64)

        # Needed for internal work / output from likelihood_worker:
        self.B = np.zeros((self.n_times, self.n_times), dtype=np.float64)
        self.b = np.zeros(self.n_times, dtype=np.float64)
        self.Ainv = np.zeros((self.n_linear, self.n_linear), dtype=np.float64)
        self.a = np.zeros(self.n_linear, dtype=np.float64)

        # TODO: Lambda should be a matrix, but we currently only support
        # diagonal variance on Lambda
        self.mu = np.zeros(self.n_linear)
        self.Lambda = np.zeros(self.n_linear)

        # ---------------------------------------------------------------------
        # TODO: This is a hack that is totally inconsistent with the flexible
        # prior specification allowed through pymc3. However, we currently only
        # support constant values for the mean, and only support constant or one
        # special scaled prior with the variance
        for i, name in enumerate(prior._linear_pars.keys()):
            self.mu[i] = prior.model[name].distribution.mean.eval()
            if not isinstance(prior.model[name].distribution.sd,
                              tt.TensorConstant):
                self.fixed_K_prior = 0
                self.sigma_K0 = prior._sigma_K0.to_value(data.rv.unit)

        for i, name in enumerate(prior._linear_pars.keys()):
            if self.fixed_K_prior == 0 and i == 0:
                # Skip K term: the prior on K is scaled with P, e, so we set it in # the loop over nonlinear samples below
                continue

            self.Lambda[i] = prior.model[name].distribution.sd.eval()
        # ---------------------------------------------------------------------

    cdef double likelihood_worker(self, int make_aAinv):
        """The argument controls whether to make a, Ainv"""

        cdef:
            int i, j, n, m  # i,j used below for n_pars, n,m used for n_times

            # Needed to LAPACK dsysv
            char* uplo = 'U'  # store the upper triangle
            int nrhs = 1  # number of columns in b
            int info = 0  # if 0, success, otherwise some shit happened
            int lwork = self.n_times

            # Temp. variables needed for computation below
            double dy
            double var
            double chi2

        # We always produce B and b:
        # B = C + M @ Λ @ M.T
        # b = M @ µ

        # First zero-out arrays
        for n in range(self.n_times):
            self.b[n] = 0.
            for m in range(self.n_times):
                self.B[n, m] = 0.

        # Make the vector b:
        for n in range(self.n_times):
            for i in range(self.n_linear):
                self.b[n] += self.M_T[i, n] * self.mu[i]

        for n in range(self.n_times):
            self.B[n, n] = 1 / self.ivar[n]  # TODO: Assumes diagonal covariance
            for m in range(self.n_times):
                # TODO: this now assumes diagonal Lambda
                # for i in range(self.n_linear):
                #     for j in range(self.n_linear):
                #         B[n, m] += M_T[j, n] * Lambda[i, j] * M_T[i, m]
                for i in range(self.n_linear):
                    self.B[n, m] += (self.M_T[i, n] * self.Lambda[i]
                                     * self.M_T[i, m])

                # Make a copy because in-place shit below
                self.Btmp[n, m] = self.B[n, m]

        # LU factorization of B, used for determinant and inverse:
        lapack.dgetrf(&(self.n_times), &(self.n_times), &(self.Btmp[0, 0]),
                      &(self.n_times), &(self.ntime_ipiv)[0], &info)
        if info != 0:
            return INF

        # Compute log-determinant:
        log_det_val = 0.
        for i in range(self.n_times):
            log_det_val += log(2*pi * fabs(self.Btmp[i, i]))
        # print(np.allclose(log_det_val,
        #                   np.linalg.slogdet(2*np.pi*np.array(B))[1]))

        # Compute Binv - Btmp is now Binv:
        lapack.dgetri(&(self.n_times), &(self.Btmp[0, 0]), &self.n_times,
                      &self.ntime_ipiv[0], &self.ntime_work[0], &lwork, &info)
        if info != 0:
            return INF
        # print(np.allclose(np.array(Btmp).ravel(),
        #                   np.linalg.inv(np.array(B)).ravel()))

        # Compute the chi2 term of the marg. likelihood
        chi2 = 0.
        for n in range(self.n_times):
            for m in range(self.n_times):
                chi2 += ((self.b[m] - self.rv[m])
                         * self.Btmp[n, m]
                         * (self.b[n] - self.rv[n]))

        if make_aAinv == 1:
            # Ainv = Λinv + M.T @ Cinv @ M
            # First construct Ainv using the temp 2D array:
            for i in range(self.n_linear):
                self.Ainv[i, j] = 1 / self.Lambda[i]
                for j in range(self.n_linear):
                    # TODO: with line above, this now assumes diagonal Lambda
                    # Ainv[i, j] = Lambda_inv[i, j]
                    for n in range(self.n_times):
                        self.Ainv[i, j] += (self.M_T[j, n] * self.ivar[n]
                                            * self.M_T[i, n])
                    self.Atmp[i, j] = self.Ainv[i, j]

            # Construct the RHS using the variable `a`:
            for i in range(self.n_linear):
                self.a[i] = 0.

            for n in range(self.n_times):
                for i in range(self.n_linear):
                    self.a[i] += self.M_T[i, n] * self.ivar[n] * self.rv[n]

            for i in range(self.n_linear):
                # TODO: this now assumes diagonal Lambda
                # for j in range(self.n_linear):
                #     a[i] += Lambda_inv[i, j] * mu[j]
                self.a[i] += self.mu[j] / self.Lambda[i]

            # `a` on input is actually the RHS (e.g., b in Ax=b), but on output
            # is the vector we want!
            lapack.dsysv(uplo, &self.n_linear, &nrhs,
                         &self.Atmp[0, 0], &self.n_linear, # lda = same as n
                         &self.npar_ipiv[0], &self.a[0], &self.n_linear,
                         &self.npar_work[0], &lwork,
                         &info)

            if info != 0:
                return INF

        return -0.5 * (chi2 + log_det_val)


    cpdef batch_marginal_ln_likelihood(self, double[:, ::1] chunk):
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
            int n, i
            int n_samples = chunk.shape[0]

            # the log-likelihood values
            double[::1] ll = np.full(n_samples, np.nan)

            double P, e, om, M0

        for n in range(n_samples):
            # TODO: need to make sure the chunk is always in this order!
            P = chunk[n, 0]
            e = chunk[n, 1]
            om = chunk[n, 2]
            M0 = chunk[n, 3]

            c_rv_from_elements(&self.t[0], &self.M_T[0, 0], self.n_times,
                               P, 1., e, om, M0, self.t0,
                               anomaly_tol, anomaly_maxiter)

            # Note: jitter must be in same units as the data RV's / ivar!
            # TODO: a consequence here is that the jitter must always be
            # defined in the chunk, so it can't be fixed to a value not set in
            # the prior cache
            get_ivar(self.ivar, chunk[n, 4], self.s_ivar)

            # TODO: this is a continuation of the massive hack introduced above.
            if self.fixed_K_prior == 0:
                self.Lambda[0] = (self.sigma_K0**2 / (1 - chunk[n, 2]**2)
                                  * (chunk[n, 0] / 365.)**(-2/3.))

            # compute things needed for the ln(likelihood)
            ll[n] = self.likelihood_worker(0)

        return ll

    cpdef batch_get_posterior_samples(self, double[:,::1] chunk, rnd,
                                      return_logprobs):
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

            # Transpose of design matrix
            double[:,::1] M_T = np.zeros((self.n_linear, self.n_times))
            double ll

            double[:,::1] pars = np.zeros((n_samples,
                self.n_pars + int(return_logprobs)))
            double[::1] linear_pars

        for n in range(n_samples):
            pars[n, 0] = chunk[n, 0] # P
            pars[n, 1] = chunk[n, 1] # e
            pars[n, 2] = chunk[n, 2] # omega
            pars[n, 3] = chunk[n, 3] # M0
            pars[n, 4] = chunk[n, 4] # jitter

            # TODO: audit order of chunk[...]'s and what c_rv_from_elements
            c_rv_from_elements(&self.t[0], &self.M_T[0, 0], self.t.shape[0],
                               chunk[n, 0], 1., chunk[n, 1],
                               chunk[n, 2], chunk[n, 3], self.t0,
                               anomaly_tol, anomaly_maxiter)

            # Note: jitter must be in same units as the data RV's / ivar!
            # TODO: a consequence here is that the jitter must always be
            # defined in the chunk, so it can't be fixed to a value not set in
            # the prior cache
            get_ivar(self.ivar, chunk[n, 4], self.s_ivar)

            # TODO: this is a continuation of the massive hack introduced above.
            if self.fixed_K_prior == 0:
                self.Lambda[0] = (self.sigma_K0**2 / (1 - chunk[n, 2]**2)
                                  * (chunk[n, 0] / 365.)**(-2/3.))

            # compute likelihood, but also generate a, Ainv
            ll = self.likelihood_worker(1)

            # TODO: this calls back to numpy!
            # TODO: https://github.com/bashtage/randomgen instead?
            linear_pars = rnd.multivariate_normal(self.a,
                                                  np.linalg.inv(self.Ainv))

            pars[n, 5] = linear_pars[0]
            for j in range(self.n_poly):
                pars[n, 6+j] = linear_pars[1+j]

            if return_logprobs:
                pars[n, 6+self.n_poly] = ll

        return np.array(pars)


# cpdef test_likelihood_worker(self, M, mu, Lambda, make_aAinv):
    #     """Used for testing the likelihood_worker function"""

    #     cdef:
    #         int n_times = len(ivar)
    #         int n_pars = len(mu)

    #         double[:, ::1] B = np.zeros((n_times, n_times), dtype=np.float64)
    #         double[::1] b = np.zeros(n_times, dtype=np.float64)
    #         double[:, ::1] Ainv = np.zeros((n_pars, n_pars), dtype=np.float64)
    #         double[::1] a = np.zeros(n_pars, dtype=np.float64)

    #         double[::1] _ivar = np.ascontiguousarray(ivar)
    #         double[::1] _y = np.ascontiguousarray(y)
    #         double[:, ::1] _M_T = np.ascontiguousarray(M.T)
    #         double[::1] _mu = np.ascontiguousarray(mu)
    #         double[:, ::1] _Lambda = np.ascontiguousarray(Lambda)

    #         # Needed for temporary storage in likelihood_worker:
    #         double[:, ::1] Btmp = np.zeros((n_times, n_times), dtype=np.float64)
    #         double[:, ::1] Atmp = np.zeros((n_pars, n_pars), dtype=np.float64)
    #         double[:, ::1] Linv = np.linalg.inv(Lambda)
    #         int[::1] npar_ipiv = np.zeros(n_pars, dtype=np.int32)
    #         int[::1] ntime_ipiv = np.zeros(n_times, dtype=np.int32)
    #         double[::1] npar_work = np.zeros(n_pars, dtype=np.float64)
    #         double[::1] ntime_work = np.zeros(n_times, dtype=np.float64)

    #     likelihood_worker(_y, _ivar, _M_T, _mu, _Lambda, Linv,
    #                     int(make_aAinv), B, b, Ainv, a,
    #                     Btmp, Atmp,
    #                     npar_ipiv, ntime_ipiv, npar_work, ntime_work)

    #     return np.array(b), np.array(B), np.array(a), np.array(Ainv)