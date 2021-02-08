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

# from libc.stdio cimport printf
from libc.math cimport pow, log, fabs, pi


cdef extern from "src/twobody.h":
    void c_rv_from_elements(double *t, double *rv, int N_t,
                            double P, double K, double e, double omega,
                            double phi0, double t0, double tol, int maxiter)

cdef:
    double INF = float('inf')
    # TODO: these should be pulled from the instance of TheJoker!
    double anomaly_tol = 1E-10  # passed to c_rv_from_elements
    int anomaly_maxiter = 128   # passed to c_rv_from_elements


# NOTE: if this order is changed, make sure to change the indexing order at
# the two other NOTE's below
_nonlinear_packed_order = ['P', 'e', 'omega', 'M0', 's']
_nonlinear_internal_units = {'P': u.day,
                             'e' : u.one,
                             'omega': u.radian,
                             'M0': u.radian}


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
        int n_offsets
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
        double P0  # TODO: total HACK
        double max_K  # TODO: total HACK

        # Needed for temporary storage in likelihood_worker:
        double[:, ::1] Btmp
        double[:, ::1] Atmp
        int[::1] npar_ipiv
        int[::1] ntime_ipiv
        double[::1] npar_work
        double[::1] ntime_work

        # Needed for internal work / output from likelihood_worker:
        public double[:, ::1] B
        public double[:, ::1] Binv
        public double[::1] b
        public double[:, ::1] A
        public double[:, ::1] Ainv
        public double[::1] a

        # Random number generation
        public object prior
        public object data
        public object internal_units
        public object packed_order

    def __reduce__(self):
        return (CJokerHelper, (self.data, self.prior, np.array(self.trend_M)))

    def __init__(self, data, prior, double[:, ::1] trend_M):
        cdef int i, j, n

        self.prior = prior
        self.data = data

        # Internal units needed for calculations below.
        # Note: order here matters! This is the order in which prior samples
        # will be unpacked externally!
        self.internal_units = {}
        self.internal_units['P'] = _nonlinear_internal_units['P']
        self.internal_units['e'] = _nonlinear_internal_units['e']
        self.internal_units['omega'] = _nonlinear_internal_units['omega']
        self.internal_units['M0'] = _nonlinear_internal_units['M0']
        self.internal_units['s'] = self.data.rv.unit
        self.internal_units['K'] = self.data.rv.unit
        self.internal_units['v0'] = self.data.rv.unit

        # v0 offsets must be between v0 and v1, if v1 is present
        for offset in prior.v0_offsets:
            self.internal_units[offset.name] = self.data.rv.unit

        for i, name in enumerate(prior._v_trend_names):
            self.internal_units[name] = self.data.rv.unit / u.day ** i

        # The assumed order of the nonlinear parameters used below to read from
        # packed samples array
        self.packed_order = _nonlinear_packed_order

        # Counting:
        self.n_times = len(data)  # number of data pints
        self.n_poly = prior.poly_trend  # polynomial trend terms
        self.n_offsets = prior.n_offsets  # v0 offsets
        self.n_linear = 1 + self.n_poly + self.n_offsets # K, trend
        self.n_pars = len(prior.par_names)

        # Data:
        self.t0 = data._t_ref_bmjd
        self.t = np.ascontiguousarray(data._t_bmjd, dtype='f8')
        self.rv = np.ascontiguousarray(data.rv.value, dtype='f8')
        self.ivar = np.ascontiguousarray(
            data.ivar.to_value(1 / self.data.rv.unit**2), dtype='f8')
        self.trend_M = trend_M

        # ivar with jitter included
        self.s_ivar = np.zeros(self.n_times, dtype='f8')

        # Transpose of design matrix: Fill the columns for the linear part of M
        # - trend shape: K, v0 + v0_offsets, poly_trend-1
        if (trend_M.shape[0] != self.n_times
                or trend_M.shape[1] != self.n_linear - 1):
            raise ValueError("Invalid design matrix shape: {}, expected: {}"
                             .format(trend_M.shape,
                                     (self.n_times,
                                     self.n_linear - 1)))

        self.M_T = np.zeros((self.n_linear, self.n_times))
        for n in range(self.n_times):
            for i in range(1, self.n_linear):
                self.M_T[i, n] = trend_M[n, i-1]

        # Needed for temporary storage in likelihood_worker:
        self.Btmp = np.zeros((self.n_times, self.n_times), dtype=np.float64)
        self.Atmp = np.zeros((self.n_linear, self.n_linear), dtype=np.float64)
        self.npar_ipiv = np.zeros(self.n_linear, dtype=np.int32)
        self.ntime_ipiv = np.zeros(self.n_times, dtype=np.int32)
        self.npar_work = np.zeros(self.n_linear, dtype=np.float64)
        self.ntime_work = np.zeros(self.n_times, dtype=np.float64)

        # Needed for internal work / output from likelihood_worker:
        self.B = np.zeros((self.n_times, self.n_times), dtype=np.float64)
        self.Binv = np.zeros((self.n_times, self.n_times), dtype=np.float64)
        self.b = np.zeros(self.n_times, dtype=np.float64)
        self.Ainv = np.zeros((self.n_linear, self.n_linear), dtype=np.float64)
        self.A = np.zeros((self.n_linear, self.n_linear), dtype=np.float64)
        self.a = np.zeros(self.n_linear, dtype=np.float64)

        # TODO: Lambda should be a matrix, but we currently only support
        # diagonal variance on Lambda
        self.mu = np.zeros(self.n_linear + self.n_offsets)
        self.Lambda = np.zeros(self.n_linear + self.n_offsets)

        # put v0_offsets variances into Lambda
        # - validated to be Normal() in JokerPrior
        import exoplanet.units as xu
        for i in range(self.n_offsets):
            name = prior.v0_offsets[i].name
            dist = prior.v0_offsets[i].distribution
            _unit = getattr(prior.model[name], xu.UNIT_ATTR_NAME)
            to_unit = self.internal_units[name]

            # K, v0 = 2 - start at index 2
            self.mu[2+i] = (dist.mean.eval() * _unit).to_value(to_unit)
            self.Lambda[2+i] = (dist.sd.eval() * _unit).to_value(to_unit) ** 2

        # ---------------------------------------------------------------------
        # TODO: This is a bit of a hack:
        from ..distributions import FixedCompanionMass
        if isinstance(prior.pars['K'].distribution, FixedCompanionMass):
            self.fixed_K_prior = 0
        else:
            self.fixed_K_prior = 1

        for i, name in enumerate(prior._linear_equiv_units.keys()):
            dist = prior.model[name].distribution
            _unit = getattr(prior.model[name], xu.UNIT_ATTR_NAME)
            to_unit = self.internal_units[name]
            mu = (dist.mean.eval() * _unit).to_value(to_unit)

            if name == 'K' and self.fixed_K_prior == 0:
                # TODO: here's the major hack
                self.sigma_K0 = dist._sigma_K0.to_value(to_unit)
                self.P0 = dist._P0.to_value(getattr(prior.pars['P'],
                                                    xu.UNIT_ATTR_NAME))
                self.max_K = dist._max_K.to_value(to_unit)
                self.mu[i] = mu

            elif name == 'v0':
                self.Lambda[i] = (dist.sd.eval() * _unit).to_value(to_unit) ** 2
                self.mu[i] = mu

            else:  # v1, v2, etc.
                j = i + self.n_offsets
                self.Lambda[j] = (dist.sd.eval() * _unit).to_value(to_unit) ** 2
                self.mu[j] = mu
        # ---------------------------------------------------------------------

    cdef int make_AAinv(self):
        cdef:
            int i, j
            int info = 0
            int lwork = self.n_linear

        # Zero-out array:
        for i in range(self.n_linear):
            for j in range(self.n_linear):
                self.Ainv[i, j] = 0.

        # Ainv = Λinv + M.T @ Cinv @ M
        # First construct Ainv using the temp 2D array:
        for i in range(self.n_linear):
            self.Ainv[i, i] = 1 / self.Lambda[i]
            for j in range(self.n_linear):
                # TODO: with line above, this now assumes diagonal Lambda
                # Ainv[i, j] = Lambda_inv[i, j]
                for n in range(self.n_times):
                    self.Ainv[i, j] += (self.M_T[j, n] * self.ivar[n]
                                        * self.M_T[i, n])

                # Make a copy because we do in-place LU decomp. below
                self.Atmp[i, j] = self.Ainv[i, j]

        # LU factorization of Ainv, used for inverting to compute A:
        lapack.dgetrf(&(self.n_linear), &(self.n_linear), &(self.Atmp[0, 0]),
                      &(self.n_linear), &(self.npar_ipiv)[0], &info)
        if info != 0:
            return -1
        # Atmp is now the LU-decomposed Ainv

        # Compute A from Ainv - Atmp is now A:
        lapack.dgetri(&(self.n_linear), &(self.Atmp[0, 0]), &self.n_linear,
                      &self.npar_ipiv[0], &self.npar_work[0], &lwork, &info)
        if info != 0:
            return -1

        for i in range(self.n_linear):
            for j in range(self.n_linear):
                self.A[i, j] = self.Atmp[i, j]

        return 0

    cdef double make_bBBinv(self):
        cdef:
            int i, j, n, m
            int info = 0
            double log_det_val

        # Make the vector b:
        for n in range(self.n_times):
            self.b[n] = 0.
            for i in range(self.n_linear):
                self.b[n] += self.M_T[i, n] * self.mu[i]

            # zero out B
            for m in range(self.n_times):
                self.B[n, m] = 0.

        # First make B:
        for n in range(self.n_times):
            self.B[n, n] = 1 / self.ivar[n]  # TODO: Assumes diagonal covariance
            for m in range(self.n_times):
                self.Binv[n, m] = 0.
                # TODO: this now assumes diagonal Lambda
                # for i in range(self.n_linear):
                #     for j in range(self.n_linear):
                #         B[n, m] += M_T[j, n] * Lambda[i, j] * M_T[i, m]
                for i in range(self.n_linear):
                    self.B[n, m] += (self.M_T[i, n] * self.Lambda[i]
                                     * self.M_T[i, m])

                self.Btmp[n, m] = self.B[n, m]

        # Compute Binv using A and the Woodbury matrix identity:
        # Binv = Cinv + Cinv @ M @ A @ M.T @ Cinv
        for n in range(self.n_times):
            self.Binv[n, n] = self.ivar[n]
            for i in range(self.n_linear):
                for m in range(self.n_times):
                    for j in range(self.n_linear):
                        self.Binv[n, m] -= (self.ivar[n] * self.M_T[i, n]
                                            * self.A[i, j] * self.M_T[j, m]
                                            * self.ivar[m])

        # Binv_py = np.diag(self.ivar) - np.diag(self.ivar) @ self.M_T.T @ self.A @ self.M_T @ np.diag(self.ivar)
        # print(np.allclose(Binv_py, np.array(self.Binv)))

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
        #                   np.linalg.slogdet(2*np.pi*np.array(self.B))[1]))

        return log_det_val

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

        info = self.make_AAinv()
        if info < 0:
            return INF

        log_det_val = self.make_bBBinv()

        # Compute the chi2 term of the marg. likelihood
        chi2 = 0.
        for n in range(self.n_times):
            for m in range(self.n_times):
                chi2 += ((self.b[m] - self.rv[m])
                         * self.Binv[n, m]
                         * (self.b[n] - self.rv[n]))

        if make_aAinv == 1:
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
                self.a[i] += self.mu[i] / self.Lambda[i]

            # `a` on input is actually the RHS (e.g., b in Ax=b), but on output
            # is the vector we want!

            for i in range(self.n_linear):
                for j in range(self.n_linear):
                    self.Atmp[i, j] = self.Ainv[i, j]

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
            A chunk of nonlinear parameter prior samples.
            Expected order: P, e, omega, M0, s (jitter).
        """
        cdef:
            int n
            int n_samples = chunk.shape[0]
            double P, e, om, M0

            # the log-likelihood values
            double[::1] ll = np.full(n_samples, np.nan)

        for n in range(n_samples):
            # NOTE: need to make sure the chunk is always in this order! If this
            # is changed, change "packed_order" above
            P = chunk[n, 0]
            e = chunk[n, 1]
            om = chunk[n, 2]
            M0 = chunk[n, 3]

            c_rv_from_elements(&self.t[0], &self.M_T[0, 0], self.n_times,
                               P, 1., e, om, M0, self.t0,
                               anomaly_tol, anomaly_maxiter)

            # Note: jitter must be in same units as the data RV's / ivar
            get_ivar(self.ivar, chunk[n, 4], self.s_ivar)

            # TODO: this is a continuation of the massive hack introduced above.
            if self.fixed_K_prior == 0:
                self.Lambda[0] = (self.sigma_K0**2 / (1 - e**2)
                                  * (P / self.P0)**(-2/3.))
                self.Lambda[0] = min(self.max_K**2, self.Lambda[0])

            # compute things needed for the ln(likelihood)
            ll[n] = self.likelihood_worker(0)

        return ll

    cpdef batch_get_posterior_samples(self, double[:, ::1] chunk,
                                      int n_linear_samples_per,
                                      object random_state):
        """TODO:

        Parameters
        ----------
        chunk : numpy.ndarray
            A chunk of nonlinear parameter prior samples.
            Expected order: P, e, omega, M0, s (jitter).
        """

        cdef:
            int n, j, k
            int n_samples = chunk.shape[0]
            double P, e, om, M0

            # Transpose of design matrix
            double[:, ::1] M_T = np.zeros((self.n_linear, self.n_times))

            # the log-likelihood values
            double[:, ::1] ll = np.full((n_samples, n_linear_samples_per),
                                        np.nan)
            double _ll

            # The samples
            double[:, :, ::1] samples = np.zeros((n_samples,
                                                  n_linear_samples_per,
                                                  self.n_pars))
            double[:, ::1] linear_pars = np.zeros((n_linear_samples_per,
                                                   self.n_linear))

        for n in range(n_samples):
            # NOTE: need to make sure the chunk is always in this order! If this
            # is changed, change "packed_order" above
            P = chunk[n, 0]
            e = chunk[n, 1]
            om = chunk[n, 2]
            M0 = chunk[n, 3]

            c_rv_from_elements(&self.t[0], &self.M_T[0, 0], self.n_times,
                               P, 1., e, om, M0, self.t0,
                               anomaly_tol, anomaly_maxiter)

            # Note: jitter must be in same units as the data RV's / ivar
            get_ivar(self.ivar, chunk[n, 4], self.s_ivar)

            # TODO: this is a continuation of the massive hack introduced above.
            if self.fixed_K_prior == 0:
                self.Lambda[0] = (self.sigma_K0**2 / (1 - e**2)
                                  * (P / self.P0)**(-2/3.))

            # compute likelihood, but also generate a, Ainv
            _ll = self.likelihood_worker(1)  # the 1 is "True"

            # TODO: FIXME: this calls back to numpy at the Python layer
            # - use https://github.com/bashtage/randomgen instead?
            # a and Ainv are populated by the likelihood_worker()
            linear_pars = random_state.multivariate_normal(
                self.a, np.linalg.inv(self.Ainv), size=n_linear_samples_per)

            for j in range(n_linear_samples_per):
                ll[n, j] = _ll

                samples[n, j, 0] = P
                samples[n, j, 1] = e
                samples[n, j, 2] = om
                samples[n, j, 3] = M0
                samples[n, j, 4] = chunk[n, 4] # s, jitter

                for k in range(self.n_linear):
                    samples[n, j, 5 + k] = linear_pars[j, k]

        return (np.array(samples).reshape(n_samples * n_linear_samples_per, -1),
                np.array(ll).reshape(n_samples * n_linear_samples_per))

    cpdef test_likelihood_worker(self, double[::1] chunk_row):
        cdef:
            double P, e, om, M0

            # Transpose of design matrix
            double[:, ::1] M_T = np.zeros((self.n_linear, self.n_times))

        # TODO: need to make sure the chunk is always in this order!
        P = chunk_row[0]
        e = chunk_row[1]
        om = chunk_row[2]
        M0 = chunk_row[3]

        # TODO: audit order of chunk[...]'s and what c_rv_from_elements
        c_rv_from_elements(&self.t[0], &self.M_T[0, 0], self.n_times,
                            P, 1., e, om, M0, self.t0,
                            anomaly_tol, anomaly_maxiter)

        # Note: jitter must be in same units as the data RV's / ivar
        get_ivar(self.ivar, chunk_row[4], self.s_ivar)

        # TODO: this is a continuation of the massive hack introduced above.
        if self.fixed_K_prior == 0:
            self.Lambda[0] = (self.sigma_K0**2 / (1 - e**2)
                              * (P / self.P0)**(-2/3.))

        # compute likelihood, but also generate a, A, etc.
        ll = self.likelihood_worker(1)  # the 1 is "True"

        return ll
