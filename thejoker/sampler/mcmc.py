# Third-party
import astropy.units as u
import numpy as np
from scipy.stats import beta, norm

# Project
from .likelihood import get_ivar, design_matrix
from .params import JokerParams
from .samples import JokerSamples
from ..data import RVData

__all__ = ['TheJokerMCMCModel']

log_2pi = np.log(2 * np.pi)


class TheJokerMCMCModel:

    def __init__(self, joker_params, data):
        """

        Parameters
        ----------
        data : `~thejoker.data.RVData`
            The radial velocity data.
        """

        # check if a JokerParams instance was passed in to specify the state
        if not isinstance(joker_params, JokerParams):
            raise TypeError("Parameter specification must be a JokerParams "
                            "instance, not a '{0}'".format(type(joker_params)))
        self.params = joker_params

        if not isinstance(data, RVData):
            raise TypeError("Data must be a valid RVData object.")
        self.data = data

        # various cached things:
        self._P_min = self.params.P_min.to(u.day).value
        self._P_max = self.params.P_max.to(u.day).value
        self._rv_unit = self.data.rv.unit
        self._jitter_factor = self._rv_unit.to(self.params._jitter_unit)

        if self.params._fixed_jitter:
            s = self.params.jitter
            self._y_jitter = 2 * np.log(s.to(self._rv_unit).value)

    @classmethod
    def to_mcmc_params(cls, p):
        r"""MCMC internal function.

        Transform from linear orbital parameter values to standard
        variables for MCMC sampling:

        .. math::

            \ln P \\
            \sqrt{K}\,\cos\phi_0, \sqrt{K}\,\sin\phi_0 \\
            \sqrt{e}\,\cos\omega, \sqrt{e}\,\sin\omega \\
            \ln s^2 \\
            v_0,..., v_n

        Parameters
        ----------
        p : iterable
            A packed parameter vector containing the orbital parameters
            and long-term velocity trend parameters.

        """
        P, M0, e, omega, s, K, *v_terms = p
        return np.vstack([np.log(P),
                          np.sqrt(K) * np.cos(M0), np.sqrt(K) * np.sin(M0),
                          np.sqrt(e) * np.cos(omega), np.sqrt(e) * np.sin(omega),
                          2*np.log(s)] + list(v_terms))

    @classmethod
    def from_mcmc_params(cls, p):
        """MCMC internal function.

        Transform from the standard MCMC parameters to the linear
        values of the orbital parameters.

        Parameters
        ----------
        p : iterable
            A packed parameter vector containing the MCMC-transforemd
            versions of the orbital parameters and long-term velocity
            trend parameters.

        """
        (ln_P,
         sqrtK_cos_M0, sqrtK_sin_M0,
         sqrte_cos_omega, sqrte_sin_omega,
         log_s2, *v_terms) = p

        return np.vstack([np.exp(ln_P),
                          np.arctan2(sqrtK_sin_M0, sqrtK_cos_M0) % (2*np.pi),
                          sqrte_cos_omega**2 + sqrte_sin_omega**2,
                          np.arctan2(sqrte_sin_omega, sqrte_cos_omega) % (2*np.pi),
                          np.sqrt(np.exp(log_s2)),
                          (sqrtK_cos_M0**2 + sqrtK_sin_M0**2)] + v_terms)

    def pack_samples(self, samples):
        """Pack a dictionary of samples as Quantity objects into a 2D array.

        Parameters
        ----------
        samples : dict
            Dictionary of `~astropy.units.Quantity` objects for period,
            M0, etc.

        Returns
        -------
        arr : `numpy.ndarray`
            A 2D numpy array with shape `(nsamples, ndim)`.
        """
        if 'jitter' in samples:
            jitter = samples['jitter'].to(self._rv_unit).value
        else:
            jitter = np.zeros_like(samples['P'].value)

        arr = [samples['P'].to(u.day).value,
               samples['M0'].to(u.radian).value,
               np.asarray(samples['e']),
               samples['omega'].to(u.radian).value,
               jitter,
               samples['K'].to(self._rv_unit).value,
               samples['v0'].to(self._rv_unit).value]
        # TODO: assumes only constant velocity offset

        return np.array(arr).T

    def pack_samples_mcmc(self, samples):
        """Pack a dictionary of samples as Quantity objects into a 2D array,
        transformed to the parametrization used by the MCMC functions.

        Parameters
        ----------
        samples : dict
            Dictionary of `~astropy.units.Quantity` objects for period,
            M0, etc.

        Returns
        -------
        arr : `numpy.ndarray`
            A 2D numpy array with shape `(nsamples, ndim)`.
        """
        samples_vec = self.pack_samples(samples, self.params)
        samples_mcmc = self.to_mcmc_params(samples_vec.T)

        if self.params._fixed_jitter:
            samples_mcmc = np.delete(samples_mcmc, 5, axis=0)

        return np.array(samples_mcmc).T

    def unpack_samples(self, samples_arr):
        """Unpack a 2D array of samples into a dictionary of samples as
        Quantity objects.

        Parameters
        ----------
        samples_arr : `numpy.ndarray`
            A 2D numpy array with shape `(nsamples, ndim)` containing the values.

        Returns
        -------
        samples : dict
            Dictionary of `~astropy.units.Quantity` objects for period,
            M0, etc.
        """
        samples = JokerSamples()
        samples['P'] = samples_arr.T[0] * u.day
        samples['M0'] = samples_arr.T[1] * u.radian
        samples['e'] = samples_arr.T[2] * u.one
        samples['omega'] = samples_arr.T[3] * u.radian

        if not self.params._fixed_jitter:
            samples['jitter'] = samples_arr.T[4] * self._rv_unit
            shift = 1
        else:
            samples['jitter'] = np.zeros_like(samples_arr.T[0]) * self._rv_unit
            shift = 0

        samples['K'] = samples_arr.T[4+shift] * self._rv_unit

        # TODO: assumes only constant velocity offset
        samples['v0'] = samples_arr.T[5+shift] * self._rv_unit

        return samples

    def unpack_samples_mcmc(self, samples_arr):
        """Unpack a 2D array of samples transformed to the parametrization used
        by the MCMC functions into a dictionary of samples as Quantity objects
        in the standard parametrization (i.e. period, angles, ...).

        Parameters
        ----------
        samples_arr : `numpy.ndarray`
            A 2D numpy array with shape `(nsamples, ndim)` containing the
            values in the MCMC coordinates.

        Returns
        -------
        samples : `thejoker.JokerSamples`
        """
        samples_arr = self.from_mcmc_params(samples_arr.T).T
        return self.unpack_samples(samples_arr)

    def ln_likelihood(self, p):
        P, M0, ecc, omega, s, K, *v_terms = p

        # a little repeated code here...

        A = design_matrix([P, M0, ecc, omega], self.data, self.params)
        p2 = np.array([K] + v_terms)
        ivar = get_ivar(self.data, s)
        dy = A.dot(p2) - self.data.rv.value

        return 0.5 * (-dy**2 * ivar - log_2pi + np.log(ivar))

    def ln_prior(self, p):
        # TODO: repeated code here and hard-coded priors

        P, M0, ecc, omega, s, K, *v_terms = p

        lnp = 0.

        if ecc < 0 or ecc > 1:
            return -np.inf

        lnp += beta.logpdf(ecc, 0.867, 3.03) # Kipping et al. 2013

        # uniform in ln(P) - we don't need the jacobian because we sample in lnP
        if P < self._P_min or P > self._P_max:
            return -np.inf
        # lnp += -np.log(P)

        if not self.params._fixed_jitter:
            # Gaussian prior in ln(s^2) - don't need Jacobian because we are
            # actually sampling in y = ln(s^2)
            s_scaled = s * self._jitter_factor
            y = 2 * np.log(s_scaled)
            lnp += norm.logpdf(y, loc=self.params.jitter[0],
                               scale=self.params.jitter[1])

        return lnp

    def ln_posterior(self, mcmc_p):
        if self.params._fixed_jitter:
            mcmc_p = list(mcmc_p)
            mcmc_p.insert(5, self._y_jitter) # whoa, major hackage!

        p = self.from_mcmc_params(mcmc_p).reshape(len(mcmc_p))

        lnp = self.ln_prior(p)
        if np.isinf(lnp):
            return lnp

        lnl = self.ln_likelihood(p)
        lnprob = lnp + lnl.sum()

        if np.isnan(lnprob):
            return -np.inf

        return lnprob
