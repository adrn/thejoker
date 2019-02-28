# Third-party
import astropy.units as u
import numpy as np
from scipy.stats import multivariate_normal
from twobody.units import UnitSystem

# Project
from .likelihood import get_ivar, design_matrix
from .params import JokerParams
from .samples import JokerSamples
from ..data import RVData
from ..stats import beta_logpdf, norm_logpdf

__all__ = ['TheJokerMCMCModel']

log_2pi = np.log(2 * np.pi)


def get_vterms(p, poly_trend):
    return dict([('v'+str(i), p['v'+str(i)])
                 for i in range(poly_trend)])


class TheJokerMCMCModel:

    def __init__(self, joker_params, data, units=None):
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

        if units is None:
            units = UnitSystem(u.day, u.radian, u.Msun, u.au,
                               self.data.rv.unit)
        else:
            if not isinstance(units, UnitSystem):
                raise TypeError('Input units must be as a '
                                'twobody.units.UnitySystem object, not {}'
                                .format(units.__class__.__name__))
        self.units = units

        if self.params._fixed_jitter:
            s = self.params.jitter.to_value(self.units['speed'])
            self._ln_s2 = 2 * np.log(s)
            self._s = s
        else:
            self._ln_s2 = None
            self._s = None

        # Covariance matrix for Gaussian prior on velocity trend terms:
        #   K, v0, v1, v2 ...
        self._V = np.linalg.inv(self.params.linear_par_Vinv)
        self._vterms_norm = multivariate_normal(
            mean=np.zeros(1 + self.params.poly_trend),
            cov=self._V)

        # various cached things:
        self._P_min = self.params.P_min.to_value(self.units['time'])
        self._P_max = self.params.P_max.to_value(self.units['time'])
        self._rv = self.data.rv.value
        self._jitter_factor = self.units['speed'].to(self.params._jitter_unit)

    ###########################################################################
    # Probability expressions
    #
    def ln_likelihood(self, p):
        A = design_matrix([p['P'], p['M0'], p['e'], p['omega']],
                          self.data, self.params)
        vterms = np.ravel(list(get_vterms(p, self.params.poly_trend).values()))
        Kv_terms = np.concatenate((np.atleast_1d(p['K']), vterms))
        ivar = get_ivar(self.data, p['jitter'])
        dy = A.dot(Kv_terms) - self._rv

        return 0.5 * (-dy**2 * ivar - log_2pi + np.log(ivar))

    def ln_prior(self, p):
        # TODO: right now, the priors are hard-coded! These should be
        # customizable by the user

        lnp = 0.

        if not 0 < p['e'] < 1:
            return -np.inf
        lnp += beta_logpdf(p['e'], 0.867, 3.03) # Kipping et al. 2013

        # TODO: not normalized properly because we don't use a truncnorm for K
        if p['K'] < 0:
            return -np.inf

        # uniform in ln(P) - we don't need the jacobian because we sample in lnP
        # TODO: but this is not normalized correctly
        if not self._P_min < p['P'] < self._P_max:
            return -np.inf

        # TODO: priors on M0, omega not normalized properly (need 1/2pi?)

        if not self.params._fixed_jitter:
            # Gaussian prior in ln(s^2) - don't need Jacobian because we are
            # actually sampling in y = ln(s^2)
            s_scaled = p['jitter'] * self._jitter_factor
            y = 2 * np.log(s_scaled)
            lnp += norm_logpdf(y, self.params.jitter[0],
                               self.params.jitter[1])

        # Gaussian priors on K, v0, v1, ...
        # TODO: technicaly, K prior is improper because half-gaussian (K>0)!
        vterms = np.ravel(list(get_vterms(p, self.params.poly_trend).values()))
        Kv_terms = np.concatenate((p['K'], vterms))
        lnp += self._vterms_norm.logpdf(Kv_terms)

        return lnp

    def ln_posterior(self, mcmc_p):
        p = self.unpack_samples(mcmc_p, add_units=False)

        lnp = self.ln_prior(p)
        if not np.isfinite(lnp):
            return -np.inf

        lnl = self.ln_likelihood(p)
        lnprob = lnp + lnl.sum()

        if not np.isfinite(lnprob):
            return -np.inf

        return lnprob

    def __call__(self, mcmc_p):
        return self.ln_posterior(mcmc_p)

    ###########################################################################
    # Parameter transformations
    #

    def _strip_units(self, p):
        """Remove units from the input samples by converting to the standard
        unit system.
        """
        new_p = dict()

        new_p['P'] = p['P'].to_value(self.units['time'])
        new_p['M0'] = p['M0'].to_value(self.units['angle'])
        new_p['e'] = np.asarray(p['e'])
        new_p['omega'] = p['omega'].to_value(self.units['angle'])
        new_p['K'] = p['K'].to_value(self.units['speed'])

        if 'jitter' in p:
            new_p['jitter'] = p['jitter'].to_value(self.units['speed'])

        v_terms = get_vterms(p, self.params.poly_trend)
        for i, k in enumerate(v_terms.keys()):
            _unit = self.units['speed'] / self.units['time']**i
            new_p[k] = p['v'+str(i)].to_value(_unit)

        return new_p

    def _add_units(self, p):
        """Add units to the input samples from the standard unit system.
        """
        new_p = JokerSamples()

        new_p['P'] = p['P'] * self.units['time']
        new_p['M0'] = p['M0'] * self.units['angle']
        new_p['e'] = np.asarray(p['e'])
        new_p['omega'] = p['omega'] * self.units['angle']
        new_p['K'] = p['K'] * self.units['speed']

        if 'jitter' in p:
            new_p['jitter'] = p['jitter'] * self.units['speed']

        v_terms = get_vterms(p, self.params.poly_trend)
        for i, k in enumerate(v_terms.keys()):
            _unit = self.units['speed'] / self.units['time']**i
            new_p[k] = p['v'+str(i)] * _unit

        return new_p

    def pack_samples(self, p, strip_units=True):
        r"""Pack a dictionary of samples (in Keplerian parameters P, e, K, etc.)
        as Quantity objects into a 2D array of MCMC parameters.

        The standard variables for MCMC sampling are:

        .. math::

            \ln P \\
            \sqrt{K}\,\cos(M_0-\omega), \sqrt{K}\,\sin(M_0-\omega) \\
            \sqrt{e}\,\cos\omega, \sqrt{e}\,\sin\omega \\
            \ln s^2 \\
            v_0,..., v_n

        Parameters
        ----------
        arr : iterable
            A packed parameter vector containing the orbital parameters
            and long-term velocity trend parameters.

        """
        if strip_units:
            p = self._strip_units(p)

        terms = [np.log(p['P']),
                 np.sqrt(p['K']) * np.cos(p['M0'] - p['omega']),
                 np.sqrt(p['K']) * np.sin(p['M0'] - p['omega']),
                 np.sqrt(p['e']) * np.cos(p['omega']),
                 np.sqrt(p['e']) * np.sin(p['omega'])]

        if not self.params._fixed_jitter:
            terms.append(2 * np.log(p['jitter']))

        # Now add the velocity terms as linear parameters
        v_terms = get_vterms(p, self.params.poly_trend)
        terms.extend(list(v_terms.values()))

        return np.stack(terms)

    def unpack_samples(self, arr, add_units=True):
        """Unpack a 2D array of samples in standard variables form into a
        dictionary of samples as Quantity objects as Kepler parameters (i.e.
        period, angles, ...).

        Parameters
        ----------
        arr : `numpy.ndarray`
            A 2D numpy array with shape `(nsamples, ndim)` containing the
            sample values.

        Returns
        -------
        samples : dict
            Dictionary of `~astropy.units.Quantity` objects for period,
            M0, etc.
        """
        arr = np.atleast_2d(arr)

        (ln_P,
         sqrtK_cos_M0, sqrtK_sin_M0,
         sqrte_cos_omega, sqrte_sin_omega) = arr.T[:5]

        M0_minus_omega = np.arctan2(sqrtK_sin_M0, sqrtK_cos_M0)
        omega = np.arctan2(sqrte_sin_omega, sqrte_cos_omega)
        M0 = (M0_minus_omega + omega)

        M0 = M0 % (2*np.pi)
        omega = omega % (2*np.pi)
        e = sqrte_cos_omega**2 + sqrte_sin_omega**2
        K = sqrtK_cos_M0**2 + sqrtK_sin_M0**2

        # Load the Keplerian sample values:
        samples = dict()

        samples['P'] = np.exp(ln_P)
        samples['M0'] = M0
        samples['e'] = e * u.one
        samples['omega'] = omega
        samples['K'] = K

        if not self.params._fixed_jitter:
            samples['jitter'] = arr.T[5]
            k = 1
        else:
            samples['jitter'] = np.full(len(arr), self._s)
            k = 0

        for i in range(self.params.poly_trend):
            samples['v' + str(i)] = arr.T[5 + k + i]

        if add_units:
            samples = self._add_units(samples)

        return samples
