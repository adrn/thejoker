# Third-party
import astropy.units as u
import numpy as np
import pymc3 as pm
from pymc3.distributions import generate_samples
import aesara_theano_fallback.tensor as tt
import exoplanet.units as xu

__all__ = ['UniformLog', 'FixedCompanionMass']


class UniformLog(pm.Continuous):

    def __init__(self, a, b, **kwargs):
        """A distribution over a value, x, that is uniform in log(x) over the
        domain :math:`(a, b)`.
        """

        self.a = float(a)
        self.b = float(b)
        assert (self.a > 0) and (self.b > 0)
        self._fac = np.log(self.b) - np.log(self.a)

        shape = kwargs.get("shape", None)
        if shape is None:
            testval = 0.5 * (self.a + self.b)
        else:
            testval = 0.5 * (self.a + self.b) + np.zeros(shape)
        kwargs["testval"] = kwargs.pop("testval", testval)
        super(UniformLog, self).__init__(**kwargs)

    def _random(self, size=None):
        uu = np.random.uniform(size=size)
        return np.exp(uu * self._fac + np.log(self.a))

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        return -tt.as_tensor_variable(value) - np.log(self._fac)


class FixedCompanionMass(pm.Normal):
    r"""
    A distribution over velocity semi-amplitude, :math:`K`, that, at
    fixed primary mass, is a fixed Normal distribution in companion mass. This
    has the form:

    .. math::

        p(K) \propto \mathcal{N}(K \,|\, \mu_K, \sigma_K)
        \sigma_K = \sigma_{K, 0} \, \left(\frac{P}{P_0}\right)^{-1/3} \,
            \left(1 - e^2\right)^{-1}

    where :math:`P` and :math:`e` are period and eccentricity, and
    ``sigma_K0`` and ``P0`` are parameters of this distribution that must
    be specified.
    """

    @u.quantity_input(sigma_K0=u.km/u.s, P0=u.day, max_K=u.km/u.s)
    def __init__(self, P, e, sigma_K0, P0, mu=0., max_K=500*u.km/u.s,
                 K_unit=None, **kwargs):
        self._sigma_K0 = sigma_K0
        self._P0 = P0
        self._max_K = max_K

        if K_unit is not None:
            self._sigma_K0 = self.sigma_K0.to(K_unit)
        self._max_K = self._max_K.to(self._sigma_K0.unit)

        if hasattr(P, xu.UNIT_ATTR_NAME):
            self._P0 = self._P0.to(getattr(P, xu.UNIT_ATTR_NAME))

        sigma_K0 = self._sigma_K0.value
        P0 = self._P0.value

        sigma = tt.min([self._max_K.value,
                        sigma_K0 * (P/P0)**(-1/3) / np.sqrt(1-e**2)])
        super().__init__(mu=mu, sigma=sigma)


class Kipping13Long(pm.Beta):

    def __init__(self):
        r"""
        The inferred long-period eccentricity distribution from Kipping (2013).
        """
        super().__init__(1.12, 3.09)


class Kipping13Short(pm.Beta):

    def __init__(self):
        r"""
        The inferred short-period eccentricity distribution from Kipping (2013).
        """
        super().__init__(0.697, 3.27)


class Kipping13Global(pm.Beta):

    def __init__(self):
        r"""
        The inferred global eccentricity distribution from Kipping (2013).
        """
        super().__init__(0.867, 3.03)
