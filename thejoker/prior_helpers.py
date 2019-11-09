# Third-party
import astropy.units as u
import numpy as np
import pymc3 as pm
from pymc3.distributions import generate_samples
import theano.tensor as tt
import exoplanet.units as xu

__all__ = ['UniformLog', 'FixedCompanionMass']


class UniformLog(pm.Continuous):

    def __init__(self, a, b, **kwargs):
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
        return tt.ones_like(value) * -np.log(self._fac)


class FixedCompanionMass(pm.Normal):

    @u.quantity_input(sigma_K0=u.km/u.s, P0=u.day)
    def __init__(self, P, e, sigma_K0, P0, mu=0., K_unit=None, **kwargs):
        self._sigma_K0 = sigma_K0
        self._P0 = P0

        if K_unit is not None:
            self._sigma_K0 = self.sigma_K0.to(K_unit)

        if hasattr(P, xu.UNIT_ATTR_NAME):
            self._P0 = self._P0.to(getattr(P, xu.UNIT_ATTR_NAME))

        sigma_K0 = self._sigma_K0.value
        P0 = self._P0.value

        sigma = sigma_K0 * (P/P0)**(-1/3) / np.sqrt(1-e**2)
        super().__init__(mu=mu, sigma=sigma)
