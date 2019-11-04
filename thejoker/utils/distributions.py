# Third-party
import numpy as np
import pymc3 as pm
from pymc3.distributions import generate_samples

__all__ = ['OneOver']


class OneOver(pm.Continuous):

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
        super(OneOver, self).__init__(**kwargs)

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
        return 1 / self._fac
