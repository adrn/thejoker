# Third-party
import astropy.units as u
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.distributions.dist_math import check_parameters
from pytensor.tensor.random.basic import NormalRV, RandomVariable

from thejoker.units import UNIT_ATTR_NAME

__all__ = ["UniformLog", "FixedCompanionMass"]


class UniformLogRV(RandomVariable):
    name = "uniformlog"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"

    @classmethod
    def rng_fn(cls, rng, a, b, size):
        _fac = np.log(b) - np.log(a)
        uu = rng.uniform(size=size)
        return np.exp(uu * _fac + np.log(a))


uniformlog = UniformLogRV()


class UniformLog(pm.Continuous):
    rv_op = uniformlog

    @classmethod
    def dist(cls, a, b, **kwargs):
        a = pt.as_tensor_variable(a)
        b = pt.as_tensor_variable(b)
        return super().dist([a, b], **kwargs)

    def support_point(rv, size, a, b):
        a, b = pt.broadcast_arrays(a, b)
        return 0.5 * (a + b)

    # TODO: remove this once new pymc version is released
    moment = support_point

    def logp(value, a, b):
        _fac = pt.log(b) - pt.log(a)
        res = -pt.as_tensor_variable(value) - pt.log(_fac)
        return check_parameters(
            res,
            (a > 0) & (a < b),
            msg="a > 0 and a < b",
        )


class FixedCompanionMassRV(NormalRV):
    _print_name = ("FixedCompanionMass", "\\mathcal{N}")


fixedcompanionmass = FixedCompanionMassRV()


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

    rv_op = fixedcompanionmass

    @classmethod
    @u.quantity_input(sigma_K0=u.km / u.s, P0=u.day, max_K=u.km / u.s)
    def dist(
        cls,
        P,
        e,
        sigma_K0,
        P0,
        mu=0.0,
        max_K=500 * u.km / u.s,
        K_unit=None,
        *args,
        **kwargs,
    ):
        if K_unit is not None:
            sigma_K0 = sigma_K0.to_value(K_unit)
        max_K = max_K.to(sigma_K0.unit)

        if hasattr(P, UNIT_ATTR_NAME):
            P0 = P0.to(getattr(P, UNIT_ATTR_NAME))

        sigma = pt.clip(
            sigma_K0.value * (P / P0.value) ** (-1 / 3) / np.sqrt(1 - e**2),
            0.0,
            max_K.value,
        )
        dist = super().dist(mu=mu, sigma=sigma, *args, **kwargs)
        dist._sigma_K0 = sigma_K0
        dist._max_K = max_K
        dist._P0 = P0
        return dist


class Kipping13Long(pm.Beta):
    rv_op = pt.random.beta

    @classmethod
    def dist(cls, *args, **kwargs):
        return super().dist(alpha=1.12, beta=3.09, *args, **kwargs)


class Kipping13Short(pm.Beta):
    rv_op = pt.random.beta

    @classmethod
    def dist(cls, *args, **kwargs):
        return super().dist(alpha=0.697, beta=3.27, *args, **kwargs)


class Kipping13Global(pm.Beta):
    rv_op = pt.random.beta

    @classmethod
    def dist(cls, *args, **kwargs):
        return super().dist(alpha=0.867, beta=3.03, *args, **kwargs)
