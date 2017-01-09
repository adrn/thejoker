# Third-party
import astropy.units as u
import numpy as np
import pytest

from ..trends import PolynomialVelocityTrend

def test_polynomial():

    # initialization
    trend = PolynomialVelocityTrend(coeffs=[10.*u.km/u.s, 1.*u.km/u.s/u.day, 1.*u.km/u.s/u.day**2])

    with pytest.raises(ValueError):
        PolynomialVelocityTrend()

    with pytest.raises(ValueError):
        PolynomialVelocityTrend(coeffs=[10.*u.km/u.s, 1.*u.km/u.s/u.day],
                                n_terms=4)

    # invalid units
    with pytest.raises(ValueError):
        PolynomialVelocityTrend(coeffs=[10., 1.*u.km/u.s/u.day])

    with pytest.raises(u.UnitsError):
        PolynomialVelocityTrend(coeffs=[10.*u.km/u.s, 1.*u.km/u.s])

    # valid even when n_terms = 0
    trend = PolynomialVelocityTrend(n_terms=0)
    assert trend.n_terms == 0

    # can't evaluate trend with no coeffs
    trend = PolynomialVelocityTrend(n_terms=2)
    with pytest.raises(ValueError):
        trend(np.random.random(size=10)*u.day)

    trend = PolynomialVelocityTrend(coeffs=[10.*u.km/u.s, 1.*u.km/u.s/u.day])
    trend(np.random.uniform(15., 23., 128))
    trend(np.random.uniform(15., 23., 128)*u.day)

    with pytest.raises(u.UnitsError):
        trend(np.random.uniform(15., 23., 128)*u.km)


