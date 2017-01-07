# Third-party
import astropy.units as u
import numpy as np
import pytest

from ..trends import PolynomialVelocityTrend

def test_polynomial():

    # initialization
    with pytest.raises(ValueError):
        PolynomialVelocityTrend()

    with pytest.raises(ValueError):
        PolynomialVelocityTrend(coeffs=[10.*u.km/u.s, 1.*u.km/u.s/u.day],
                                n_terms=4)

    trend = PolynomialVelocityTrend(n_terms=2)
    with pytest.raises(ValueError):
        trend(np.random.random(size=10)*u.day)

    trend = PolynomialVelocityTrend(coeffs=[10.*u.km/u.s, 1.*u.km/u.s/u.day])
    trend(np.random.uniform(15., 23., 128))
    trend(np.random.uniform(15., 23., 128)*u.day)

    # TODO: for now
    with pytest.raises(NotImplementedError):
        PolynomialVelocityTrend(coeffs=[10.*u.km/u.s, 1.*u.km/u.s/u.day],
                                data_mask=lambda x: x)


