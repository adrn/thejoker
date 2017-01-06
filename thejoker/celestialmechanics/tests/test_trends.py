# Third-party
import numpy as np
import pytest

from ..trends import PolynomialVelocityTrend

def test_polynomial():

    # initialization
    with pytest.raises(ValueError):
        PolynomialVelocityTrend()

    with pytest.raises(ValueError):
        PolynomialVelocityTrend(coeffs=[10., 100.], n_terms=4)

    trend = PolynomialVelocityTrend(n_terms=2)
    with pytest.raises(ValueError):
        trend(np.random.random(size=10))

    trend = PolynomialVelocityTrend(coeffs=[10., 100.])
    trend(np.random.uniform(15., 23., 128))

    # TODO: for now
    with pytest.raises(NotImplementedError):
        PolynomialVelocityTrend(coeffs=[10., 100.], data_mask=lambda x: x)
