# Third-party
import astropy.units as u
import numpy as np
import pytest

# Project
from ..params import JokerParams


def test_init():
    L = np.diag([1e2, 1e2]) ** 2

    pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day,
                       linear_par_Lambda=L)
    assert pars.jitter.value == 0

    # test invalid input

    with pytest.raises(ValueError):
        JokerParams(P_min=8.*u.day, P_max=8192*u.day)

    with pytest.raises(ValueError):
        pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day,
                           linear_par_Lambda=L,
                           jitter="derp")

    with pytest.raises(ValueError):
        pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day,
                           linear_par_Lambda=L,
                           jitter="derp", jitter_unit="cat")

    with pytest.raises(TypeError):
        pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day,
                           linear_par_Lambda=L,
                           jitter=(0.5, 1.), jitter_unit="cat")

    with pytest.raises(ValueError):
        pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day,
                           linear_par_Lambda=L,
                           jitter=(0.1, 5., 1.))
