# Third-party
import astropy.units as u
import pytest

# Project
from ..params import JokerParams


def test_init():

    pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day)
    assert pars.jitter.value == 0

    # test invalid input
    # with pytest.raises(TypeError):
    #     pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day, trend="derp")

    with pytest.raises(ValueError):
        pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day, jitter="derp")

    with pytest.raises(ValueError):
        pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day,
                           jitter="derp", jitter_unit="cat")

    with pytest.raises(TypeError):
        pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day,
                           jitter=(0.5, 1.), jitter_unit="cat")

    with pytest.raises(ValueError):
        pars = JokerParams(P_min=8.*u.day, P_max=8192*u.day,
                           jitter=(0.1, 5., 1.))
