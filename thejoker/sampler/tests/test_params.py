from __future__ import division, print_function

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np
import pytest

# Project
from ..params import JokerParams

def test_init():

    pars = JokerParams()
    assert len(pars.trends) == 1
    assert pars.jitter.value == 0

    # test invalid input
    with pytest.raises(TypeError):
        pars = JokerParams(trends=["derp"])

    with pytest.raises(ValueError):
        pars = JokerParams(jitter="derp")

    with pytest.raises(ValueError):
        pars = JokerParams(jitter="derp", jitter_unit="cat")

    with pytest.raises(TypeError):
        pars = JokerParams(jitter=(0.5, 1.), jitter_unit="cat")

    with pytest.raises(ValueError):
        pars = JokerParams(jitter=(0.1, 5., 1.))
