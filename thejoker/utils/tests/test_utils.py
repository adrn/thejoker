# Third-party
from astropy.time import Time
import numpy as np
import pytest


from ..utils import guess_time_format


def test_guess_time_format():
    for yr in np.arange(1975, 2040, 5):
        assert guess_time_format(Time(f'{yr}-05-23').jd) == 'jd'
        assert guess_time_format(Time(f'{yr}-05-23').mjd) == 'mjd'

    with pytest.raises(NotImplementedError):
        guess_time_format('asdfasdf')

    for bad_val in np.array([0., 1450., 2500., 5000.]):
        with pytest.raises(ValueError):
            guess_time_format(bad_val)
