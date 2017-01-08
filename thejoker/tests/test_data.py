# Third-party
import astropy.time as atime
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Package
from ..data import RVData

def test_rvdata():

    # test various initializations
    t = np.random.uniform(55555., 56012., size=1024)
    rv = 100 * np.sin(0.5*t) * u.km/u.s
    ivar = 1 / (np.random.normal(0,5,size=1024)*u.km/u.s)**2
    RVData(t=t, rv=rv, ivar=ivar)

    t = atime.Time(t, format='mjd', scale='utc')
    RVData(t=t, rv=rv, ivar=ivar)

    with pytest.raises(TypeError):
        RVData(t=t, rv=rv.value, ivar=ivar)

    with pytest.raises(TypeError):
        RVData(t=t, rv=rv, ivar=ivar.value)

    # shapes must be consistent
    with pytest.raises(ValueError):
        RVData(t=t[:-1], rv=rv, ivar=ivar)

    with pytest.raises(ValueError):
        RVData(t=t, rv=rv[:-1], ivar=ivar)

    with pytest.raises(ValueError):
        RVData(t=t, rv=rv, ivar=ivar[:-1])

    # check that copy works
    t = atime.Time(t, format='mjd', scale='utc')
    data1 = RVData(t=t, rv=rv, ivar=ivar)
    data2 = data1.copy()

    data1._t_bmjd *= 1.5
    data1.rv *= 1.5
    data1.ivar *= 1.5
    assert np.all(data2._t_bmjd != data1._t_bmjd)
    assert np.all(data2.rv != data1.rv)
    assert np.all(data2.ivar != data1.ivar)

    # check that plotting at least succeeds (TODO: could be better)
    # data1.plot()
    # data1.plot(color='r')
    # data1.plot(ax=plt.gca())
