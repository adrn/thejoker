# Third-party
import astropy.time as atime
import astropy.units as u
import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except:
    HAS_MPL = False

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

    # pass both
    with pytest.raises(ValueError):
        RVData(t=t, rv=rv, ivar=ivar, stddev=np.sqrt(1/ivar))

    # not velocity units
    with pytest.raises(u.UnitsError):
        RVData(t=t, rv=rv, ivar=ivar.value*u.km)

    # no error
    data = RVData(t=t, rv=rv)
    assert np.isnan(data.stddev.value).all()

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

    data1._t_bmjd += 1.5
    data1.rv *= 1.5
    data1.ivar *= 1.5
    assert np.all(data2._t_bmjd != data1._t_bmjd)
    assert np.all(data2.rv != data1.rv)
    assert np.all(data2.ivar != data1.ivar)

    # check slicing
    t = atime.Time(t, format='mjd', scale='utc')
    data1 = RVData(t=t, rv=rv, ivar=ivar)
    data2 = data1[:16]
    assert len(data2) == 16
    assert len(data2.t) == 16
    assert len(data2.rv) == 16
    assert len(data2.ivar) == 16

    # check filtering NaN's
    t = np.random.uniform(55555., 56012., size=128)
    rv = 100 * np.sin(0.5*t)
    rv[:16] = np.nan
    rv = rv * u.km/u.s
    ivar = 1 / (np.random.normal(0,5,size=t.size)*u.km/u.s)**2

    data = RVData(t=t, rv=rv, ivar=ivar)
    assert len(data) == (128-16)


@pytest.mark.skipif(not HAS_MPL, reason='matplotlib not installed')
def test_plotting():
    # check that plotting at least succeeds with allowed arguments
    t = np.random.uniform(55555., 56012., size=128)
    rv = 100 * np.sin(0.5*t) * u.km/u.s
    ivar = 1 / (np.random.normal(0,5,size=t.size)*u.km/u.s)**2
    data = RVData(t=t, rv=rv, ivar=ivar)

    data.plot()

    # style
    data.plot(color='r')

    # custom axis
    fig,ax = plt.subplots(1,1)
    data.plot(ax=plt.gca())

    # formatting
    data.plot(rv_unit=u.m/u.s)
    data.plot(rv_unit=u.m/u.s, time_format='jd')
    data.plot(rv_unit=u.m/u.s, time_format=lambda x: x.utc.mjd)

    # try with no errors
    data = RVData(t=t, rv=rv)
    data.plot()
    data.plot(ecolor='r')

    plt.close('all')
