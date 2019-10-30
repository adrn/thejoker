# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Package
from ..data import RVData


def test_rvdata_init():
    rnd = np.random.RandomState(42)

    # Test valid initialization combos
    t_arr = rnd.uniform(55555., 56012., size=32)
    t_obj = Time(t_arr, format='mjd')

    rv = 100 * np.sin(0.5 * t_arr) * u.km / u.s
    err = rnd.normal(0, 5, size=len(t_arr)) * u.km/u.s
    cov = (np.diag(err.value) * err.unit) ** 2

    # These should succeed:
    RVData(t=t_arr, rv=rv, rv_err=err)
    RVData(t_arr, rv, err)
    RVData(t_obj, rv, err)
    RVData(t_arr, rv, cov)

    # With/without clean:
    for i in range(1, 3):  # skip time, because Time() catches nan values
        inputs = [t_arr, rv, err]
        arr = inputs[i].copy()
        arr[0] = np.nan
        inputs[i] = arr

        data = RVData(*inputs)
        assert len(data) == (len(arr)-1)

        data = RVData(*inputs, clean=True)
        assert len(data) == (len(arr)-1)

        data = RVData(*inputs, clean=False)
        assert len(data) == len(arr)

    # With/without t0
    data = RVData(t_arr, rv, err, t0=False)
    assert data.t0 is None

    data = RVData(t_arr, rv, err, t0=t_obj[3])
    assert np.isclose(data.t0.mjd, t_obj[3].mjd)

    # ------------------------------------------------------------------------
    # Test expected failures:

    # no units on something
    with pytest.raises(TypeError):
        RVData(t_arr, rv.value, err)

    with pytest.raises(TypeError):
        RVData(t_arr, rv, err.value)

    # shapes must be consistent
    with pytest.raises(ValueError):
        RVData(t_obj[:-1], rv, err)

    with pytest.raises(ValueError):
        RVData(t_obj, rv[:-1], err)

    with pytest.raises(ValueError):
        RVData(t_obj, rv, err[:-1])

    with pytest.raises(ValueError):
        RVData(t_obj, rv, cov[:-1])

    bad_cov = np.arange(8).reshape((2, 2, 2)) * (u.km/u.s)**2
    with pytest.raises(ValueError):
        RVData(t_obj, rv, bad_cov)

    # t0 must be a Time instance
    with pytest.raises(TypeError):
        RVData(t_arr, rv, err, t0=t_arr[3])


def test_data_methods():

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
