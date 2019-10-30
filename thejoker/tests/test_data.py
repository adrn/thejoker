# Third-party
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u
import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import fuzzywuzzy
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

# Package
from ..data import RVData


def get_valid_input(rnd=None):
    if rnd is None:
        rnd = np.random.RandomState(42)

    t_arr = rnd.uniform(55555., 56012., size=32)
    t_obj = Time(t_arr, format='mjd')

    rv = 100 * np.sin(2*np.pi * t_arr / 15.) * u.km / u.s
    err = rnd.uniform(0.1, 0.5, size=len(t_arr)) * u.km/u.s
    cov = (np.diag(err.value) * err.unit) ** 2

    raw = {'t_arr': t_arr,
           't_obj': t_obj,
           'rv': rv,
           'err': err,
           'cov': cov}

    return [dict(t=t_arr, rv=rv, rv_err=err),
            (t_arr, rv, err),
            (t_obj, rv, err),
            (t_arr, rv, cov),
            (t_obj, rv, cov)], raw


def test_rvdata_init():
    rnd = np.random.RandomState(42)

    # Test valid initialization combos
    # These should succeed:
    valid_inputs, raw = get_valid_input(rnd)
    for x in valid_inputs:
        if isinstance(x, tuple):
            RVData(*x)
        else:
            RVData(**x)

    t_arr = raw['t_arr']
    t_obj = raw['t_obj']
    rv = raw['rv']
    err = raw['err']
    cov = raw['cov']

    # With/without clean:
    for i in range(1, 3):  # skip time, because Time() catches nan values
        inputs = list(valid_inputs[1])
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


@pytest.mark.parametrize("inputs",
                         get_valid_input()[0])
def test_data_methods(inputs):

    # check that copy works
    if isinstance(inputs, tuple):
        data1 = RVData(*inputs)
    else:
        data1 = RVData(**inputs)
    data2 = data1.copy()

    data1._t_bmjd += 1.5
    data1.rv *= 1.5
    assert np.all(data2._t_bmjd != data1._t_bmjd)
    assert np.all(data2.rv != data1.rv)

    # check slicing
    data2 = data1[:16]
    assert len(data2) == 16
    assert len(data2.t) == 16
    assert len(data2.rv) == 16
    assert len(data2.rv_err) == 16

    # converting to a timeseries object:
    ts = data1.to_timeseries()
    assert isinstance(ts, TimeSeries)

    # get phase from data object
    phase1 = data1.phase(P=15.*u.day)
    assert phase1.min() >= 0
    assert phase1.max() <= 1

    phase2 = data1.phase(P=15.*u.day, t0=Time(58585.24, format='mjd'))
    assert not np.allclose(phase1, phase2)


def test_guess_from_table():
    """TODO: this is not an exhaustive set of tests, but at least checks a few
    common cases"""

    for rv_name in ['rv', 'vr', 'radial_velocity']:
        tbl = Table()
        tbl['t'] = np.linspace(56423.234, 59324.342, 16) * u.day
        tbl[rv_name] = np.random.normal(0, 1, len(tbl['t']))
        tbl[f'{rv_name}_err'] = np.random.uniform(0.1, 0.2, len(tbl['t']))
        data = RVData.guess_from_table(tbl, rv_unit=u.km/u.s)
        assert np.allclose(data.t.utc.mjd, tbl['t'])

    if HAS_FUZZY:
        for rv_name in ['VHELIO', 'VHELIO_AVG', 'vr', 'vlos']:
            tbl = Table()
            tbl['t'] = np.linspace(56423.234, 59324.342, 16) * u.day
            tbl[rv_name] = np.random.normal(0, 1, len(tbl['t']))
            tbl[f'{rv_name}_err'] = np.random.uniform(0.1, 0.2, len(tbl['t']))
            data = RVData.guess_from_table(tbl, rv_unit=u.km/u.s, fuzzy=True)
            assert np.allclose(data.t.utc.mjd, tbl['t'])

    tbl = Table()
    tbl['t'] = np.linspace(2456423.234, 2459324.342, 16) * u.day
    tbl['rv'] = np.random.normal(0, 1, len(tbl['t'])) * u.km/u.s
    tbl['rv_err'] = np.random.uniform(0.1, 0.2, len(tbl['t'])) * u.km/u.s
    data = RVData.guess_from_table(tbl)
    assert np.allclose(data.t.utc.jd, tbl['t'])

    data = RVData.guess_from_table(tbl, time_kwargs=dict(scale='tcb'))
    assert np.allclose(data.t.tcb.jd, tbl['t'])


@pytest.mark.skipif(not HAS_MPL, reason='matplotlib not installed')
@pytest.mark.parametrize("inputs",
                         get_valid_input()[0])
def test_plotting(inputs):

    # check that copy works
    if isinstance(inputs, tuple):
        data = RVData(*inputs)
    else:
        data = RVData(**inputs)

    data.plot()

    # style
    data.plot(color='r')

    # custom axis
    fig, ax = plt.subplots(1, 1)
    data.plot(ax=plt.gca())

    # formatting
    data.plot(rv_unit=u.m/u.s)
    data.plot(rv_unit=u.m/u.s, time_format='jd')
    data.plot(rv_unit=u.m/u.s, time_format=lambda x: x.utc.mjd)
    data.plot(ecolor='r')

    plt.close('all')

