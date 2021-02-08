"""Tests for data.py and data_helpers.py"""

import warnings

# Third-party
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u
try:
    from erfa import ErfaWarning
except ImportError:  # lts version of Astropy
    ErfaWarning = Warning
import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import fuzzywuzzy  # noqa
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

# Package
from ..data import RVData
from ..data_helpers import guess_time_format, validate_prepare_data
from ..prior import JokerPrior
from ..utils import DEFAULT_RNG


def test_guess_time_format():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ErfaWarning)
        for yr in np.arange(1975, 2040, 5):
            assert guess_time_format(Time(f'{yr}-05-23').jd) == 'jd'
            assert guess_time_format(Time(f'{yr}-05-23').mjd) == 'mjd'

        with pytest.raises(NotImplementedError):
            guess_time_format('asdfasdf')

        for bad_val in np.array([0., 1450., 2500., 5000.]):
            with pytest.raises(ValueError):
                guess_time_format(bad_val)


def get_valid_input(rnd=None, size=32):
    if rnd is None:
        rnd = DEFAULT_RNG(42)

    t_arr = rnd.uniform(55555., 56012., size=size)
    t_obj = Time(t_arr, format='mjd')

    rv = 100 * np.sin(2*np.pi * t_arr / 15.) * u.km / u.s
    err = rnd.uniform(0.1, 0.5, size=len(t_arr)) * u.km/u.s
    cov = (np.diag(err.value) * err.unit) ** 2

    _tbl = Table()
    _tbl['rv'] = rnd.uniform(size=len(rv))
    _tbl['rv'].unit = u.km/u.s
    _tbl['rv_err'] = rnd.uniform(size=len(rv))
    _tbl['rv_err'].unit = u.km/u.s

    raw = {'t_arr': t_arr,
           't_obj': t_obj,
           'rv': rv,
           'err': err,
           'cov': cov}

    return [dict(t=t_arr, rv=rv, rv_err=err),
            (t_arr, rv, err),
            (t_obj, rv, err),
            (t_obj, _tbl['rv'], _tbl['rv_err']),
            (t_arr, rv, cov),
            (t_obj, rv, cov)], raw


def test_rvdata_init():
    rnd = DEFAULT_RNG(42)

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

    # With/without t_ref
    data = RVData(t_arr, rv, err, t_ref=False)
    assert data.t_ref is None

    data = RVData(t_arr, rv, err, t_ref=t_obj[3])
    assert np.isclose(data.t_ref.mjd, t_obj[3].mjd)

    #  deprecated:
    with warnings.catch_warnings(record=True) as warns:
        RVData(t_arr, rv, err, t0=t_obj[3])
        assert len(warns) != 0

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

    # t_ref must be a Time instance
    with pytest.raises(TypeError):
        RVData(t_arr, rv, err, t_ref=t_arr[3])


@pytest.mark.parametrize("inputs",
                         get_valid_input()[0])
def test_data_methods(tmpdir, inputs):

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

    assert isinstance(data1.rv, u.Quantity)
    assert isinstance(data1.rv_err, u.Quantity)

    # check slicing
    data2 = data1[:16]
    assert len(data2) == 16
    assert len(data2.t) == 16
    assert len(data2.rv) == 16
    assert len(data2.rv_err) == 16

    # converting to a timeseries object:
    ts = data1.to_timeseries()
    assert isinstance(ts, TimeSeries)

    filename = str(tmpdir / 'test.hdf5')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        ts.write(filename, serialize_meta=True)
    data2 = RVData.from_timeseries(filename)
    assert u.allclose(data1.t.mjd, data2.t.mjd)
    assert u.allclose(data1.rv, data2.rv)
    assert u.allclose(data1.rv_err, data2.rv_err)
    assert u.allclose(data1.t_ref.mjd, data2.t_ref.mjd)

    #  deprecated:
    with warnings.catch_warnings(record=True) as warns:
        data1.t0
        assert len(warns) != 0

    # get phase from data object
    phase1 = data1.phase(P=15.*u.day)
    assert phase1.min() >= 0
    assert phase1.max() <= 1

    phase2 = data1.phase(P=15.*u.day, t0=Time(58585.24, format='mjd'))
    assert not np.allclose(phase1, phase2)

    # compute inverse variance
    ivar = data1.ivar
    assert ivar.unit == (1 / data1.rv.unit**2)

    cov = data1.cov
    assert cov.shape == (len(data1), len(data1))


def test_guess_from_table():
    """NOTE: this is not an exhaustive set of tests, but at least checks a few
    common cases"""

    for rv_name in ['rv', 'vr', 'radial_velocity']:
        tbl = Table()
        tbl['t'] = np.linspace(56423.234, 59324.342, 16) * u.day
        tbl[rv_name] = np.random.normal(0, 1, len(tbl['t']))
        tbl[f'{rv_name}_err'] = np.random.uniform(0.1, 0.2, len(tbl['t']))
        data = RVData.guess_from_table(tbl, rv_unit=u.km/u.s)
        assert np.allclose(data.t.utc.mjd, tbl['t'])

    if HAS_FUZZY:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            for rv_name in ['VHELIO', 'VHELIO_AVG', 'vr', 'vlos']:
                tbl = Table()
                tbl['t'] = np.linspace(56423.234, 59324.342, 16) * u.day
                tbl[rv_name] = np.random.normal(0, 1, len(tbl['t']))
                tbl[f'{rv_name}_err'] = np.random.uniform(0.1, 0.2,
                                                          len(tbl['t']))
                data = RVData.guess_from_table(tbl, rv_unit=u.km/u.s,
                                               fuzzy=True)
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


def test_multi_data():
    import exoplanet.units as xu
    import pymc3 as pm

    rnd = DEFAULT_RNG(42)

    # Set up mulitple valid data objects:
    _, raw1 = get_valid_input(rnd=rnd)
    data1 = RVData(raw1['t_obj'], raw1['rv'], raw1['err'])

    _, raw2 = get_valid_input(rnd=rnd, size=8)
    data2 = RVData(raw2['t_obj'], raw2['rv'], raw2['err'])

    _, raw3 = get_valid_input(rnd=rnd, size=4)
    data3 = RVData(raw3['t_obj'], raw3['rv'], raw3['err'])

    prior1 = JokerPrior.default(1*u.day, 1*u.year,
                                25*u.km/u.s,
                                sigma_v=100*u.km/u.s)

    # Object should return input:
    multi_data, ids, trend_M = validate_prepare_data(data1,
                                                     prior1.poly_trend,
                                                     prior1.n_offsets)
    assert np.allclose(multi_data.rv.value, data1.rv.value)
    assert np.all(ids == 0)
    assert np.allclose(trend_M[:, 0], 1.)

    # Three valid objects as a list:
    with pm.Model():
        dv1 = xu.with_unit(pm.Normal('dv0_1', 0, 1.),
                           u.km/u.s)
        dv2 = xu.with_unit(pm.Normal('dv0_2', 4, 5.),
                           u.km/u.s)
        prior2 = JokerPrior.default(1*u.day, 1*u.year,
                                    25*u.km/u.s,
                                    sigma_v=100*u.km/u.s,
                                    v0_offsets=[dv1, dv2])

    datas = [data1, data2, data3]
    multi_data, ids, trend_M = validate_prepare_data(datas,
                                                     prior2.poly_trend,
                                                     prior2.n_offsets)
    assert len(np.unique(ids)) == 3
    assert len(multi_data) == sum([len(d) for d in datas])
    assert 0 in ids and 1 in ids and 2 in ids
    assert np.allclose(trend_M[:, 0], 1.)

    # Three valid objects with names:
    datas = {'apogee': data1, 'lamost': data2, 'weave': data3}
    multi_data, ids, trend_M = validate_prepare_data(datas,
                                                     prior2.poly_trend,
                                                     prior2.n_offsets)
    assert len(np.unique(ids)) == 3
    assert len(multi_data) == sum([len(d) for d in datas.values()])
    assert 'apogee' in ids and 'lamost' in ids and 'weave' in ids
    assert np.allclose(trend_M[:, 0], 1.)

    # Check it fails if n_offsets != number of data sources
    with pytest.raises(ValueError):
        validate_prepare_data(datas,
                              prior1.poly_trend,
                              prior1.n_offsets)

    with pytest.raises(ValueError):
        validate_prepare_data(data1,
                              prior2.poly_trend,
                              prior2.n_offsets)

    # Check that this fails if one has a covariance matrix
    data_cov = RVData(raw3['t_obj'], raw3['rv'], raw3['cov'])
    with pytest.raises(NotImplementedError):
        validate_prepare_data({'apogee': data1, 'test': data2,
                               'weave': data_cov},
                              prior2.poly_trend, prior2.n_offsets)

    with pytest.raises(NotImplementedError):
        validate_prepare_data([data1, data2, data_cov],
                              prior2.poly_trend,
                              prior2.n_offsets)
