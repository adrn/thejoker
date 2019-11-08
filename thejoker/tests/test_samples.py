# Third-party
from astropy.time import Time
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import h5py
import numpy as np
import pytest

# Project
from ..samples import JokerSamples


def test_joker_samples(tmpdir):
    N = 100

    # Generate some fake samples
    samples = JokerSamples(P=np.random.random(size=N)*u.day)
    assert 'P' in samples
    assert len(samples) == N

    # Test slicing with a number
    s2 = samples[:10]
    assert len(s2) == 10

    # Invalid name
    with pytest.raises(ValueError):
        samples = JokerSamples(derp=15.)

    # Length conflicts with previous length
    samples = JokerSamples(P=np.random.random(size=N)*u.day)
    with pytest.raises(ValueError):
        samples['v0'] = np.random.random(size=10)*u.km/u.s

    # Write to HDF5 file
    samples = JokerSamples()
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian

    fn = str(tmpdir / 'test.hdf5')
    with h5py.File(fn, 'w') as f:
        samples.to_hdf5(f)

    with h5py.File(fn, 'r') as f:
        samples2 = JokerSamples.from_hdf5(f)

    for k in samples.keys():
        assert quantity_allclose(samples[k], samples2[k])

    new_samples = samples[samples['P'].argmin()]
    assert quantity_allclose(new_samples['P'],
                             samples['P'][samples['P'].argmin()])

    # Check that t0 writes / gets loaded
    samples = JokerSamples(t0=Time('J2000'))
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian

    fn = str(tmpdir / 'test.hdf5')
    with h5py.File(fn, 'w') as f:
        samples.to_hdf5(f)

    with h5py.File(fn, 'r') as f:
        samples2 = JokerSamples.from_hdf5(f)

    assert samples2.t0 is not None
    assert np.isclose(samples2.t0.mjd, Time('J2000').mjd)

    # Check that scalar samples are supported:
    samples = JokerSamples(t0=Time('J2000'))
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian
    new_samples = samples[0]

    assert new_samples.shape == ()
    assert new_samples.size == 1

    # Check that polynomial trends work
    samples = JokerSamples(t0=Time('J2015.5'),
                           poly_trend=3)
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['K'] = 100 * np.random.normal(size=N) * u.km/u.s
    samples['v0'] = np.random.uniform(0, 10, size=N) * u.km/u.s
    samples['v1'] = np.random.uniform(0, 1, size=N) * u.km/u.s/u.day
    samples['v2'] = np.random.uniform(0, 1e-2, size=N) * u.km/u.s/u.day**2
    new_samples = samples[0]

    orb = samples.get_orbit(0)
    orb.radial_velocity(Time('J2015.6'))

    orb = new_samples.get_orbit(0)
    orb.radial_velocity(Time('J2015.6'))


def test_apply_methods():
    N = 100

    # Test that samples objects reduce properly
    samples = JokerSamples(t0=Time('J2000'))
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian

    new_samples = samples.mean()
    assert quantity_allclose(new_samples['P'], np.mean(samples['P']))

    # try just executing others:
    new_samples = samples.median()
    new_samples = samples.std()


@pytest.mark.parametrize("t0,poly_trend",
                         [(None, 1),
                          (None, 3),
                          (Time('J2015.5'), 1),
                          (Time('J2015.5'), 3)])
def test_table(tmp_path, t0, poly_trend):
    N = 16
    samples = JokerSamples(t0=t0, poly_trend=poly_trend)
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['K'] = 100 * np.random.normal(size=N) * u.km/u.s
    samples['v0'] = np.random.uniform(0, 10, size=N) * u.km/u.s

    if poly_trend > 1:
        samples['v1'] = np.random.uniform(0, 1, size=N) * u.km/u.s/u.day
        samples['v2'] = np.random.uniform(0, 1e-2, size=N) * u.km/u.s/u.day**2

    d = tmp_path / "table"
    d.mkdir()
    path = str(d / "t_{t0}_{pt}.fits".format(t0=str(t0), pt=poly_trend))

    tbl = samples.to_table()
    tbl.write(path)

    samples2 = JokerSamples.from_table(path)
    assert samples2.poly_trend == samples.poly_trend
    if t0 is not None:
        assert np.allclose(samples2.t0.mjd, samples.t0.mjd)

    for k in samples.keys():
        assert u.allclose(samples2[k], samples[k])