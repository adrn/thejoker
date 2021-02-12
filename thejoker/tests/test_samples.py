# Third-party
from astropy.time import Time
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import h5py
import numpy as np
import pytest

# Project
from ..samples import JokerSamples
from ..samples_helpers import _custom_tbl_dtype_compare


def test_joker_samples(tmpdir):
    N = 100

    # Generate some fake samples
    samples = JokerSamples(dict(P=np.random.random(size=N)*u.day))
    assert 'P' in samples.par_names
    assert len(samples) == N

    # Test slicing with a number
    s2 = samples[:10]
    assert len(s2) == 10

    # Slicing with arrays:
    s3 = samples[np.array([0, 1, 5])]
    assert len(s3) == 3
    idx = np.zeros(len(samples)).astype(bool)
    idx[:3] = True
    s4 = samples[idx]
    assert len(s4) == 3

    # Invalid name
    with pytest.raises(ValueError):
        JokerSamples(dict(derp=[15.]*u.kpc))

    # Length conflicts with previous length
    samples = JokerSamples({'P': np.random.random(size=N)*u.day})
    with pytest.raises(ValueError):
        samples['v0'] = np.random.random(size=10)*u.km/u.s

    # Write to HDF5 file
    samples = JokerSamples()
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian

    fn = str(tmpdir / 'test.hdf5')
    samples.write(fn)
    samples2 = JokerSamples.read(fn)

    for k in samples.par_names:
        assert quantity_allclose(samples[k], samples2[k])

    new_samples = samples[samples['P'].argmin()]
    assert quantity_allclose(new_samples['P'],
                             samples['P'][samples['P'].argmin()])

    # Check that t_ref writes / gets loaded
    samples = JokerSamples(t_ref=Time('J2000'))
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian

    fn = str(tmpdir / 'test2.hdf5')
    samples.write(fn)
    samples.write(fn, overwrite=True)
    samples2 = JokerSamples.read(fn)

    assert samples2.t_ref is not None
    assert np.isclose(samples2.t_ref.mjd, Time('J2000').mjd)

    # Check that scalar samples are supported:
    samples = JokerSamples(t_ref=Time('J2000'))
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian
    new_samples = samples[0]
    assert len(new_samples) == 1

    # Check that polynomial trends work
    samples = JokerSamples(t_ref=Time('J2015.5'),
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

    orb = new_samples.get_orbit()
    orb.radial_velocity(Time('J2015.6'))


def test_append_write(tmpdir):
    N = 64
    samples = JokerSamples(t_ref=Time('J2000'))
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian

    fn = str(tmpdir / 'test-tbl.hdf5')
    samples.write(fn)
    samples.write(fn, append=True)

    fn2 = str(tmpdir / 'test-tbl2.hdf5')
    with h5py.File(fn2, 'w') as f:
        g = f.create_group('2M00')
        samples.write(g)

    samples2 = JokerSamples.read(fn)
    assert np.all(samples2['P'][:N] == samples2['P'][N:])


def test_apply_methods():
    N = 100

    # Test that samples objects reduce properly
    samples = JokerSamples(t_ref=Time('J2000'))
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian

    new_samples = samples.mean()
    assert quantity_allclose(new_samples['P'], np.mean(samples['P']))

    # try just executing others:
    new_samples = samples.median()
    new_samples = samples.std()


@pytest.mark.parametrize("t_ref", [None, Time('J2015.5')])
@pytest.mark.parametrize("poly_trend", [1, 3])
@pytest.mark.parametrize("ext", ['.hdf5', '.fits'])
def test_table(tmp_path, t_ref, poly_trend, ext):
    N = 16
    samples = JokerSamples(t_ref=t_ref, poly_trend=poly_trend)
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
    path = str(d / f"t_{str(t_ref)}_{poly_trend}{ext}")

    samples.write(path)
    samples2 = JokerSamples.read(path)
    assert samples2.poly_trend == samples.poly_trend
    if t_ref is not None:
        assert np.allclose(samples2.t_ref.mjd, samples.t_ref.mjd)

    for k in samples.par_names:
        assert u.allclose(samples2[k], samples[k])


def _get_dtype_compare_cases():
    d1s = []
    d2s = []
    evals = []

    # True
    d1 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'datatype': 'float64'},
          {'name': 'omega', 'unit': 'rad', 'datatype': 'float64'},
          {'name': 'M0', 'unit': 'rad', 'datatype': 'float64'},
          {'name': 's', 'unit': 'm / s', 'datatype': 'float64'}]
    d2 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'unit': '', 'datatype': 'float64'},
          {'name': 'omega', 'unit': 'rad', 'datatype': 'float64'},
          {'name': 'M0', 'unit': 'rad', 'datatype': 'float64'},
          {'name': 's', 'unit': 'm / s', 'datatype': 'float64'}]
    d1s.append(d1)
    d2s.append(d2)
    evals.append(True)

    # True
    d1 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'datatype': 'float64'}]
    d2 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'unit': '', 'datatype': 'float64'}]
    d1s.append(d1)
    d2s.append(d2)
    evals.append(True)

    # True
    d1 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'unit': '', 'datatype': 'float64'}]
    d2 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'unit': '', 'datatype': 'float64'}]
    d1s.append(d1)
    d2s.append(d2)
    evals.append(True)

    # False
    d1 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'unit': 'a', 'datatype': 'float64'}]
    d2 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'unit': '', 'datatype': 'float64'}]
    d1s.append(d1)
    d2s.append(d2)
    evals.append(False)

    # False
    d1 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'unit': 'a', 'datatype': 'float64'}]
    d2 = [{'name': 'P', 'unit': 'd', 'datatype': 'float64'},
          {'name': 'e', 'unit': 'b', 'datatype': 'float64'}]
    d1s.append(d1)
    d2s.append(d2)
    evals.append(False)

    return zip(d1s, d2s, evals)


@pytest.mark.parametrize('d1,d2,equal', _get_dtype_compare_cases())
def test_table_dtype_compare(d1, d2, equal):
    assert _custom_tbl_dtype_compare(d1, d2) == equal
