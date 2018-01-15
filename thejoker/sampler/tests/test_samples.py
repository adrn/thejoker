# Third-party
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

    # Trend class
    samples = JokerSamples()
    samples['P'] = np.random.uniform(800, 1000, size=N)*u.day
    samples['M0'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['e'] = np.random.random(size=N)
    samples['omega'] = 2*np.pi*np.random.random(size=N)*u.radian
    samples['v0'] = np.random.random(size=N) * u.km/u.s

    with pytest.raises(ValueError):
        samples['v1'] = np.random.random(size=N)

    new_samples = samples[samples['P'].argmin()]
    assert new_samples.trend_cls == samples.trend_cls
