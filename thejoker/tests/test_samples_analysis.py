# Third-party
import astropy.units as u
import numpy as np

# Project
from ..data import RVData
from ..samples import JokerSamples
from ..samples_analysis import (
    MAP_sample, is_P_unimodal, is_P_Kmodal,
    max_phase_gap, phase_coverage, periods_spanned)


def test_MAP_sample():
    samples = JokerSamples()
    samples['ln_likelihood'] = np.array([1., 2, 3.])
    samples['ln_prior'] = np.array([1., 2, 3.])

    sample = MAP_sample(samples)
    assert sample['ln_likelihood'] == 3.

    sample, i = MAP_sample(samples, return_index=True)
    assert i == 2


def test_is_P_unimodal():
    np.random.seed(42)

    samples = JokerSamples()
    samples['P'] = np.random.normal(8, 1e-3, size=32) * u.day

    size = 8
    t = 56000 + np.random.uniform(0, 100, size=size)
    rv = np.random.normal(0, 10, size=size) * u.km/u.s
    rv_err = np.random.uniform(0.1, 0.3, size=size) * u.km/u.s
    data = RVData(t=t, rv=rv, rv_err=rv_err)

    assert is_P_unimodal(samples, data)


def test_is_P_Kmodal():
    np.random.seed(42)

    samples = JokerSamples()
    samples['P'] = np.concatenate((np.random.normal(8, 1e-3, size=32),
                                   np.random.normal(21, 1e-3, size=32))) * u.day
    samples['ln_likelihood'] = np.random.uniform(size=len(samples['P']))
    samples['ln_prior'] = np.random.uniform(size=len(samples['P']))

    size = 8
    t = 56000 + np.random.uniform(0, 100, size=size)
    rv = np.random.normal(0, 10, size=size) * u.km/u.s
    rv_err = np.random.uniform(0.1, 0.3, size=size) * u.km/u.s
    data = RVData(t=t, rv=rv, rv_err=rv_err)

    Kmodal, means, npermode = is_P_Kmodal(samples, data, n_clusters=2)
    assert Kmodal
    assert np.min(np.abs(means - 8.*u.day)) < 0.1*u.day
    assert np.min(np.abs(means - 21.*u.day)) < 0.1*u.day
    assert np.all(npermode == 32)


def test_max_phase_gap():
    samples = JokerSamples()
    samples['P'] = 6.243 * np.ones(8) * u.day

    phase = np.array([0, 0.1, 0.7, 0.8, 0.9])
    t = 56000 + phase * samples['P'][0].value
    rv = np.random.normal(0, 10, size=len(phase)) * u.km/u.s
    rv_err = np.random.uniform(0.1, 0.3, size=len(phase)) * u.km/u.s
    data = RVData(t=t, rv=rv, rv_err=rv_err)

    assert np.isclose(max_phase_gap(samples[0], data), 0.6, atol=1e-5)


def test_phase_coverage():
    samples = JokerSamples()
    samples['P'] = 6.243 * np.ones(8) * u.day

    phase = np.array([0, 0.1, 0.7, 0.8, 0.9])
    t = 56000 + phase * samples['P'][0].value
    rv = np.random.normal(0, 10, size=len(phase)) * u.km/u.s
    rv_err = np.random.uniform(0.1, 0.3, size=len(phase)) * u.km/u.s
    data = RVData(t=t, rv=rv, rv_err=rv_err)

    phasecov = phase_coverage(samples[0], data)
    assert phasecov >= 0.4
    assert phasecov <= 0.6


def test_periods_spanned():
    samples = JokerSamples()
    samples['P'] = 6.243 * np.ones(8) * u.day

    phase = np.linspace(0, 3.3, 16)
    t = 56000 + phase * samples['P'][0].value
    rv = np.random.normal(0, 10, size=len(phase)) * u.km/u.s
    rv_err = np.random.uniform(0.1, 0.3, size=len(phase)) * u.km/u.s
    data = RVData(t=t, rv=rv, rv_err=rv_err)

    span = periods_spanned(samples[0], data)
    assert np.isclose(span, phase[-1], atol=1e-5)
