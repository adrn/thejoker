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
from ..prior import JokerPrior
from ..samples import JokerSamples
from ..plot import plot_rv_curves, plot_phase_fold
from .test_sampler import make_data


@pytest.mark.skipif(not HAS_MPL, reason='matplotlib not installed')
@pytest.mark.parametrize('prior', [
    JokerPrior.default(10*u.day, 20*u.day,
                       25*u.km/u.s, sigma_v=100*u.km/u.s),
    JokerPrior.default(10*u.day, 20*u.day,
                       25*u.km/u.s, poly_trend=2,
                       sigma_v=[100*u.km/u.s, 0.2*u.km/u.s/u.day])
])
def test_plot_rv_curves(prior):

    data, _ = make_data()
    samples = prior.sample(100, generate_linear=True, t_ref=Time('J2000'))

    t_grid = np.random.uniform(56000, 56500, 1024)
    t_grid.sort()

    plot_rv_curves(samples, t_grid)
    plot_rv_curves(samples, data=data)
    plot_rv_curves(samples[0], data=data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    plot_rv_curves(samples, t_grid, ax=ax)


@pytest.mark.skipif(not HAS_MPL, reason='matplotlib not installed')
@pytest.mark.parametrize('prior', [
    JokerPrior.default(10*u.day, 20*u.day,
                       25*u.km/u.s, sigma_v=100*u.km/u.s),
    JokerPrior.default(10*u.day, 20*u.day,
                       25*u.km/u.s, poly_trend=2,
                       sigma_v=[100*u.km/u.s, 0.2*u.km/u.s/u.day])
])
def test_plot_phase_fold(prior):

    data, _ = make_data()
    samples = prior.sample(100, generate_linear=True, t_ref=Time('J2000'))

    plot_phase_fold(samples.median(), data)
    plot_phase_fold(samples[0:1], data)


def test_big_grid_warning():
    data, _ = make_data()

    samples = JokerSamples(t_ref=data.t_ref)
    samples['P'] = [0.001] * u.day
    samples['e'] = [0.3]
    samples['omega'] = [0.5] * u.rad
    samples['M0'] = [0.34] * u.rad
    samples['s'] = [0] * u.km/u.s
    samples['v0'] = [10.] * u.km/u.s
    samples['K'] = [5.] * u.km/u.s

    with pytest.warns(ResourceWarning, match="10,000"):
        plot_rv_curves(samples, data=data, max_t_grid=20000)

    # No warning should be raised:
    plot_rv_curves(samples, data=data, max_t_grid=5000)
