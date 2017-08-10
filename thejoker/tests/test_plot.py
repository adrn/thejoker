# Third-party
import astropy.time as atime
import astropy.units as u
import numpy as np
import pytest
from twobody.celestial import (VelocityTrend1,
                               VelocityTrend2)

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except:
    HAS_MPL = False

# Package
from ..sampler import JokerSamples
from ..plot import plot_rv_curves

@pytest.mark.skipif(not HAS_MPL, reason='matplotlib not installed')
def test_plot_rv_curves():

    pars = dict(P=[30.]*u.day, K=[10.]*u.km/u.s, ecc=[0.11239],
                omega=[0.]*u.radian, phi0=[np.pi/2]*u.radian)

    samples = JokerSamples(VelocityTrend2)
    for k in pars:
        samples[k] = pars[k]

    samples['v0'] = [150] * u.km/u.s
    samples['v1'] = [0.01] * u.km/u.s/u.day

    t_grid = np.random.uniform(56000, 56500, 1024)
    t_grid.sort()

    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    plot_rv_curves(samples, t_grid, ax=ax, trend_t0=56000.)
