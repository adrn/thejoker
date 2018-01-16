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
from ..sampler import JokerSamples
from ..plot import plot_rv_curves


@pytest.mark.skipif(not HAS_MPL, reason='matplotlib not installed')
def test_plot_rv_curves():

    pars = dict(P=[30.]*u.day, K=[10.]*u.km/u.s, e=[0.11239],
                omega=[0.]*u.radian, M0=[np.pi/2]*u.radian,
                v0=[150]*u.km/u.s)

    samples = JokerSamples()
    for k in pars:
        samples[k] = pars[k]

    t_grid = np.random.uniform(56000, 56500, 1024)
    t_grid.sort()

    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    plot_rv_curves(samples, t_grid, ax=ax)
