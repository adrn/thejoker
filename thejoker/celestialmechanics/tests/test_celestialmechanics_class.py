# Third-party
from astropy.time import Time
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np
import pytest

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except:
    HAS_MPL = False

# Package
from ..celestialmechanics_class import SimulatedRVOrbit
from ..trends import PolynomialVelocityTrend

pars = dict(P=30.*u.day, K=100.*u.m/u.s, ecc=0.11239,
            omega=0.*u.radian, phi0=0.25524*u.radian)

def test_init_simulatedrvorbit():

    # without trend
    orbit = SimulatedRVOrbit(**pars)

    t = np.random.uniform(55612., 55792, 128)
    t.sort()
    rv = orbit.generate_rv_curve(t)
    rv2 = orbit(t)
    assert quantity_allclose(rv, rv2)

    # get pericenter time - check something?!
    orbit = SimulatedRVOrbit(**pars)
    orbit.t0(Time.now().tcb.mjd).tcb.mjd

    # with trend
    # trend = PolynomialVelocityTrend(coeffs=[100.*u.km/u.s, 1*u.km/u.s/u.day])
    trend = PolynomialVelocityTrend(coeffs=[100.*u.km/u.s, 1E-3*u.km/u.s/u.day])
    orbit1 = SimulatedRVOrbit(trend=trend, **pars)
    orbit2 = SimulatedRVOrbit(v0=100.*u.km/u.s, v1=1E-3*u.km/u.s/u.day, **pars)

    t = np.linspace(0, 100., 128)
    rv1 = orbit1.generate_rv_curve(t)
    rv2 = orbit2.generate_rv_curve(t)
    assert quantity_allclose(rv1, rv2)

@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plotting():
    orbit = SimulatedRVOrbit(**pars)

    mjd = np.linspace(56823.123, 57293.2345, 128) # magic numbers
    t = Time(mjd, format='mjd', scale='utc')

    # plotting
    for _t in [mjd, t]:
        orbit.plot(_t)
        orbit.plot(_t, t_format='jd', t_scale='utc')

    fig,ax = plt.subplots(1,1)
    orbit.plot(t=t, ax=ax)

    fig,ax = plt.subplots(1,1)
    orbit.plot(t=mjd, ax=ax)

    plt.close('all')
