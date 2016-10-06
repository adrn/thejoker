# Third-party
import astropy.units as u
import numpy as np

# Project
from ..celestialmechanics import *

def test_everything():
    np.random.seed(42)
    tt0 = 0. # d
    tt1 = 400. # d
    for n in range(256):
        P = np.exp(np.log(10.) + np.log(400./10.) * np.random.uniform()) # d
        a = (P / 300.) ** (2. / 3.) # AU
        e = np.random.uniform()
        omega = 2. * np.pi * np.random.uniform() # rad
        time0, time = tt0 + (tt1 - tt0) * np.random.uniform(size=2) # d
        phi0 = 2*np.pi*time0/P

        sini = 1.0
        K = (2*np.pi*a*sini / (P * np.sqrt(1-e**2)) * u.au / u.day).to(u.m/u.s).value

        print("testing", P, K, e, omega, time, time0)

        big = 65536.0 # MAGIC
        dt = P / big # d ; MAGIC
        dMdt = 2. * np.pi / P # rad / d
        threetimes = [time, time - 0.5 * dt, time + 0.5 * dt]
        M, M1, M2 = ((t - time0) * dMdt for t in threetimes)
        E, E1, E2 = (eccentric_anomaly_from_mean_anomaly(MM, e) for MM in [M, M1, M2])
        dEdM = d_eccentric_anomaly_d_mean_anomaly(E, e)
        dEdM2 = (E2 - E1) / (M2 - M1)
        if np.abs(dEdM - dEdM2) > (1. / big):
            print("dEdM", dEdM, dEdM2, dEdM - dEdM2)
            assert False

        f, f1, f2 = (true_anomaly_from_eccentric_anomaly(EE, e) for EE in [E, E1, E2])
        dfdE = d_true_anomaly_d_eccentric_anomaly(E, f, e)
        dfdE2 = (f2 - f1) / (E2 - E1)
        if np.abs(dfdE - dfdE2) > (1. / big):
            print("dfdE", dfdE, dfdE2, dfdE - dfdE2)
            assert False

        Z, Z1, Z2 = Z_from_elements(threetimes, P, K, e, omega, time0) # AU
        rv = rv_from_elements(time, P, K, e, omega, phi0, 0.) # m/s
        rv2 = (Z2 - Z1) / dt # au/day
        rv2 = (rv2 * u.au/u.day).to(u.m/u.s).value

        assert np.allclose(rv, rv2, atol=1E-6) # TODO: this precision is not so good...
