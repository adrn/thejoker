# Third-party
from astropy.time import Time
import astropy.units as u
from astropy.tests.helper import quantity_allclose
import numpy as np

# Project
from ..utils import get_t0, get_phi0, a1_sini, mf, mf_a1_sini_ecc_to_P_K

def test_get_t0():
    ref_mjd = 7025*8.
    t0 = get_t0(0., 8.*u.day, ref_mjd)
    assert np.allclose(ref_mjd, t0.tcb.mjd)

    t0 = get_t0(0.*u.degree, 8.*u.day, ref_mjd)
    assert np.allclose(ref_mjd, t0.tcb.mjd)


def test_get_phi0():

    P = 4.183*u.day
    mjd = 57243.25235
    t0 = Time(mjd, scale='tcb', format='mjd')

    phi0_1 = get_phi0(mjd, P)
    assert phi0_1.unit == u.radian

    phi0_2 = get_phi0(t0, P)
    assert phi0_2.unit == u.radian

    assert np.allclose(phi0_1.value, phi0_2.value)

def test_roundtrip_asini_mf():

    for i in range(32):
        P = np.random.uniform(1., 100.) * u.day
        K = np.random.uniform(1., 100.) * u.km/u.s
        ecc = np.random.uniform()

        asini = a1_sini(P, K, ecc)
        massf = mf(P, K, ecc)
        P2,K2 = mf_a1_sini_ecc_to_P_K(massf, asini, ecc)

        assert quantity_allclose(P, P2)
        assert quantity_allclose(K, K2)
