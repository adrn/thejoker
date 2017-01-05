# Third-party
from astropy.constants import G
import numpy as np

__all__ = ['find_t0', 'a1_sini', 'mf', 'mf_a1_sini_ecc_to_P_K']

def find_t0(phi0, P, epoch):
    """
    This is carefully written to not subtract large numbers, but might
    be incomprehensible.

    Parameters
    ----------
    phi0 : numeric [rad]
    P : numeric [day]
    epoch : numeric [day]
        MJD.
    """
    phi0 = np.arctan2(np.sin(phi0), np.cos(phi0)) # HACK
    epoch_phi = (2 * np.pi * epoch / P) % (2. * np.pi)

    delta_phi = np.inf
    iter = 0
    guess = 0.
    while np.abs(delta_phi) > 1E-15 and iter < 16:
        delta_phi = (2*np.pi*guess/P) % (2*np.pi) - (phi0 - epoch_phi)
        delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi)) # HACK
        guess -= delta_phi / (2*np.pi) * P
        iter += 1

    return epoch + guess

def a1_sini(P, K, ecc):
    return K/(2*np.pi) * (P * np.sqrt(1-ecc**2))

def mf(P, K, ecc):
    mf = P * K**3 / (2*np.pi*G) * (1 - ecc**2)**(3/2.)
    return mf

def mf_a1_sini_ecc_to_P_K(mf, a1_sini, ecc):
    P = 2*np.pi * a1_sini**(3./2) / np.sqrt(G * mf)
    K = 2*np.pi * a1_sini / (P * np.sqrt(1-ecc**2))
    return P, K
