from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

def find_t0(phi0, P, epoch):
    """
    This is carefully written to not subtract large numbers, but might
    be incomprehensible.
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
