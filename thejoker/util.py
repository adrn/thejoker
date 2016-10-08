from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np

__all__ = ['find_t0', 'quantity_from_hdf5']

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

def quantity_from_hdf5(f, key, n=None):
    """
    Return an Astropy Quantity object from a key in an HDF5 file,
    group, or dataset. This checks to see if the input file/group/dataset
    contains a ``'unit'`` attribute (e.g., in `f.attrs`).

    Parameters
    ----------
    f : :class:`h5py.File`, :class:`h5py.Group`, :class:`h5py.DataSet`
    key : str
        The key name.
    n : int (optional)
        The number of rows to load.

    Returns
    -------
    q : `astropy.units.Quantity`, `numpy.ndarray`
        If a unit attribute exists, this returns a Quantity. Otherwise, it
        returns a numpy array.
    """
    if 'unit' in f[key].attrs and f[key].attrs['unit'] is not None:
        unit = u.Unit(f[key].attrs['unit'])
    else:
        unit = 1.

    if n is not None:
        return f[key][:n] * unit
    else:
        return f[key][:] * unit
