# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np

__all__ = ['pack_prior_samples', 'save_prior_samples'] #, 'quantity_from_hdf5', 'quantity_to_hdf5']

# These units are required by the celestial mechanics code and the order
#   is required for the likelihood code
_name_to_unit = OrderedDict()
_name_to_unit['P'] = u.day
_name_to_unit['phi0'] = u.radian
_name_to_unit['ecc'] = u.one
_name_to_unit['omega'] = u.radian

# TODO: make data 2nd argument. Could just pass in units instead of full data object. In
# pack/unpack full_samples, also need joker_params because of trend

def pack_prior_samples(data, samples):
    """
    Pack a dictionary of prior samples as Astropy Quantity
    objects into a single 2D array.
    """

    arrs = []
    units = []
    for name, unit in _name_to_unit.items():
        if unit == u.one:
            arr = np.asarray(samples[name])
        else:
            arr = samples[name].to(unit).value
        arrs.append(arr)
        units.append(unit)

    if 'jitter' not in samples:
        jitter = np.zeros_like(arrs[0])

    else:
        jitter = samples['jitter'].to(data.rv.unit).value
    arrs.append(jitter)
    units.append(data.rv.unit)

    return np.vstack(arrs).T, units

def save_prior_samples(f, data, samples):
    """
    TODO:
    """

    packed_samples, units = pack_prior_samples(data, samples)

    if isinstance(f, str):
        import h5py
        with h5py.File(f, 'a') as g:
            g.attrs['units'] = np.array([str(x) for x in units]).astype('|S6')
            g['samples'] = packed_samples

    else:
        f.attrs['units'] = np.array([str(x) for x in units]).astype('|S6')
        f['samples'] = packed_samples

def unpack_full_samples():
    pass

# def quantity_from_hdf5(f, key, n=None):
#     """
#     Return an Astropy Quantity object from a key in an HDF5 file,
#     group, or dataset. This checks to see if the input file/group/dataset
#     contains a ``'unit'`` attribute (e.g., in `f.attrs`).

#     Parameters
#     ----------
#     f : :class:`h5py.File`, :class:`h5py.Group`, :class:`h5py.DataSet`
#     key : str
#         The key name.
#     n : int (optional)
#         The number of rows to load.

#     Returns
#     -------
#     q : `astropy.units.Quantity`, `numpy.ndarray`
#         If a unit attribute exists, this returns a Quantity. Otherwise, it
#         returns a numpy array.
#     """
#     if 'unit' in f[key].attrs and f[key].attrs['unit'] is not None:
#         unit = u.Unit(f[key].attrs['unit'])
#     else:
#         unit = 1.

#     if n is not None:
#         return f[key][:n] * unit
#     else:
#         return f[key][:] * unit

# def quantity_to_hdf5(f, name, q):
#     """
#     Turn an Astropy Quantity object into something we can write out to
#     an HDF5 file.

#     Parameters
#     ----------
#     f : :class:`h5py.File`, :class:`h5py.Group`, :class:`h5py.DataSet`
#     key : str
#         The name.
#     q : float, `astropy.units.Quantity`
#         The quantity.

#     """

#     if hasattr(q, 'unit'):
#         f[name] = q.value
#         f[name].attrs['unit'] = str(q.unit)

#     else:
#         f[name] = q
#         f[name].attrs['unit'] = ""
