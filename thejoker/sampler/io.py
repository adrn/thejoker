# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np

__all__ = ['pack_prior_samples', 'save_prior_samples'] # 'quantity_from_hdf5', 'quantity_to_hdf5']

# These units are required by the celestial mechanics code and the order
#   is required for the likelihood code
_name_to_unit = OrderedDict()
_name_to_unit['P'] = u.day
_name_to_unit['phi0'] = u.radian
_name_to_unit['ecc'] = u.one
_name_to_unit['omega'] = u.radian

def pack_prior_samples(samples, rv_unit):
    """
    Pack a dictionary of prior samples as Astropy Quantity
    objects into a single 2D array. The prior samples dictionary
    must contain keys for:

        - ``P``, period
        - ``phi0``, phase at t=0
        - ``ecc``, eccentricity
        - ``omega``, argument of periastron
        - ``jitter``, velocity jitter (optional)

    Parameters
    ----------
    samples : dict
        A dictionary of prior samples as `~astropy.units.Quantity`
        objects.
    rv_unit : `~astropy.units.UnitBase`
        The radial velocity data unit.

    Returns
    -------
    arr_samples : `numpy.ndarray`
        An array of ``n`` prior samples with shape ``(n, 5)``. If
        jitter was not passed in, all jitter values will be
        automatically set to 0.
    units : list
        A list of `~astropy.units.UnitBase` objects specifying the
        units for each column.

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
        jitter = samples['jitter'].to(rv_unit).value
    arrs.append(jitter)
    units.append(rv_unit)

    return np.vstack(arrs).T, units

def save_prior_samples(f, samples, rv_unit):
    """
    Save a dictionary of Astropy Quantity prior samples to
    an HDF5 file in a format expected and used by
    `thejoker.sampler.TheJoker`. The prior samples dictionary
    must contain keys for:

        - ``P``, period
        - ``phi0``, phase at pericenter
        - ``ecc``, eccentricity
        - ``omega``, argument of periastron
        - ``jitter``, velocity jitter (optional)

    Parameters
    ----------
    f : str, :class:`h5py.File`, :class:`h5py.Group`, :class:`h5py.DataSet`
        A string filename, or an instantiated `h5py` class.
    samples : dict
        A dictionary of prior samples as `~astropy.units.Quantity`
        objects.
    rv_unit : `~astropy.units.UnitBase`
        The radial velocity data unit.

    Returns
    -------
    units : list
        A list of `~astropy.units.UnitBase` objects specifying the
        units for each column.

    """

    packed_samples, units = pack_prior_samples(samples, rv_unit)

    if isinstance(f, str):
        import h5py
        with h5py.File(f, 'a') as g:
            g.attrs['units'] = np.array([str(x) for x in units]).astype('|S6')
            g['samples'] = packed_samples

    else:
        f.attrs['units'] = np.array([str(x) for x in units]).astype('|S6')
        f['samples'] = packed_samples

    return units

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
