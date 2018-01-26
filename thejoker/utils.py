# Third-party
import astropy.units as u

# Package
from .log import log as logger


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

    if f[key].ndim == 0:
        if n is not None:
            logger.warning("Dataset '{}' is a scalar.".format(key))

        return f[key].value * unit

    else:
        if n is not None:
            return f[key][:n] * unit
        else:
            return f[key][:] * unit


def quantity_to_hdf5(f, name, q):
    """
    Turn an Astropy Quantity object into something we can write out to
    an HDF5 file.

    Parameters
    ----------
    f : :class:`h5py.File`, :class:`h5py.Group`, :class:`h5py.DataSet`
    key : str
        The name.
    q : float, `astropy.units.Quantity`
        The quantity.

    """

    if hasattr(q, 'unit'):
        f[name] = q.value
        f[name].attrs['unit'] = str(q.unit)

    else:
        f[name] = q
        f[name].attrs['unit'] = ""
