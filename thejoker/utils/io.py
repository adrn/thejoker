# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np

# Package
from .logging import logger


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


def guess_time_format(time_val):
    """Guess the `astropy.time.Time` format from the input value(s).

    Parameters
    ----------
    time_val : float, array_like
        The value or values to guess the format from.

    Returns
    -------
    format : str
        The `astropy.time.Time` format string, guessed from the value.

    """
    arr = np.array(time_val)

    # Only support float or int times
    if arr.dtype.char not in ['f', 'd', 'i', 'l']:
        raise NotImplementedError("We can only try to guess "
                                  "the time format with a numeric "
                                  "time value. Sorry about that!")

    # HACK: magic number!
    dt_thresh = 50*u.year

    jd_check = np.abs(arr - Time('2010-01-01').jd) * u.day < dt_thresh
    jd_check = np.all(jd_check)

    mjd_check = np.abs(arr - Time('2010-01-01').mjd) * u.day < dt_thresh
    mjd_check = np.all(mjd_check)

    err_msg = ("Unable to guess time format! Initialize with an "
               "explicit astropy.time.Time() object instead.")
    if jd_check and mjd_check:
        raise ValueError(err_msg)

    elif jd_check:
        return 'jd'

    elif mjd_check:
        return 'mjd'

    else:
        raise ValueError(err_msg)
