# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np

__all__ = ['guess_time_format']


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
