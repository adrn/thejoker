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


def _prepare_multi_data(data):
    """Internal function.

    Used to take an input ``RVData`` instance, or a list/dict of ``RVData``
    instances, and produce concatenated time, RV, and error arrays, along
    with a consistent t0.
    """
    from .data import RVData
    if isinstance(data, RVData):  # single instance
        return data, None

    # Turn a list-like into a dict object:
    if not hasattr(data, 'keys'):
        _d = {}
        for i, d in enumerate(data):
            _d[i] = d
        data = _d

    # If we've gotten here, data is dict-like:
    rv_unit = None
    t = []
    rv = []
    err = []
    ids = []
    for k in data.keys():
        d = data[k]

        if d._has_cov:
            raise NotImplementedError("We currently don't support "
                                      "multi-survey data when a full "
                                      "covariance matrix is specified. "
                                      "Raise an issue in adrn/thejoker if "
                                      "you want this functionality.")

        if rv_unit is None:
            rv_unit = d.rv.unit

        t.append(d.t.tcb.mjd)
        rv.append(d.rv.to_value(rv_unit))
        err.append(d.rv_err.to_value(rv_unit))
        ids.append([k] * len(d))

    t = np.concatenate(t)
    rv = np.concatenate(rv) * rv_unit
    err = np.concatenate(err) * rv_unit
    ids = np.concatenate(ids)

    all_data = RVData(t=Time(t, format='mjd', scale='tcb'),
                      rv=rv, rv_err=err)

    return all_data, ids
