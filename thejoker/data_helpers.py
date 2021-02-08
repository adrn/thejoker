# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np

# Project
from .likelihood_helpers import get_trend_design_matrix


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


def validate_prepare_data(data, poly_trend, n_offsets):
    """Internal function.

    Used to take an input ``RVData`` instance, or a list/dict of ``RVData``
    instances, and produce concatenated time, RV, and error arrays, along
    with a consistent t0.
    """
    from .data import RVData
    if isinstance(data, RVData):  # single instance
        if n_offsets != 0:
            raise ValueError("If sampling over velocity offsets between data "
                             "sources, you must pass in multiple data sources."
                             " To do this, pass in a list of RVData instances "
                             "or a dictionary with RVData instances as values."
                             )

        trend_M = get_trend_design_matrix(data, None, poly_trend)
        return data, np.zeros(len(data), dtype=int), trend_M

    # Turn a list-like into a dict object:
    if not hasattr(data, 'keys'):
        # assume it's iterable:
        try:
            _d = {}
            for i, d in enumerate(data):
                _d[i] = d
        except Exception:
            raise TypeError("Failed to parse input data: data must either be "
                            "an RVData instance, an iterable of RVData "
                            "instances, or a dictionary with RVData instances "
                            "as values. Received: {}".format(type(data)))
        data = _d

    # If we've gotten here, data is dict-like:
    rv_unit = None
    t = []
    rv = []
    err = []
    ids = []
    for k in data.keys():
        d = data[k]

        if not isinstance(d, RVData):
            raise TypeError(f"All data must be specified as RVData instances: "
                            f"Object at key '{k}' is a '{type(d)}' instead.")

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

    # validate number of unique ids vs. number of v0_offsets in prior
    if (len(np.unique(ids)) - 1) != n_offsets:
        raise ValueError("Number of data IDs + 1 must equal the number of "
                         "priors on constant offsets specified (i.e. "
                         "v0_offsets)")

    all_data = RVData(t=Time(t, format='mjd', scale='tcb'),
                      rv=rv, rv_err=err)

    trend_M = get_trend_design_matrix(all_data, ids, poly_trend)

    return all_data, ids, trend_M
