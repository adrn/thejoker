from __future__ import division, print_function

def get_ivar(data, s):
    """
    Return a copy of the inverse variance array with jitter included.

    This is safe for zero'd out inverse variances.

    Parameters
    ----------
    data : `~thejoker.data.RVData`
    s : numeric
        Jitter in the same units as the RV data.

    """
    return data.ivar.value / (1 + s**2 * data.ivar.value)
