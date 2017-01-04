from __future__ import division, print_function

def get_ivar(data, s2):
    """
    Return a copy of the inverse variance array with jitter included.

    This is safe for zero'd out inverse variances.

    Parameters
    ----------
    data : `~thejoker.data.RVData`
    s2 : numeric
        Jitter squared in the same units as the RV data.

    """
    return data.ivar.value / (1 + s2 * data.ivar.value)
