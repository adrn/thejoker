# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np

__all__ = ['pack_prior_samples', 'save_prior_samples']

# These units and the order are required for the likelihood code
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
        A dictionary of prior samples as `~astropy.units.Quantity` objects.
    rv_unit : `~astropy.units.UnitBase`
        The radial velocity data unit.

    Returns
    -------
    arr_samples : `numpy.ndarray`
        An array of ``n`` prior samples with shape ``(n, 5)``. If jitter was not
        passed in, all jitter values will be automatically set to 0.
    units : list
        A list of `~astropy.units.UnitBase` objects specifying the units for
        each column.

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

def save_prior_samples(f, samples, rv_unit, ln_prior_probs=None):
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

            if ln_prior_probs is not None:
                g['ln_prior_probs'] = ln_prior_probs

    else:
        f.attrs['units'] = np.array([str(x) for x in units]).astype('|S6')
        f['samples'] = packed_samples

        if ln_prior_probs is not None:
            f['ln_prior_probs'] = ln_prior_probs

    return units
