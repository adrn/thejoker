# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np

__all__ = ['pack_prior_samples', 'save_prior_samples']

# These units are required by the celestial mechanics code and the order
#   is required for the likelihood code
_name_to_unit = OrderedDict()
_name_to_unit['P'] = u.day
_name_to_unit['phi0'] = u.radian
_name_to_unit['ecc'] = u.one
_name_to_unit['omega'] = u.radian

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
            g.attrs['units'] = [str(x) for x in units]
            g['samples'] = packed_samples

    else:
        f.attrs['units'] = [str(x) for x in units]
        f['samples'] = packed_samples
