# Third-party
import astropy.units as u

# Package
from ..utils import quantity_to_hdf5

__all__ = ['save_prior_samples']

def save_prior_samples(f, samples, rv_unit, ln_prior_probs=None):
    """Save a dictionary of Astropy Quantity prior samples to an HDF5 file in a
    format expected and used by `thejoker.sampler.TheJoker`. The prior samples
    dictionary must contain keys for:

        - ``P``, period
        - ``M0``, phase at pericenter
        - ``e``, eccentricity
        - ``omega``, argument of periastron
        - ``jitter``, velocity jitter (optional)

    Parameters
    ----------
    f : str, :class:`h5py.File`, :class:`h5py.Group`, :class:`h5py.DataSet`
        A string filename, or an instantiated `h5py` class.
    samples : dict
        A dictionary of prior samples as `~astropy.units.Quantity` objects.
    rv_unit : `~astropy.units.UnitBase`
        The radial velocity data unit.

    Returns
    -------
    units : list
        A list of `~astropy.units.UnitBase` objects specifying the units for
        each column.

    """
    _units = {'P': u.day,
              'M0': u.radian,
              'e': u.one,
              'omega': u.radian,
              'jitter': rv_unit}

    if isinstance(f, str):
        import h5py
        close = True
        f_ = h5py.File(f, 'a')

    else:
        f_ = f
        close = False

    g = f_.create_group('samples')
    units = dict()
    for name in samples.keys():
        quantity_to_hdf5(g, name, samples[name].to(_units[name]))
        units[name] = _units[name]

    if ln_prior_probs is not None:
        f_['ln_prior_probs'] = ln_prior_probs

    if close:
        f_.close()

    return units
