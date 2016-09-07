from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy.io import fits
import astropy.time as at
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import six

# Project
from .units import usys

__all__ = ['RVData']

class RVData(object):
    """
    Parameters
    ----------
    t : array_like, `~astropy.time.Time`
        Array of times. Either in BJD or as an Astropy time.
    rv : `~astropy.units.Quantity` [speed]
        Radial velocity measurements.
    ivar : `~astropy.units.Quantity` [1/speed^2]
        Inverse variance.
    """
    @u.quantity_input(rv=u.km/u.s)
    def __init__(self, t, rv, ivar=None, stddev=None, metadata=None, t_offset=None):
        if isinstance(t, at.Time):
            _t = t.tcb.mjd
        else:
            _t = t
        self._t = _t

        self._rv = rv.decompose(usys).value

        if ivar is None and stddev is None:
            self._ivar = 1.

        elif ivar is not None and stddev is not None:
            raise ValueError("You must pass in 'ivar' or 'stddev', not both.")

        elif ivar is not None:
            self._ivar = ivar.decompose(usys).value

        elif stddev is not None:
            self._ivar = 1 / stddev.decompose(usys).value**2

        self.metadata = metadata

        if t_offset is None:
            t_offset = np.median(self._t)
            self._t = self._t - t_offset
        self.t_offset = t_offset

    @property
    def t(self):
        return at.Time(self._t + self.t_offset, scale='tcb', format='mjd')

    @property
    def phase(self, t0, P):
        return ((self.t - t0) / P) % 1.

    @property
    def rv(self):
        return self._rv * usys['length'] / usys['time']

    @property
    def ivar(self):
        return self._ivar * (usys['time'] / usys['length'])**2

    @property
    def stddev(self):
        return 1 / np.sqrt(self.ivar)

    # ---

    def plot(self, ax=None, rv_unit=None, **kwargs):
        """
        """
        if ax is None:
            fig,ax = plt.subplots(1,1)

        if rv_unit is None:
            rv_unit = self.rv.unit

        style = kwargs.copy()
        style.setdefault('linestyle', 'none')
        style.setdefault('alpha', 1.)
        style.setdefault('marker', 'o')
        style.setdefault('color', 'k')
        style.setdefault('ecolor', '#666666')

        ax.errorbar(self.t.value, self.rv.to(rv_unit).value,
                    self.stddev.to(rv_unit).value, **style)

        return ax

    # copy methods
    def __copy__(self):
        return self.__class__(t=self.t.copy(),
                              rv=self.rv.copy(),
                              ivar=self.ivar.copy())

    def copy(self):
        return self.__copy__()

    def __getitem__(self, slc):
        return self.__class__(t=self.t.copy()[slc],
                              rv=self.rv.copy()[slc],
                              ivar=self.ivar.copy()[slc])

    def __len__(self):
        return len(self._t)

    @classmethod
    def from_apogee(cls, path_or_data, apogee_id=None, store_metadata=False):
        """
        Parameters
        ----------
        path_or_data : str, numpy.ndarray
            Either a string path to the location of an APOGEE allVisit
            file, or a selection of rows from the allVisit file.
        apogee_id : str
            The APOGEE ID of the desired target, e.g., 2M03080601+7950502.
        """

        if isinstance(path_or_data, six.string_types):
            if apogee_id is None:
                raise ValueError("If path is supplied, you must also supply an APOGEE_ID.")

            if os.path.splitext(path_or_data)[1].lower() == '.fits':
                _allvisit = fits.getdata(path_or_data, 1)
                data = _allvisit[_allvisit['APOGEE_ID'].astype(str) == apogee_id]

            elif os.path.splitext(path_or_data)[1].lower() in ['.hdf5', '.h5']:
                with h5py.File(path_or_data, 'r') as f:
                    data = f[apogee_id][:]

            else:
                raise ValueError("Unrecognized file type.")

        else:
            data = path_or_data

        rv = np.array(data['VHELIO']) * u.km/u.s
        ivar = 1 / (np.array(data['VRELERR'])**2 * (u.km/u.s)**2)
        t = at.Time(np.array(data['JD']), format='jd', scale='tcb')
        bmjd = t.mjd

        idx = np.isfinite(rv.value) & np.isfinite(t.value) & np.isfinite(ivar.value)
        if store_metadata:
            return cls(bmjd[idx], rv[idx], ivar[idx], metadata=data)
        else:
            return cls(bmjd[idx], rv[idx], ivar[idx])
