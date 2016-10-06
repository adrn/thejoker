from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy import log as logger
from astropy.io import fits
import astropy.time as at
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import six

# Project
from .units import default_units

__all__ = ['RVData']

class RVData(object):
    """
    Parameters
    ----------
    t : array_like, `~astropy.time.Time`
        Array of times. Either in BMJD or as an Astropy time.
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

        self._rv = rv.to(default_units['v0']).value

        if ivar is None and stddev is None:
            self._ivar = 1.

        elif ivar is not None and stddev is not None:
            raise ValueError("You must pass in 'ivar' or 'stddev', not both.")

        elif ivar is not None:
            if not hasattr(ivar, 'unit'):
                raise TypeError("ivar must be an Astropy Quantity object!")
            self._ivar = ivar.to(1/default_units['v0']**2).value

        elif stddev is not None:
            if not hasattr(stddev, 'unit'):
                raise TypeError("stddev must be an Astropy Quantity object!")
            self._ivar = 1 / stddev.to(default_units['v0']).value**2

        idx = (np.isfinite(self._t) & np.isfinite(self._rv) & np.isfinite(self._ivar) &
               (self._ivar > 0))
        if idx.sum() < len(self._rv):
            logger.warning("Rejecting {} NaN data points".format(len(self._rv)-idx.sum()))
        self._t = self._t[idx]
        self._rv = self._rv[idx]
        self._ivar = self._ivar[idx]

        # sort on times
        idx = self._t.argsort()
        self._t = self._t[idx]
        self._rv = self._rv[idx]
        self._ivar = self._ivar[idx]

        self.metadata = metadata

        if t_offset is None:
            t_offset = np.median(self._t)
            self._t = self._t - t_offset
        self.t_offset = t_offset

    @u.quantity_input(jitter=u.km/u.s)
    def add_jitter(self, jitter):
        self._ivar = (1 / (self.stddev**2 + jitter**2)).to(1/default_units['v0']**2).value

    @property
    def t(self):
        return at.Time(self._t + self.t_offset, scale='tcb', format='mjd')

    @property
    def phase(self, t0, P):
        return ((self.t - t0) / P) % 1.

    @property
    def rv(self):
        return self._rv * default_units['v0']

    @property
    def ivar(self):
        return self._ivar / default_units['v0']**2

    @property
    def stddev(self):
        return 1 / np.sqrt(self.ivar)

    def get_ivar(self, jitter_squared):
        return self._ivar / (1 + jitter_squared * self._ivar)

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

    def to_hdf5(self, file_or_path):
        """
        Write data to an HDF5 file.

        Parameters
        ----------
        file_or_path : str, `h5py.File`, `h5py.Group`
        """

        if isinstance(file_or_path, six.string_types):
            f = h5py.File(file_or_path, 'w')
            close = True

        else:
            f = file_or_path
            close = False

        d = f.create_dataset('mjd', data=self._t + self.t_offset)
        d.attrs['format'] = 'mjd'
        d.attrs['scale'] = 'tcb'

        d = f.create_dataset('rv', data=self._rv)
        d.attrs['unit'] = str(default_units['v0'])

        d = f.create_dataset('rv_err', data=1/np.sqrt(self._ivar))
        d.attrs['unit'] = str(default_units['v0'])

        if close:
            f.close()

    @classmethod
    def from_hdf5(cls, file_or_path):
        """
        Read data to an HDF5 file.

        Parameters
        ----------
        file_or_path : str, `h5py.File`, `h5py.Group`
        """

        if isinstance(file_or_path, six.string_types):
            f = h5py.File(file_or_path, 'r')
            close = True

        else:
            f = file_or_path
            close = False

        t = f['mjd']
        rv = f['rv'][:] * u.Unit(f['rv'].attrs['unit'])
        stddev = f['rv_err'][:] * u.Unit(f['rv_err'].attrs['unit'])

        if close:
            f.close()

        return cls(t=t, rv=rv, stddev=stddev)

    @classmethod
    def from_apogee(cls, path_or_data, apogee_id=None):
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

                rv = np.array(data['VHELIO']) * u.km/u.s
                ivar = 1 / (np.array(data['VRELERR'])**2 * (u.km/u.s)**2)
                t = at.Time(np.array(data['JD']), format='jd', scale='tcb')
                bmjd = t.mjd

            elif os.path.splitext(path_or_data)[1].lower() in ['.hdf5', '.h5']:
                with h5py.File(path_or_data, 'r') as f:
                    data = f[apogee_id][:]

                rv = np.array(data['rv']) * u.km/u.s
                ivar = 1 / (np.array(data['rv_err'])**2 * (u.km/u.s)**2)
                t = at.Time(np.array(data['MJD']), format='mjd', scale='utc')
                bmjd = t.tcb.mjd

            else:
                raise ValueError("Unrecognized file type.")

        else:
            data = path_or_data

        idx = np.isfinite(rv.value) & np.isfinite(t.value) & np.isfinite(ivar.value)
        return cls(bmjd[idx], rv[idx], ivar[idx])
