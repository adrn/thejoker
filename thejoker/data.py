from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy.io import fits
import astropy.time as at
import astropy.units as u
import h5py
import numpy as np
import six

# Project
from .units import default_units
from .log import log

__all__ = ['RVData']

class RVData(object):
    """
    Time-domain radial velocity measurements for a single target.

    Parameters
    ----------
    t : `~astropy.time.Time`, array_like
        Array of measurement times. Either as an array of BMJD values
        or as an Astropy time object.
    rv : `~astropy.units.Quantity` [speed]
        Radial velocity (RV) measurements.
    stddev : `~astropy.units.Quantity` [speed] (optional)
        Standard deviation for each RV measurement. Specify this or ``ivar``.
    ivar : `~astropy.units.Quantity` [1/speed^2] (optional)
        Inverse variance for each RV measurement. Specify this or ``stddev``.
    metadata : (optional)
        Any metadata associated with the object.
    t_offset : numeric (optional) [day]
        A time offset to apply before processing. Default is to subtract off
        the median time in BMJD days.

    """
    @u.quantity_input(rv=u.km/u.s)
    def __init__(self, t, rv, ivar=None, stddev=None, metadata=None, t_offset=None):

        # For speed, many of the attributes are saved without units and only
        #   returned with units if asked for.
        if isinstance(t, at.Time):
            _t_bmjd = t.tcb.mjd
        else:
            _t_bmjd = np.asarray(t)
        self._t_bmjd = _t_bmjd

        if not hasattr(rv, 'unit') or rv.unit.physical_type != 'speed':
            raise ValueError("Input radial velocities must be passed in as an "
                             "Astropy Quantity with speed (velocity) units.")
        self.rv = rv

        # parse input specification of errors
        if ivar is None and stddev is None:
            self._ivar = 1.

        elif ivar is not None and stddev is not None:
            raise ValueError("You must pass in 'ivar' or 'stddev', not both.")

        elif ivar is not None:
            if not hasattr(ivar, 'unit'):
                raise TypeError("ivar must be an Astropy Quantity object!")
            self.ivar = ivar.to(1/self.rv.unit**2)

        elif stddev is not None:
            if not hasattr(stddev, 'unit'):
                raise TypeError("stddev must be an Astropy Quantity object!")
            self.ivar = 1 / stddev.to(self.rv.unit)**2

        # filter out NAN or INF data points
        idx = (np.isfinite(self._t_bmjd) & np.isfinite(self.rv) &
               np.isfinite(self.ivar) & (self.ivar.value > 0))
        if idx.sum() < len(self._rv):
            log.info("Filtering {} NaN/Inf data points".format(len(self.rv) - idx.sum()))

        self._t_bmjd = self._t_bmjd[idx]
        self.rv = self.rv[idx]
        self.ivar = self.ivar[idx]

        # sort on times
        idx = self._t_bmjd.argsort()
        self._t_bmjd = self._t_bmjd[idx]
        self.rv = self.rv[idx]
        self.ivar = self.ivar[idx]

        # metadata can be anything
        self.metadata = metadata

        # if no offset is provided, subtract the median time
        if t_offset is None:
            t_offset = np.median(self._t_bmjd)
            self._t_bmjd = self._t_bmjd - t_offset
        self.t_offset = t_offset

    @property
    def t(self):
        return at.Time(self._t_bmjd + self.t_offset, scale='tcb', format='mjd')

    @property
    def phase(self, t0, P):
        return ((self.t - t0) / P) % 1.

    @property
    def stddev(self):
        return 1 / np.sqrt(self.ivar)

    # ---

    def plot(self, ax=None, rv_unit=None, **kwargs):
        """
        """
        if ax is None:
            import matplotlib.pyplot as plt
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
