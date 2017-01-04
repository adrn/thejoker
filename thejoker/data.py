from __future__ import division, print_function

# Third-party
from astropy.extern import six
import astropy.time as at
import astropy.units as u
import numpy as np

# Project
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
    metadata : any (optional)
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
        self._has_err = True
        if ivar is None and stddev is None:
            self._ivar = 1.
            self._has_err = False

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
        """
        The times of each observation.

        Returns
        -------
        t : `~astropy.time.Time`
            An Astropy Time object for all times.
        """
        return at.Time(self._t_bmjd + self.t_offset, scale='tcb', format='mjd')

    def phase(self, t0, P):
        """
        Convert time to a phase relative to the input epoch ``t0``
        and period ``P``.

        Parameters
        ----------
        t0 : `~astropy.time.Time`
            The reference epoch.
        P : `~astropy.units.Quantity` [time]
            The period.

        Returns
        -------
        phase : `~numpy.ndarray`
            The dimensionless phase of each observation.

        """
        return ((self.t - t0) / P) % 1.

    @property
    def stddev(self):
        """
        The Gaussian error for each radial velocity measurement.

        Returns
        -------
        stddev : `~astropy.units.Quantity` [speed]
            An Astropy Quantity with velocity (speed) units.
        """
        return 1 / np.sqrt(self.ivar)

    # ---

    def plot(self, ax=None, rv_unit=None, time_format='mjd', **kwargs):
        """
        Plot the data points.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes` (optional)
            The matplotlib axes object to draw on (default is to grab
            the current axes object using `~matplotlib.pyplot.gca`).
        rv_unit : `~astropy.units.UnitBase` (optional)
            Display the radial velocities with a different unit
            (default uses whatever unit was passed on creation).
        time_format : str, callable (optional)
            The time format to use for the x-axis. This can either be
            a string, in which case it is assumed to be an attribute of
            the `~astropy.time.Time` object, or it can be a callable (e.g.,
            function) that does more complex things (for example:
            ``time_format=lambda t: t.datetime.day``).
        **kwargs
            All other keyword arguments are passed to the
            `~matplotlib.pyplot.errorbar` (if errors were provided) or
            `~matplotlib.pyplot.plot` (if no errors provided) call.

        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        if rv_unit is None:
            rv_unit = self.rv.unit

        # some default stylings
        style = kwargs.copy()
        style.setdefault('linestyle', 'none')
        style.setdefault('alpha', 1.)
        style.setdefault('marker', 'o')
        style.setdefault('color', 'k')
        style.setdefault('ecolor', '#666666')

        if callable(time_format):
            t = time_format(self.t)
        else:
            t = getattr(self.t, time_format)

        if self._has_err:
            ax.errorbar(t, self.rv.to(rv_unit).value,
                        self.stddev.to(rv_unit).value,
                        **style)
        else:
            style.pop('ecolor')
            ax.plot(t, self.rv.to(rv_unit).value, **style)

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
        return len(self.t)

    def to_hdf5(self, file_or_path):
        """
        Write data to an HDF5 file.

        Parameters
        ----------
        file_or_path : str, `h5py.File`, `h5py.Group`
        """

        import h5py
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

        import h5py
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

