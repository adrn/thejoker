# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np

# Project
from .log import log as logger

__all__ = ['RVData']


class RVData:
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
    t0 : numeric (optional) [day]
        A reference time. Default is to use the minimum time in barycentric MJD
        (days).

    """
    @u.quantity_input(rv=u.km/u.s)
    def __init__(self, t, rv, ivar=None, stddev=None,
                 metadata=None, t0=None):

        # For speed, many of the attributes are saved without units and only
        #   returned with units if asked for.
        if isinstance(t, Time):
            _t_bmjd = t.tcb.mjd
        else:
            _t_bmjd = np.atleast_1d(t)
        self._t_bmjd = _t_bmjd

        self.rv = np.atleast_1d(rv)

        # parse input specification of errors
        self._has_err = True
        if ivar is None and stddev is None:
            self.ivar = np.full_like(self.rv.value, np.nan) * self.rv.unit
            self._has_err = False

        elif ivar is not None and stddev is not None:
            raise ValueError("You must pass in 'ivar' or 'stddev', not both.")

        elif ivar is not None:
            if not hasattr(ivar, 'unit'):
                raise TypeError("ivar must be an Astropy Quantity object")
            elif not ivar.unit.is_equivalent(1/self.rv.unit**2):
                raise u.UnitsError("ivar must have same unit type as RV^-2")

            self.ivar = ivar.to(1/self.rv.unit**2)

        elif stddev is not None:
            if not hasattr(stddev, 'unit'):
                raise TypeError("stddev must be an Astropy Quantity object!")
            elif not stddev.unit.is_equivalent(self.rv.unit):
                raise u.UnitsError("stddev must have same unit type as RV")

            self.ivar = 1 / stddev.to(self.rv.unit)**2
        self.ivar = np.atleast_1d(self.ivar)

        # make sure shapes are consistent
        if self._t_bmjd.shape != self.rv.shape or self.rv.shape != self.ivar.shape:
            raise ValueError("Shape of input time, RV, and errors must be consistent! "
                             "({} vs {} vs {})".format(self._t_bmjd.shape,
                                                       self.rv.shape,
                                                       self.ivar.shape))
        # filter out NAN or INF data points
        idx = np.isfinite(self._t_bmjd) & np.isfinite(self.rv)

        if self._has_err:
            idx &= np.isfinite(self.ivar) & (self.ivar.value > 0)

        if idx.sum() < len(self.rv):
            logger.info("Filtering {} NaN/Inf data points".format(len(self.rv) - idx.sum()))

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

        # if no offset is provided, subtract the minimum time
        if t0 is None:
            t0 = self.t.min()

        if not isinstance(t0, Time):
            raise TypeError('If a reference time t0 is specified, it must '
                            'be an astropy.time.Time object.')

        self.t0 = t0
        self._t0_bmjd = self.t0.tcb.mjd

    @property
    def t(self):
        """
        The times of each observation.

        Returns
        -------
        t : `~astropy.time.Time`
            An Astropy Time object for all times.
        """
        return Time(self._t_bmjd, scale='tcb', format='mjd')

    def phase(self, P, t0=None):
        """
        Convert time to a phase.

        By default, the phase is relative to the internal reference epoch,
        ``t0``, but a new epoch can also be specified to this method.

        Parameters
        ----------
        P : `~astropy.units.Quantity` [time]
            The period.
        t0 : `~astropy.time.Time` (optional)
            Default uses the internal reference epoch. Use this to compute the
            phase relative to some other epoch

        Returns
        -------
        phase : `~numpy.ndarray`
            The dimensionless phase of each observation.

        """
        if t0 is None:
            t0 = self.t0
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

    def plot(self, ax=None, rv_unit=None, time_format='mjd', phase_fold=None,
             relative_to_t0=False, add_labels=True, **kwargs):
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
        phase_fold : bool (optional)
            Plot the phase instead of the time by folding on the specified
            period.
        relative_to_t0 : bool (optional)
            Plot the time relative to the reference epoch, ``t0``.
        add_labels : bool (optional)
            Add labels to the figure.
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
        style.setdefault('elinewidth', 1)

        if callable(time_format):
            t = time_format(self.t)
            t0 = time_format(self.t0)
        else:
            t = getattr(self.t, time_format)
            t0 = getattr(self.t0, time_format)

        if relative_to_t0:
            t = t - t0

        if phase_fold:
            t = (t / phase_fold.to(u.day).value) % 1

        if self._has_err:
            ax.errorbar(t, self.rv.to(rv_unit).value,
                        self.stddev.to(rv_unit).value,
                        **style)
        else:
            style.pop('ecolor')
            style.pop('elinewidth')
            ax.plot(t, self.rv.to(rv_unit).value, **style)

        if add_labels:
            ax.set_xlabel('time [BMJD]')
            ax.set_ylabel('RV [{:latex_inline}]'.format(rv_unit))

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
        return len(self.rv.value)

    def to_hdf5(self, file_or_path):
        """
        Write data to an HDF5 file.

        Parameters
        ----------
        file_or_path : str, `h5py.File`, `h5py.Group`
        """

        import h5py
        if isinstance(file_or_path, str):
            f = h5py.File(file_or_path, 'w')
            close = True

        else:
            f = file_or_path
            close = False

        d = f.create_dataset('mjd', data=self.t.tcb.mjd)
        d.attrs['format'] = 'mjd'
        d.attrs['scale'] = 'tcb'

        d = f.create_dataset('rv', data=self.rv.value)
        d.attrs['unit'] = str(self.rv.unit)

        d = f.create_dataset('rv_err', data=self.stddev.value)
        d.attrs['unit'] = str(self.stddev.unit)

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
        if isinstance(file_or_path, str):
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
