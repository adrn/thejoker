import warnings

# Third-party
from astropy.table import Table
from astropy.time import Time
from astropy.utils.decorators import deprecated_renamed_argument
import astropy.units as u
import numpy as np

# Project
from .logging import logger
from .data_helpers import guess_time_format
from .exceptions import TheJokerDeprecationWarning

__all__ = ['RVData']


class RVData:
    """
    Time-domain radial velocity measurements for a single target.

    Parameters
    ----------
    t : `~astropy.time.Time`, array_like
        Array of measurement times. Either as an astropy `~astropy.time.Time`
        object, or as a numpy array of Barycentric MJD (BMJD) values.
    rv : `~astropy.units.Quantity` [speed]
        Radial velocity (RV) measurements.
    rv_err : `~astropy.units.Quantity` [speed] (optional)
        If 1D, assumed to be the standard deviation for each RV measurement. If
        this input is 2-dimensional, this is assumed to be a covariance matrix
        for all data points.
    t_ref : numeric (optional) [day]
        A reference time. Default is to use the minimum time in barycentric MJD
        (days). Set to ``False`` to disable subtracting the reference time.
    clean : bool (optional)
        Filter out any NaN or Inf data points.

    """
    @deprecated_renamed_argument('t0', 't_ref', since='v1.2',
                                 warning_type=TheJokerDeprecationWarning)
    @u.quantity_input(rv=u.km/u.s, rv_err=[u.km/u.s, (u.km/u.s)**2])
    def __init__(self, t, rv, rv_err, t_ref=None, clean=True):

        # For speed, time is saved internally as BMJD:
        if isinstance(t, Time):
            _t_bmjd = t.tcb.mjd
        else:
            _t_bmjd = np.atleast_1d(t)
        self._t_bmjd = _t_bmjd

        self.rv = u.Quantity(np.atleast_1d(rv))

        # Figure out what kind of error is specified
        self.rv_err = u.Quantity(np.atleast_1d(rv_err))

        if self.rv_err.ndim == 1:
            self._has_cov = False
        elif self.rv_err.ndim == 2:
            self._has_cov = True

        if (self.rv_err.shape != (self.rv.size, self.rv.size)
                and self.rv_err.shape != (self.rv.size, )):
            raise ValueError(f"Invalid shape for input RV error "
                             f"{self.rv_err.shape}. Should either be "
                             f"({self.rv.size},) or "
                             f"({self.rv.size}, {self.rv.size})")

        # make sure shapes are consistent
        if self._t_bmjd.shape != self.rv.shape:
            raise ValueError(f"Shape of input times and RVs must be consistent"
                             f" ({self._t_bmjd.shape} vs {self.rv.shape})")

        if clean:
            # filter out NAN or INF data points
            idx = (np.isfinite(self._t_bmjd)
                   & np.isfinite(self.rv))

            if self._has_cov:
                idx &= np.isfinite(self.rv_err).all(axis=0)
            else:
                idx &= np.isfinite(self.rv_err)

            n_filter = len(self.rv) - idx.sum()
            if n_filter > 0:
                logger.info(f"Filtering {n_filter} NaN/Inf data points")

            self._t_bmjd = self._t_bmjd[idx]
            self.rv = self.rv[idx]

            if self._has_cov:
                self.rv_err = self.rv_err[idx]
                self.rv_err = self.rv_err[:, idx]
            else:
                self.rv_err = self.rv_err[idx]

        # sort on times
        idx = self._t_bmjd.argsort()
        self._t_bmjd = self._t_bmjd[idx]
        self.rv = self.rv[idx]
        if self._has_cov:
            self.rv_err = self.rv_err[idx]
            self.rv_err = self.rv_err[:, idx]
        else:
            self.rv_err = self.rv_err[idx]

        if t_ref is False:
            self.t_ref = None
            self._t_ref_bmjd = 0.

        else:
            if t_ref is None:
                t_ref = self.t.min()

            if not isinstance(t_ref, Time):
                raise TypeError('If a reference time t_ref is specified, it '
                                'must be an astropy.time.Time object.')

            self.t_ref = t_ref
            self._t_ref_bmjd = self.t_ref.tcb.mjd

    @property
    def t0(self):
        warnings.warn('The argument and attribute "t0" has been renamed '
                      'and should now be specified / accessed as "t_ref"',
                      TheJokerDeprecationWarning)
        return self.t_ref

    # ------------------------------------------------------------------------
    # Computed or convenience properties

    @property
    def t(self):
        """The times of each observation.

        Returns
        -------
        t : `~astropy.time.Time`
            An Astropy Time object for all times.
        """
        return Time(self._t_bmjd, scale='tcb', format='mjd')

    @property
    def cov(self):
        """Covariance matrix"""
        if self._has_cov:
            return self.rv_err
        else:
            return np.diag(self.rv_err.value**2) * self.rv_err.unit**2

    @property
    def ivar(self):
        """Inverse-variance."""
        if self._has_cov:
            return np.linalg.inv(self.rv_err.value) / self.rv_err.unit
        else:
            return 1 / self.rv_err ** 2

    # ------------------------------------------------------------------------
    # Other initialization methods:

    @classmethod
    @deprecated_renamed_argument('t0', 't_ref', since='v1.2',
                                 warning_type=TheJokerDeprecationWarning)
    def guess_from_table(cls, tbl, time_kwargs=None, rv_unit=None,
                         fuzzy=False, t_ref=None):
        """
        Try to construct an ``RVData`` instance by guessing column names from
        the input table.

        .. note::

            This is an experimental feature! Use at your own risk.

        Parameters
        ----------
        tbl : `~astropy.table.Table`
            The source data table.
        time_kwargs : dict (optional)
            Additional keyword arguments to pass to the `~astropy.time.Time`
            initializer when passing in the inferred time data column. For
            example, if you know the time data are in Julian days, you can pass
            in ``time_kwargs=dict(format='jd')`` to improve the guessing.
        rv_unit : `astropy.units.Unit` (optional)
            If not specified via the relevant table column, this specifies the
            velocity units.
        fuzzy : bool (optional)
            Use fuzzy string matching to guess data column names. This requires
            the ``fuzzywuzzy`` package.
        """
        tbl = Table(tbl)
        lwr_to_col = {x.lower(): x for x in tbl.colnames}
        lwr_cols = [x.lower() for x in tbl.colnames]

        # --------------------------------------------------------------------
        # First handle time data:

        time_data = None
        if time_kwargs is None:
            time_kwargs = dict()

        # First check for any of the valid astropy Time format names:
        # FUTURETODO: right now we only support jd and mjd (and b-preceding)
        for fmt in ['jd', 'mjd']:
            if fmt in lwr_cols:
                time_kwargs['format'] = time_kwargs.get('format', fmt)
                time_data = tbl[lwr_to_col[fmt]]
                break

            elif f'b{fmt}' in lwr_cols:
                time_kwargs['format'] = time_kwargs.get('format', fmt)
                time_kwargs['scale'] = time_kwargs.get('scale', 'tcb')
                time_data = tbl[lwr_to_col[f'b{fmt}']]
                _scale_specified = True
                break

        time_info_msg = ("Assuming time scale is '{}' because it was not "
                         "specified. To change this, pass in: "
                         "time_kwargs=dict(scale='...') with whatever time "
                         "scale your data are in.")
        _fmt_specified = 'format' in time_kwargs
        _scale_specified = 'scale' in time_kwargs

        # check colnames for "t" or "time"
        for name in ['t', 'time']:
            if name in lwr_cols:
                time_data = tbl[lwr_to_col[name]]
                time_kwargs['format'] = time_kwargs.get(
                    'format', guess_time_format(tbl[lwr_to_col[name]]))
                if not _fmt_specified:
                    logger.info("Guessed time format: '{}'. If this is "
                                "incorrect, try passing in "
                                "time_kwargs=dict(format='...') with the "
                                "correct format, and open an issue at "
                                "https://github.com/adrn/thejoker/issues"
                                .format(time_kwargs['format']))
                break

        if not _scale_specified:
            logger.info(time_info_msg.format(time_kwargs.get('scale', 'utc')))

        if time_data is None:
            raise RuntimeWarning("Failed to parse time data and format from "
                                 "input table. Instead, try using the "
                                 "initializer directly, and specify the time "
                                 "as an astropy.time.Time instance.")

        time = Time(time_data, **time_kwargs)

        # --------------------------------------------------------------------
        # Now deal with RV data:

        # FUTURETODO: could make this customizable...
        _valid_rv_names = ['rv', 'vr', 'radial_velocity',
                           'vhelio', 'vrad', 'vlos']

        if fuzzy:
            try:
                from fuzzywuzzy import process
            except ImportError:
                raise ImportError("Fuzzy column name matching requires "
                                  "`fuzzywuzzy`. Install with pip install "
                                  "fuzzywuzzy.")

            # FUTURETODO: could make this customizable too...
            score_thresh = 90

            matches = []
            scores = []
            for name in _valid_rv_names:
                match, score = process.extractOne(name, lwr_cols)
                matches.append(match)
                scores.append(score)
            scores = np.array(scores)
            matches = np.array(matches)

            # error if the best match is below threshold
            if scores.max() < score_thresh:
                raise RuntimeError("Failed to parse radial velocity data from "
                                   "input table: No column names looked "
                                   "good with fuzzy name matching.")

            # check for multiple bests:
            if np.sum(scores == scores.max()) > 1:
                raise RuntimeError("Failed to parse radial velocity data from "
                                   "input table: Multiple column names looked "
                                   "good with fuzzy matching {}."
                                   .format(matches[scores == scores.max()]))

            best_rv_name = matches[scores.argmax()]

        else:
            for name in _valid_rv_names:
                if name in lwr_cols:
                    best_rv_name = name
                    break
            else:
                raise RuntimeError("Failed to parse radial velocity data from "
                                   "input table: no matches to input names: "
                                   f"{_valid_rv_names}. Use fuzzy=True or "
                                   "use the initializer directly.")

        rv_data = u.Quantity(tbl[lwr_to_col[best_rv_name]])

        # FUTURETODO: allow customizing?
        _valid_err_names = [f'{best_rv_name}err', f'{best_rv_name}_err',
                            f'{best_rv_name}_e', f'e_{best_rv_name}']
        for err_name in _valid_err_names:
            if err_name in lwr_cols:
                err_data = u.Quantity(tbl[lwr_to_col[err_name]])
                break
        else:
            raise RuntimeError("Failed to parse radial velocity error data "
                               "from input table: no matches to input names: "
                               f"{_valid_err_names}. Try using the "
                               "initializer directly.")

        if rv_unit is not None:
            if rv_data.unit is u.one:
                rv_data = rv_data * rv_unit

            if err_data is not None and err_data.unit is u.one:
                err_data = err_data * rv_unit

        return cls(time, rv_data, err_data, t_ref=t_ref)

    # ------------------------------------------------------------------------
    # To other classes

    def to_timeseries(self):
        """
        Convert this object into an `astropy.timeseries.TimeSeries` instance.
        """
        from astropy.timeseries import TimeSeries

        ts = TimeSeries(time=self.t, data={'rv': self.rv,
                                           'rv_err': self.rv_err})
        ts.meta['t_ref'] = self.t_ref
        return ts

    @classmethod
    def from_timeseries(cls, f, path=None):
        from astropy.timeseries import TimeSeries
        ts = TimeSeries.read(f, path=path)
        t_ref = ts.meta.get('t_ref', None)
        return cls(t=ts['time'],
                   rv=ts['rv'],
                   rv_err=ts['rv_err'],
                   t_ref=t_ref)

    # ------------------------------------------------------------------------
    # Other methods

    def phase(self, P, t0=None):
        """
        Convert time to a phase.

        By default, the phase is relative to the internal reference epoch,
        ``t_ref``, but a new epoch can also be specified to this method.

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
            t0 = self.t_ref
        return ((self.t - t0) / P) % 1.

    @deprecated_renamed_argument('relative_to_t0', 'relative_to_t_ref',
                                 since='v1.2',
                                 warning_type=TheJokerDeprecationWarning)
    def plot(self, ax=None, rv_unit=None, time_format='mjd', phase_fold=None,
             relative_to_t_ref=False, add_labels=True, color_by=None,
             **kwargs):
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
        phase_fold : quantity_like (optional)
            Plot the phase instead of the time by folding on a period value
            passed in to this argument as an Astropy `~astropy.units.Quantity`.
        relative_to_t_ref : bool (optional)
            Plot the time relative to the reference epoch, ``t_ref``.
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
        style.setdefault('elinewidth', 1)

        if style.get('color', 'k') is not None:
            style.setdefault('color', 'k')
            style.setdefault('ecolor', '#666666')

        if callable(time_format):
            t = time_format(self.t)
            t0 = time_format(self.t_ref)
        else:
            t = getattr(self.t, time_format)
            t0 = getattr(self.t_ref, time_format)

        if relative_to_t_ref:
            t = t - t0

        if phase_fold:
            t = (t / phase_fold.to(u.day).value) % 1

        if self._has_cov:
            # FIXME: this is a bit of a hack
            diag_var = np.diag(self.rv_err.value)
            err = np.sqrt(diag_var) * self.rv_err.unit ** 0.5
        else:
            err = self.rv_err

        ax.errorbar(t, self.rv.to(rv_unit).value,
                    err.to(rv_unit).value, **style)

        if add_labels:
            ax.set_xlabel('time [BMJD]')
            ax.set_ylabel('RV [{:latex_inline}]'.format(rv_unit))

        return ax

    def __copy__(self):
        return self.__class__(t=self.t.copy(),
                              rv=self.rv.copy(),
                              rv_err=self.rv_err.copy())

    def copy(self):
        return self.__copy__()

    def __getitem__(self, slc):
        if self._has_cov:
            return self.__class__(t=self.t.copy()[slc],
                                  rv=self.rv.copy()[slc],
                                  rv_err=self.rv_err.copy()[slc][:, slc])
        else:
            return self.__class__(t=self.t.copy()[slc],
                                  rv=self.rv.copy()[slc],
                                  rv_err=self.rv_err.copy()[slc])

    def __len__(self):
        return len(self.rv.value)

    def __repr__(self):
        return f"<RVData: {len(self)} epochs>"
