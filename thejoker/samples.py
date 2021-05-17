# Standard library
from collections import OrderedDict
import copy
import os
import warnings

# Third-party
import astropy.units as u
from astropy.time import Time
from astropy.table import Table, QTable, Row, meta, serialize
from astropy.utils.decorators import deprecated_renamed_argument
import numpy as np
from twobody import KeplerOrbit, PolynomialRVTrend

# Project
from thejoker.src.fast_likelihood import (_nonlinear_packed_order,
                                          _nonlinear_internal_units)
from .prior_helpers import (validate_poly_trend, validate_n_offsets,
                            get_nonlinear_equiv_units,
                            get_linear_equiv_units,
                            get_v0_offsets_equiv_units)
from .samples_helpers import write_table_hdf5
from .exceptions import TheJokerDeprecationWarning
from .likelihood_helpers import ln_normal

__all__ = ['JokerSamples']


class JokerSamples:
    _hdf5_path = 'samples'

    @deprecated_renamed_argument('t0', 't_ref', since='v1.2',
                                 warning_type=TheJokerDeprecationWarning)
    def __init__(self, samples=None, t_ref=None, n_offsets=None,
                 poly_trend=None, **kwargs):
        """
        A dictionary-like object for storing prior or posterior samples from
        The Joker, with some extra functionality.

        Parameters
        ----------
        samples : `~astropy.table.QTable`, table-like (optional)
            The samples data as an Astropy table object, or something
            convertable to an Astropy table (e.g., a dictionary with
            `~astropy.units.Quantity` object values). This is optional because
            the samples data can be added later by setting keys on the
            resulting instance.
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. See
            the docstring for `thejoker.JokerPrior` for more details.
        t_ref : `astropy.time.Time`, numeric (optional)
            The reference time for the orbital parameters.
        **kwargs
            Additional keyword arguments are stored internally as metadata.
        """

        if poly_trend is None:
            poly_trend = 1

        if n_offsets is None:
            n_offsets = 0

        if isinstance(samples, Table) or isinstance(samples, Row):
            t_ref = samples.meta.pop('t_ref', t_ref)
            poly_trend = samples.meta.pop('poly_trend', poly_trend)
            n_offsets = samples.meta.pop('n_offsets', n_offsets)
            kwargs.update(samples.meta)

            if isinstance(samples, Table):
                self.tbl = QTable()
            elif isinstance(samples, Row):
                self.tbl = samples

        else:
            self.tbl = QTable()

        # Validate input poly_trend / n_offsets:
        poly_trend, _ = validate_poly_trend(poly_trend)
        n_offsets, _ = validate_n_offsets(n_offsets)

        valid_units = {
            **get_nonlinear_equiv_units(),
            **get_linear_equiv_units(poly_trend),
            **get_v0_offsets_equiv_units(n_offsets)
        }

        # log-prior and log-likelihood values are also valid:
        valid_units['ln_prior'] = u.one
        valid_units['ln_likelihood'] = u.one
        valid_units['ln_posterior'] = u.one
        self._valid_units = valid_units

        self.tbl.meta['t_ref'] = t_ref
        self.tbl.meta['poly_trend'] = poly_trend
        self.tbl.meta['n_offsets'] = n_offsets
        for k, v in kwargs.items():
            self.tbl.meta[k] = v

        # Doing this here validates the input:
        if samples is not None:
            _tbl = QTable(samples)
            for colname in _tbl.colnames:
                self[colname] = _tbl[colname]

        # used for speed-ups below
        self._cache = dict()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.__class__(samples=self.tbl[key])

        elif isinstance(key, str) and key in self.par_names:
            return self.tbl[key]

        else:
            return self.__class__(samples=self.tbl[key])

    def __setitem__(self, key, val):
        if key not in self._valid_units:
            raise ValueError(f"Invalid parameter name '{key}'. Must be one "
                             "of: {0}".format(list(self._valid_units.keys())))

        if not hasattr(val, 'unit'):
            val = val * u.one  # eccentricity

        expected_unit = self._valid_units[key]
        if not val.unit.is_equivalent(expected_unit):
            raise u.UnitsError(f"Units of '{key}' must be convertable to "
                               f"{expected_unit}")

        self.tbl[key] = val

    @property
    def t_ref(self):
        return self.tbl.meta['t_ref']

    @property
    def t0(self):
        warnings.warn('The argument and attribute "t0" has been renamed '
                      'and should now be specified / accessed as "t_ref"',
                      TheJokerDeprecationWarning)
        return self.t_ref

    @u.quantity_input(phase=u.rad)
    def get_time_with_phase(self, phase=0*u.rad, t_ref=None):
        """
        Use the phase at reference time to convert to a time with given phase.

        Parameters
        ----------
        phase : quantity_like [angle] (optional)
            Get the time at which the phase is equal to this value.
        t_ref : `~astropy.time.Time` (optional)
            The reference time.

        """
        if t_ref is None:
            if self.t_ref is None:
                raise ValueError("This samples object has no reference time "
                                 "t_ref, so you must pass in the reference "
                                 "time via t_ref")
            else:
                t_ref = self.t_ref

        elif self.t_ref is not None:
            raise ValueError("You passed in a reference time t_ref, but this "
                             "samples object already has a reference time.")

        dt = (self['P'] * self['M0'] / (2*np.pi)).to(
            u.day, u.dimensionless_angles())
        t0 = t_ref + dt

        return t0 + (self['P'] * phase / (2*np.pi)).to(
            u.day, u.dimensionless_angles())

    # TODO: make a property, after deprecation cycle, to replace .t0
    def get_t0(self, t_ref=None):
        return self.get_time_with_phase(phase=0*u.rad, t_ref=t_ref)

    @property
    def poly_trend(self):
        return self.tbl.meta['poly_trend']

    @property
    def n_offsets(self):
        return self.tbl.meta['n_offsets']

    @property
    def par_names(self):
        return self.tbl.colnames

    def __len__(self):
        if isinstance(self.tbl, Table):
            return len(self.tbl)
        else:
            return 1

    def __repr__(self):
        return (f'<JokerSamples [{", ".join(self.par_names)}] '
                f'({len(self)} samples)>')

    def __str__(self):
        return self.__repr__()

    @property
    def isscalar(self):
        if isinstance(self.tbl, Row):
            return True
        else:
            return False

    ##########################################################################
    # Interaction with TwoBody

    def get_orbit(self, index=None, **kwargs):
        """
        Get a `twobody.KeplerOrbit` object for the samples at the specified
        index.

        Parameters
        ----------
        index : int (optional)
            The index of the samples to turn into a `twobody.KeplerOrbit`
            instance. If the samples object is scalar, no index is necessary.
        **kwargs
            Other keyword arguments are passed to the `twobody.KeplerOrbit`
            initializer. For example, you can specify the inclination by
            passing ``i=...`, or  longitude of the ascending node by passing
            ``Omega=...``.

        Returns
        -------
        orbit : `twobody.KeplerOrbit`
            The samples converted to an orbit object. The barycenter position
            and distance are set to arbitrary values.
        """
        if 'orbit' not in self._cache:
            self._cache['orbit'] = KeplerOrbit(P=1*u.yr, e=0., omega=0*u.deg,
                                               Omega=0*u.deg, i=90*u.deg,
                                               a=1*u.au, t0=self.t_ref)

        # all of this to avoid the __init__ of KeplerOrbit / KeplerElements
        orbit = copy.copy(self._cache['orbit'])

        P = self['P']
        e = self['e']
        K = self['K']
        omega = self['omega']
        M0 = self['M0']
        a = kwargs.pop('a', P * K / (2*np.pi) * np.sqrt(1 - e**2))

        names = list(get_linear_equiv_units(self.poly_trend).keys())
        trend_coeffs = [self[x] for x in names[1:]]  # skip K

        if index is None or self.isscalar:
            if len(self) > 1:
                raise ValueError("You must specify an index when the number "
                                 f"of samples is >1 (here, it's {len(self)})")

        else:
            P = P[index]
            e = e[index]
            a = a[index]
            omega = omega[index]
            M0 = M0[index]
            trend_coeffs = [x[index] for x in trend_coeffs]

        orbit.elements.t0 = self.t_ref
        orbit.elements._P = P
        orbit.elements._e = e * u.dimensionless_unscaled
        orbit.elements._a = a
        orbit.elements._omega = omega
        orbit.elements._M0 = M0
        orbit.elements._Omega = kwargs.pop('Omega', 0*u.deg)
        orbit.elements._i = kwargs.pop('i', 90*u.deg)
        orbit._vtrend = PolynomialRVTrend(trend_coeffs, t0=self.t_ref)
        orbit._barycenter = kwargs.pop('barycenter', None)

        if kwargs:
            raise ValueError("Unrecognized arguments {0}"
                             .format(', '.join(list(kwargs.keys()))))

        return orbit

    @property
    def orbits(self):
        """
        A generator that successively returns `twobody.KeplerOrbit` objects
        for each sample. See docstring for `thejoker.JokerSamples.get_orbit` for
        more information.

        """
        for i in range(len(self)):
            yield self.get_orbit(i)

    # Numpy reduce function
    def _apply(self, func):
        cls = self.__class__

        new_samples = dict()
        for k in self.tbl.colnames:
            new_samples[k] = np.atleast_1d(func(self[k]))

        return cls(samples=new_samples, **self.tbl.meta)

    def mean(self):
        """Return a new scalar object by taking the mean across all samples"""
        return self._apply(np.mean)

    def median(self):
        """
        Return a new scalar object by taking the median in period, and
        returning the values for that sample
        """
        med_val = np.percentile(self['P'], 0.5, interpolation='nearest')
        if hasattr(med_val, 'unit'):
            med_val = med_val.value

        median_i, = np.where(self['P'].value == med_val)
        return self[median_i]

    def std(self):
        """Return a new scalar object by taking the standard deviation across
        all samples"""
        return self._apply(np.std)

    def wrap_K(self):
        """
        Change negative K values to positive K values and wrap omega to adjust
        """
        mask = self.tbl['K'] < 0
        if np.any(mask):
            self.tbl['K'][mask] = np.abs(self.tbl['K'][mask])
            self.tbl['omega'][mask] = self.tbl['omega'][mask] + np.pi * u.rad
            self.tbl['omega'][mask] = self.tbl['omega'][mask] % (2*np.pi*u.rad)

        return self

    # Packing and unpacking
    def pack(self, units=None, names=None, nonlinear_only=True):
        """
        Pack the sample data into a single numpy array (i.e. strip the units and
        return those separately).

        Parameters
        ----------
        units : `dict` (optional)
            If specified, this controls the units that the samples are converted
            to before packing into a single array.
        names : `list` (optional)
            The order of names to pack into.
        nonlinear_only : bool (optional)
            Only pack the data for the nonlinear parameters into the returned
            array.

        Returns
        -------
        packed_samples : `numpy.ndarray`
            The samples data packed into a single numpy array with no units.
        units : `~collections.OrderedDict`
            The units of the packed sample data.

        """
        if units is None:
            units = dict()
        out_units = OrderedDict()
        [units.setdefault(k, v) for k, v in _nonlinear_internal_units.items()]

        if names is None:
            if nonlinear_only:
                names = _nonlinear_packed_order
            else:
                names = self.par_names

        arrs = []
        for name in names:
            unit = units.get(name, self.tbl[name].unit)
            arrs.append(self.tbl[name].to_value(unit))
            out_units[name] = unit

        return np.stack(arrs, axis=1), out_units

    @classmethod
    def unpack(cls, packed_samples, units, **kwargs):
        """
        Unpack the array of packed (prior) samples and return a
        `~thejoker.JokerSamples` instance.

        Parameters
        ----------
        packed_samples : array_like
        units : `~collections.OrderedDict`, dict_like
            The units of each column in the packed samples array. The order of
            columns in this dictionary determines the assumed order of the
            columns in the packed samples array.
        **kwargs
            Additional keyword arguments are passed through to the initializer,
            so this supports any arguments accepted by the initializer (e.g.,
            ``t_ref``, ``n_offsets``, ``poly_trend``)

        Returns
        -------
        samples : `~thejoker.JokerSamples`
            The unpacked samples.

        """
        packed_samples = np.array(packed_samples)
        nsamples, npars = packed_samples.shape

        samples = cls(**kwargs)
        for i, k in enumerate(list(units.keys())[:npars]):
            unit = units[k]
            samples[k] = packed_samples[:, i] * unit
        return samples

    def write(self, output, overwrite=False, append=False):
        """
        Write the samples data to a file.

        Currently, we only support writing to / reading from HDF5 files, so the
        filename must end in a .hdf5 or .h5 extension.

        Parameters
        ----------
        output : str, `h5py.File`, `h5py.Group`
            The output filename or ``h5py`` group.
        overwrite : bool (optional)
            Overwrite the existing file.
        append : bool (optional)
            Append the samples to an existing table in the specified filename.
        """
        if isinstance(output, str):
            try:
                ext = os.path.splitext(output)[1]
            except Exception:
                raise ValueError("Invalid file name to save samples to: "
                                 f"{output}")
            if ext not in ['.hdf5', '.h5', '.fits']:
                raise NotImplementedError("We currently only support writing "
                                          "to HDF5 files, with extension .hdf5 "
                                          "or .h5, or FITS files.")
        else:
            ext = ''

        if ext == '.fits':
            if append:
                raise NotImplementedError()

            t = self.tbl.copy()

            if 't0' in t.meta:
                warnings.warn('This data file was produced with a deprecated '
                              'version of The Joker and uses old naming '
                              'conventions for the reference time. This file '
                              'may not work with future versions of thejoker.',
                              TheJokerDeprecationWarning)
                t.meta['t_ref'] = t.meta['t0']

            if t.meta.get('t_ref', None) is not None:
                t.meta['__t_ref_bmjd'] = t.meta.pop('t_ref').tcb.mjd
            t.write(output, overwrite=overwrite)
        else:
            write_table_hdf5(self.tbl, output, path=self._hdf5_path,
                             compression=False,
                             append=append, overwrite=overwrite,
                             serialize_meta=True, metadata_conflicts='error',
                             maxshape=(None, ))

    @classmethod
    def _read_tables(cls, group, path=None):
        if path is None:
            path = cls._hdf5_path

        samples = group[f'{path}']
        metadata = group[f'{path}.__table_column_meta__']

        header = meta.get_header_from_yaml(
            h.decode('utf-8') for h in metadata.read())

        table = Table(np.array(samples.read()))
        if 'meta' in list(header.keys()):
            table.meta = header['meta']

        table = serialize._construct_mixins_from_columns(table)

        return cls(table)

    @classmethod
    def read(cls, filename, path=None):
        """
        Read the samples data to a file.

        Currently, we only support writing to / reading from HDF5 and FITS
        files, so the filename must end in a .hdf5, .h5, or .fits extension.

        Parameters
        ----------
        filename : str, `h5py.File`, `h5py.Group`
            The output filename or HDF5 group.
        """
        import tables as tb
        if isinstance(filename, tb.group.Group):
            return cls._read_tables(filename)

        if path is None:
            path = cls._hdf5_path

        if isinstance(filename, str):
            try:
                ext = os.path.splitext(filename)[1]
            except Exception:
                raise ValueError(f"Invalid file name {filename}")

            if ext in ['.hdf5', '.h5']:
                tbl = QTable.read(filename, path=path)
            else:
                tbl = QTable.read(filename)

                if '__t_ref_bmjd' in tbl.meta.keys():
                    tbl.meta['t_ref'] = Time(tbl.meta['__t_ref_bmjd'],
                                             format='mjd', scale='tcb')

        else:
            tbl = QTable.read(filename, path=path)

        return cls(samples=tbl, **tbl.meta)

    def copy(self):
        """Return a copy of this instance"""
        return self.__class__(self.tbl.copy(), t_ref=self.t_ref)

    def ln_unmarginalized_likelihood(self, data):
        """
        Compute the log (unmarginalized) likelihood of the data for each sample
        """

        data_rv = data.rv.value
        data_unit = data.rv.unit
        data_var = (data.rv_err.to_value(data_unit)) ** 2

        if 's' in self.tbl.colnames:
            s_vars = self['s'].to_value(data_unit) ** 2
        else:
            s_vars = np.zeros(len(self))

        lls = np.full(len(self), np.nan)
        for i, (orbit, s) in enumerate(zip(self.orbits, s_vars)):
            model_rv = orbit.radial_velocity(data.t)
            lls[i] = ln_normal(model_rv.to_value(data_unit),
                               data_rv,
                               data_var + s).sum()

        return lls
