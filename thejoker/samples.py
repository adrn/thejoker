# Standard library
from collections import OrderedDict
import copy
import os

# Third-party
import astropy.units as u
from astropy.table import Table, QTable
from astropy.table.meta import get_header_from_yaml, get_yaml_from_table
import numpy as np
from twobody import KeplerOrbit, PolynomialRVTrend

# Project
from .prior_helpers import (_validate_polytrend, _get_nonlinear_equiv_units,
                            _get_linear_equiv_units)
from .samples_helpers import write_table_hdf5

__all__ = ['JokerSamples']


class JokerSamples:
    _hdf5_path = 'samples'

    def __init__(self, samples=None, poly_trend=1, t0=None, **kwargs):
        """A dictionary-like object for storing prior or posterior samples from
        The Joker, with some extra functionality.

        Parameters
        ----------
        samples : `~astropy.table.QTable`, table-like (optional)
            The samples data as an Astropy table object, or something
            convertable to an Astropy table (e.g., a dictionary with
            `~astropy.units.Quantity` object values). This is optional because
            the samples data can be added later by setting keys on the resulting
            instance.
        poly_trend : int (optional)
            Specifies the number of coefficients in an additional polynomial
            velocity trend, meant to capture long-term trends in the data. See
            the docstring for `thejoker.JokerPrior` for more details.
        t0 : `astropy.time.Time`, numeric (optional)
            The reference time for the orbital parameters.
        **kwargs
            Additional keyword arguments are stored internally as metadata.
        """
        self.tbl = QTable()

        if isinstance(samples, Table):
            poly_trend = samples.meta.pop('poly_trend', poly_trend)
            t0 = samples.meta.pop('t0', t0)
            kwargs.update(samples.meta)

        poly_trend, v_trend_names = _validate_polytrend(poly_trend)

        self._valid_units = {**_get_nonlinear_equiv_units(),
                             **_get_linear_equiv_units(v_trend_names)}

        self.tbl.meta['poly_trend'] = poly_trend
        self.tbl.meta['t0'] = t0
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
            key = slice(key, key+1)
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
    def poly_trend(self):
        return self.tbl.meta['poly_trend']

    @property
    def t0(self):
        return self.tbl.meta['t0']

    @property
    def par_names(self):
        return self.tbl.colnames

    def __len__(self):
        return len(self.tbl)

    def __repr__(self):
        return (f'<JokerSamples [{", ".join(self.par_names)}] '
                f'({len(self)} samples)>')

    def __str__(self):
        return self.__repr__()

    ##########################################################################
    # Interaction with TwoBody

    def get_orbit(self, index=None, **kwargs):
        """Get a `twobody.KeplerOrbit` object for the samples at the specified
        index.

        Parameters
        ----------
        index : int (optional)
            The index of the samples to turn into a `twobody.KeplerOrbit`
            instance. If the samples object is scalar, no index is necessary.
        **kwargs
            Other keyword arguments are passed to the `twobody.KeplerOrbit`
            initializer. For example, you can specify the inclination by passing
            ``i=...`, or  longitude of the ascending node by passing
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
                                               a=1*u.au, t0=self.t0)

        # all of this to avoid the __init__ of KeplerOrbit / KeplerElements
        orbit = copy.copy(self._cache['orbit'])

        P = self['P']
        e = self['e']
        K = self['K']
        omega = self['omega']
        M0 = self['M0']
        a = kwargs.pop('a', P * K / (2*np.pi) * np.sqrt(1 - e**2))

        _, v_trend_names = _validate_polytrend(self.poly_trend)
        names = list(_get_linear_equiv_units(v_trend_names).keys())
        if len(self) == 1:
            if index is not None and index > 0:
                raise ValueError('Samples are scalar-valued!')

            trend_coeffs = [self[x] for x in names[1:]]  # skip K

        else:
            P = P[index]
            e = e[index]
            a = a[index]
            omega = omega[index]
            M0 = M0[index]

            trend_coeffs = [self[x][index] for x in names[1:]]  # skip K

        orbit.elements._P = P
        orbit.elements._e = e * u.dimensionless_unscaled
        orbit.elements._a = a
        orbit.elements._omega = omega
        orbit.elements._M0 = M0
        orbit.elements._Omega = kwargs.pop('Omega', 0*u.deg)
        orbit.elements._i = kwargs.pop('i', 90*u.deg)
        orbit._vtrend = PolynomialRVTrend(trend_coeffs, t0=self.t0)
        orbit._barycenter = kwargs.pop('barycenter', None)

        if kwargs:
            raise ValueError("Unrecognized arguments {0}"
                             .format(', '.join(list(kwargs.keys()))))

        return orbit

    @property
    def orbits(self):
        """A generator that successively returns `twobody.KeplerOrbit` objects
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
        """Return a new scalar object by taking the median across all samples"""
        return self._apply(np.mean)

    def std(self):
        """Return a new scalar object by taking the standard deviation across
        all samples"""
        return self._apply(np.std)

    # Packing and unpacking
    def pack(self, units=None, nonlinear_only=True):
        """Pack the sample data into a single numpy array (i.e. strip the units
        and return those separately).

        Parameters
        ----------
        units : `dict` (optional)
            If specified, this controls the units that the samples are converted
            to before packing into a single array.
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

        arrs = []
        for name in self.par_names:
            unit = units.get(name, self.tbl[name].unit)
            arrs.append(self.tbl[name].to_value(unit))
            out_units[name] = unit

        if 's' not in self.par_names:
            arrs.append(np.zeros_like(arrs[0]))
            out_units['s'] = u.m/u.s

        return np.stack(arrs, axis=1), out_units

    @classmethod
    def unpack(cls, packed_samples, units, poly_trend=1, t0=None, **kwargs):
        """Unpack the array of packed (prior) samples and return a
        `~thejoker.JokerSamples` instance.

        Parameters
        ----------
        packed_samples : array_like
        units : `~collections.OrderedDict`, dict_like
            The units of each column in the packed samples array. The order of
            columns in this dictionary determines the assumed order of the
            columns in the packed samples array.
        **kwargs
            Additional keyword arguments are stored internally as metadata.

        Returns
        -------
        samples : `~thejoker.JokerSamples`
            The unpacked samples.

        """
        packed_samples = np.array(packed_samples)
        nsamples, npars = packed_samples.shape

        samples = cls(poly_trend=poly_trend, t0=t0, **kwargs)
        for i, k in enumerate(list(units.keys())[:npars]):
            unit = units[k]
            samples[k] = packed_samples[:, i] * unit
        return samples

    def write(self, filename, overwrite=False, append=False):
        """
        Save the samples data to a file. This is a thin wrapper around the
        ``astropy.table`` write machinery, so the output format is inferred from
        the filename extension, and all kwargs are passed to
        `astropy.table.Table.write()`.

        Currently, we only support writing to / reading from HDF5 files, so the
        filename must end in a .hdf5 or .h5 extension.

        Parameters
        ----------
        filename : str
            The output filename.
        """
        # To get the table metadata / units:
        # from astropy.table import meta
        # test = meta.get_header_from_yaml(
        #     h.decode('utf-8') for h in f['__astropy_table__.__table_column_meta__'])
        # h5file.root.samples.append()

        import h5py

        ext = os.path.splitext(filename)[1]
        if ext not in ['.hdf5', '.h5']:
            raise NotImplementedError("We currently only support writing to "
                                      "HDF5 files, with extension .hdf5 or .h5")

        write_table_hdf5(self.tbl, filename, path=self._hdf5_path,
                         compression=False,
                         append=append, overwrite=overwrite,
                         serialize_meta=True, metadata_conflicts='error',
                         maxshape=(None, ))

        return

        if overwrite and append_existing:
            raise ValueError("overwrite and append cannot both be set to True.")

        if os.path.exists(filename) and not overwrite:
            with h5py.File(filename, 'r') as f:
                if HDF5_PATH_NAME in f.keys():
                    has_data = True

                    # load existing column metadata
                    header_grp = f[f"{HDF5_PATH_NAME}.__table_column_meta__"]
                    header = get_header_from_yaml(
                        h.decode('utf-8') for h in header_grp)

                    meta = {k: v for k, v in header['meta'].items()
                            if not k.startswith('_')}

                else:
                    has_data = False

            if has_data and append_existing:
                # First, compare metadata between this object and on disk
                for k, v in meta.items():
                    if k not in self.tbl.meta or self.tbl.meta[k] != v:
                        raise ValueError(
                            "Cannot append table to existing file because the "
                            "existing file table metadata and this object's "
                            "table metadata do not match. Key with conflict: "
                            f"{k}, {self.tbl.meta[k]} vs. {v}")

                # Now compare datatype of this object and on disk
                this_header = get_header_from_yaml(
                    get_yaml_from_table(self.tbl))

                if not _custom_tbl_dtype_compare(header['datatype'],
                                                 this_header['datatype']):
                    raise ValueError(
                        "Cannot append table to existing file because "
                        "the existing file table datatype and this "
                        "object's table datatype do not match. "
                        f"{header['datatype']} vs. {this_header['datatype']}")

                # If we got here, we can now try to append:
                with h5py.File(filename, 'a') as f:
                    current_size = len(f[HDF5_PATH_NAME])
                    f[HDF5_PATH_NAME].resize((current_size + len(self), ))
                    f[HDF5_PATH_NAME][current_size:] = self.tbl.as_array()

                return None

        self.tbl.write(filename, format='hdf5',
                       path=HDF5_PATH_NAME, serialize_meta=True,
                       overwrite=overwrite)

    @classmethod
    def read(cls, filename):
        tbl = QTable.read(filename, path=cls._hdf5_path)
        return cls(samples=tbl, **tbl.meta)
