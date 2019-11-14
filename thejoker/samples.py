# Standard library
from collections import OrderedDict
import copy
import os

# Third-party
import astropy.units as u
from astropy.table import Table, QTable
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
        """
        A dictionary-like object for storing prior or posterior samples from
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

        # log-prior and log-likelihood values are also valid:
        self._valid_units['ln_prior'] = u.one
        self._valid_units['ln_likelihood'] = u.one

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
        """Return a new scalar object by taking the median across all samples"""
        return self._apply(np.mean)

    def std(self):
        """Return a new scalar object by taking the standard deviation across
        all samples"""
        return self._apply(np.std)

    # Packing and unpacking
    def pack(self, units=None, nonlinear_only=True):
        """
        Pack the sample data into a single numpy array (i.e. strip the units and
        return those separately).

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
        Write the samples data to a file.

        Currently, we only support writing to / reading from HDF5 files, so the
        filename must end in a .hdf5 or .h5 extension.

        Parameters
        ----------
        filename : str
            The output filename.
        overwrite : bool (optional)
            Overwrite the existing file.
        append : bool (optional)
            Append the samples to an existing table in the specified filename.
        """
        ext = os.path.splitext(filename)[1]
        if ext not in ['.hdf5', '.h5']:
            raise NotImplementedError("We currently only support writing to "
                                      "HDF5 files, with extension .hdf5 or .h5")

        write_table_hdf5(self.tbl, filename, path=self._hdf5_path,
                         compression=False,
                         append=append, overwrite=overwrite,
                         serialize_meta=True, metadata_conflicts='error',
                         maxshape=(None, ))

    @classmethod
    def read(cls, filename):
        """
        Read the samples data to a file.

        Currently, we only support writing to / reading from HDF5 files, so the
        filename must end in a .hdf5 or .h5 extension.

        Parameters
        ----------
        filename : str
            The output filename.
        """
        tbl = QTable.read(filename, path=cls._hdf5_path)
        return cls(samples=tbl, **tbl.meta)
