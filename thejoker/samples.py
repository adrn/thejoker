# Standard library
from collections import OrderedDict
import copy

# Third-party
import astropy.units as u
from astropy.table import QTable
import numpy as np
from twobody import KeplerOrbit, PolynomialRVTrend

__all__ = ['JokerSamples']


class JokerSamples:

    def __init__(self, prior, samples=None, t0=None, **kwargs):
        """A dictionary-like object for storing posterior samples from
        The Joker, with some extra functionality.

        Parameters
        ----------
        prior : `thejoker.JokerPrior`
            TODO:
        samples : `~astropy.table.Table` or table-like
            TODO:
        t0 : `astropy.time.Time`, numeric (optional)
            The reference time for the orbital parameters.
        **kwargs :
            TODO: Stored as metadata.
        """

        from .prior import JokerPrior
        if not isinstance(prior, JokerPrior):
            raise TypeError("TODO")
        self.prior = prior

        self.tbl = QTable(samples)
        self.tbl.meta['poly_trend'] = self.prior.poly_trend
        self.tbl.meta['t0'] = t0
        for k, v in kwargs.items():
            self.tbl.meta[k] = v

        self._valid_units = {**self.prior._nonlinear_pars,
                             **self.prior._linear_pars}

        self._cache = dict()

    def __getitem__(self, key):
        return self.tbl[key]

    def __setitem__(self, key, val):
        if key not in self._valid_units:
            raise ValueError(f"Invalid parameter name '{key}'. Must be one "
                             "of: {0}".format(list(self._valid_units.keys())))

        if not hasattr(val, 'unit'):
            raise TypeError("Values must be added an astropy Quantity object.")

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

        if len(self) == 1:
            if index is not None and index > 0:
                raise ValueError('Samples are scalar-valued!')

            trend_coeffs = [self[x] for x in self._trend_names]

        else:
            P = P[index]
            e = e[index]
            a = a[index]
            omega = omega[index]
            M0 = M0[index]
            trend_coeffs = [self[x][index] for x in self._trend_names]

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
            new_samples[k] = func(self[k])

        return cls(self.prior, tbl=new_samples, **self.meta)

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
        """TODO:

        Parameters
        ----------
        nonlinear_only : bool (optional)
            TODO
        """
        if units is None:
            units = OrderedDict()

        arrs = []
        for name in self.par_names:
            unit = units.get(name, self.tbl[name].unit)
            arrs.append(self.tbl[name].to_value(unit))
            units[name] = unit

        if 's' not in self.par_names:
            arrs.append(np.zeros_like(arrs[0]))
            units['s'] = u.m/u.s

        return np.stack(arrs, axis=1), units

    @classmethod
    def unpack(cls, packed_samples, prior, units, t0=None):
        """TODO:

        Parameters
        ----------
        packed_samples :
        units : `~collections.OrderedDict`, dict_like
            TODO: sets the order of packed_samples...
        """

        nsamples, npars = packed_samples.shape

        samples = cls(prior=prior, t0=t0)
        for i, k in enumerate(list(units.keys())[:npars]):
            unit = units[k]
            samples[k] = packed_samples[:, i] * unit
        return samples

    def save(self, filename, save_prior=True):
        pass

    @classmethod
    def load(cls, filename, prior_filename=None):
        pass
