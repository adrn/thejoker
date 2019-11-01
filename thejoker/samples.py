# Standard library
import copy

# Third-party
import astropy.units as u
from astropy.table import QTable
import numpy as np
from twobody import KeplerOrbit, PolynomialRVTrend

__all__ = ['JokerSamples']


class JokerSamples(QTable):
    _valid_keys = ['P', 'M0', 'e', 'omega', 'jitter', 'K']

    def __init__(self, samples, *args, t0=None, poly_trend=1, **kwargs):
        """A dictionary-like object for storing posterior samples from
        The Joker, with some extra functionality.

        Parameters
        ----------
        samples :
        t0 : `astropy.time.Time`, numeric (optional)
            The reference time for the orbital parameters.
        poly_trend : int, optional
            If specified, sample over a polynomial velocity trend with the
            specified number of coefficients. For example, ``poly_trend=3`` will
            sample over parameters of a long-term quadratic velocity trend.
            Default is 1, just a constant velocity shift.
        **kwargs
            These are the orbital element names.
        """

        # initialize empty dictionary
        super().__init__(samples, *args, **kwargs)

        self.meta['poly_trend'] = int(poly_trend)
        self._trend_names = ['v{0}'.format(i)
                             for i in range(self.poly_trend)]
        for name in self._trend_names:
            if name not in keys:
                keys.append(name)
        return keys

        self.meta['t0'] = t0

        self._cache = dict()

    def __setitem__(self, key, val):
        if key not in self._valid_keys:
            raise ValueError(f"Invalid parameter name '{key}'. Must be one of: {self._valid_keys}")
        super().__setitem__(key, val)

    @property
    def poly_trend(self):
        return self.meta['poly_trend']

    @property
    def t0(self):
        return self.meta['t0']

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

        kw = dict()
        for k in self.keys():
            kw[k] = func(self[k])

        kw['t0'] = self.t0
        kw['poly_trend'] = self.poly_trend
        return cls(**kw)

    def mean(self):
        """Return a new scalar object by taking the mean across all samples"""
        return self._apply(np.mean)

    def median(self):
        """Return a new scalar object by taking the medin across all samples"""
        return self._apply(np.mean)

    def std(self):
        """Return a new scalar object by taking the medin across all samples"""
        return self._apply(np.std)
