# Standard library
from collections import OrderedDict
import copy
import warnings

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time
import numpy as np
from twobody import KeplerOrbit, Barycenter

# Package
from ..utils import quantity_to_hdf5, quantity_from_hdf5

__all__ = ['JokerSamples']


class JokerSamples(OrderedDict):
    _valid_keys = ['P', 'M0', 'e', 'omega', 'jitter', 'K', 'v0']

    def __init__(self, t0=None, **kwargs):
        """A dictionary-like object for storing posterior samples from
        The Joker, with some extra functionality.

        Parameters
        ----------
        t0 : `astropy.time.Time`, numeric (optional)
            The reference time for the orbital parameters.
        **kwargs
            These are the orbital element names.
        """

        # initialize empty dictionary
        super(JokerSamples, self).__init__()

        # reference time
        self.t0 = t0

        self._size = None
        self._shape = None
        for key, val in kwargs.items():
            self[key] = val # calls __setitem__ below

        self._cache = dict()

    def _validate_key(self, key):
        if key not in self._valid_keys:
            raise ValueError("Invalid key '{0}'.".format(key))

    def _validate_val(self, val):
        val = u.Quantity(val)
        if self._shape is not None and val.shape != self.shape:
            raise ValueError("Shape of new samples must match those already "
                             "stored! ({0}, expected {1})"
                             .format(len(val), self.shape))

        return val

    def __getitem__(self, slc):
        if isinstance(slc, str):
            return super(JokerSamples, self).__getitem__(slc)

        else:
            new = copy.copy(self)
            new._size = None # reset number of samples
            new._shape = None # reset number of samples

            for k in self.keys():
                new[k] = self[k][slc]

            return new

    def __setitem__(self, key, val):
        self._validate_key(key)
        val = self._validate_val(val)

        if self._shape is None:
            self._shape = val.shape
            self._size = val.size

        super(JokerSamples, self).__setitem__(key, val)

    @property
    def n_samples(self):
        warnings.warn(".n_samples is deprecated in favor of .size",
                      DeprecationWarning)
        return self.size

    @property
    def size(self):
        if self._size is None:
            raise ValueError("No samples stored!")
        return self._size

    @property
    def shape(self):
        if self._shape is None:
            raise ValueError("No samples stored!")
        return self._shape

    def __len__(self):
        return self.n_samples

    def __str__(self):
        return ("<JokerSamples in [{0}], {1} samples>"
                .format(','.join(self.keys()), len(self)))

    @classmethod
    def from_hdf5(cls, f, n=None, **kwargs):
        """
        Parameters
        ----------
        f : :class:`h5py.File`, :class:`h5py.Group`
        n : int (optional)
            The number of samples to load.
        **kwargs
            All other keyword arguments are passed to the class initializer.
        """

        if 't0_bmjd' in f.attrs:
            # Read the reference time:
            t0 = Time(f.attrs['t0_bmjd'], format='mjd', scale='tcb')
        else:
            t0 = None

        samples = cls(t0=t0, **kwargs)
        for key in cls._valid_keys:
            if key in f:
                samples[key] = quantity_from_hdf5(f, key, n=n)

        return samples

    def to_hdf5(self, f):
        """
        Parameters
        ----------
        f : :class:`h5py.File`, :class:`h5py.Group`
        """

        for key in self.keys():
            quantity_to_hdf5(f, key, self[key])

        if self.t0 is not None:
            f.attrs['t0_bmjd'] = self.t0.tcb.mjd

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
        v0 = self['v0']
        a = kwargs.pop('a', P * K / (2*np.pi) * np.sqrt(1 - e**2))

        if len(self) == 1 and len(self.shape) == 0:
            if index > 0:
                raise ValueError('Samples are scalar-valued!')

        else:
            P = P[index]
            e = e[index]
            a = a[index]
            omega = omega[index]
            M0 = M0[index]
            v0 = v0[index]

        orbit.elements._P = P
        orbit.elements._e = e * u.dimensionless_unscaled
        orbit.elements._a = a
        orbit.elements._omega = omega
        orbit.elements._M0 = M0
        orbit.elements._Omega = kwargs.pop('Omega', 0*u.deg)
        orbit.elements._i = kwargs.pop('i', 90*u.deg)

        orbit._barycenter = kwargs.pop('barycenter', None)

        if kwargs:
            raise ValueError("Unrecognized arguments {0}"
                             .format(', '.join(list(kwargs.keys()))))

        # TODO: slight abuse of the _v0 cache attribute on KeplerOrbit...
        orbit._v0 = v0

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
