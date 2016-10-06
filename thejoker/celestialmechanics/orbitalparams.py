# Standard library
from collections import OrderedDict

# Third-party
from astropy import log as logger
from astropy.constants import G
import astropy.units as u
import h5py
import numpy as np
import six

# Project
from ..util import quantity_from_hdf5
from ..units import default_units

class OrbitalParams(object):
    # mapping from parameter name to default unit. the order here is the order in
    #   which parameters are `pack()`ed and `unpack()`ed.
    _name_to_unit = OrderedDict()
    _name_to_unit['P'] = default_units['P']
    _name_to_unit['ecc'] = default_units['ecc']
    _name_to_unit['omega'] = default_units['omega']
    _name_to_unit['phi0'] = default_units['phi0']
    _name_to_unit['jitter'] = default_units['jitter']
    _name_to_unit['K'] = default_units['K']
    _name_to_unit['v0'] = default_units['v0']

    # Latex plot labels for the parameters
    # TODO: update
    _latex_labels = [r'$\ln (P/{\rm day})$', r'$\ln (a_1\,\sin i/{\rm R}_\odot)$', '$e$',
                     r'$\omega$ [deg]', r'$\phi_0$ [deg]', '$v_0$ [km s$^{-1}$]', 's [m s$^{-1}]$']

    def __init__(self, P, K, ecc, omega, phi0, v0, jitter=0.*u.m/u.s):
        """
        """

        # parameters are stored internally without units (for speed) but can
        #   be accessed with units without the underscore prefix (e.g., .P vs ._P)
        for name,unit in self._name_to_unit.items():
            if unit is not None and unit is not u.one:
                setattr(self, "_{}".format(name), np.atleast_1d(eval(name)).to(unit).value)
            else:
                setattr(self, "_{}".format(name), np.atleast_1d(eval(name)))

        # validate shape of inputs
        if self._P.ndim > 1:
            raise ValueError("Only ndim=1 arrays are supported!")

        for key in self._name_to_unit.keys():
            if getattr(self, key).shape != self._P.shape:
                raise ValueError("All inputs must have the same length!")

    def __getattr__(self, name):
        # this is a crazy hack to automatically apply units to attributes
        #   named after each of the parameters
        if name not in self._name_to_unit.keys():
            raise AttributeError("Invalid attribute name '{}'".format(name))

        # get unitless representation
        val = getattr(self, '_{}'.format(name))

        return val * self._name_to_unit[name]

    def __len__(self):
        return len(self._P)

    def __copy__(self):
        kw = dict()
        for key in self._name_to_unit.keys():
            kw[key] = getattr(self, key).copy()
        return self.__class__(**kw)

    def __getitem__(self, slicey):
        cpy = self.copy()
        for key in self._name_to_unit.keys():
            slice_val = getattr(self, "_{}".format(key))[slicey]
            setattr(cpy, "_{}".format(key), slice_val)
        return cpy

    @classmethod
    def from_hdf5(cls, f):
        kwargs = dict()
        if isinstance(f, six.string_types):
            with h5py.File(f, 'r') as g:
                for key in cls._name_to_unit.keys():
                    kwargs[key] = quantity_from_hdf5(g, key)

        else:
            for key in cls._name_to_unit.keys():
                kwargs[key] = quantity_from_hdf5(f, key)

        return cls(**kwargs)

    def pack(self, units=None, plot_transform=False):
        """
        Pack the orbital parameters into a single array structure
        without associated units. The components will have units taken
        from the unit system defined in `thejoker.units.usys`.

        Parameters
        ----------
        units : dict (optional)
        plot_transform : bool (optional)

        Returns
        -------
        pars : `numpy.ndarray`
            A single 2D array containing the parameter values with no
            units. Will have shape ``(n,6)``.

        """
        if units is None:
            all_samples = np.vstack([getattr(self, "_{}".format(key))
                                     for key in self._name_to_unit.keys()]).T

        else:
            all_samples = np.vstack([getattr(self, format(key)).to(units[key]).value
                                     for key in self._name_to_unit.keys()]).T

        if plot_transform:
            # ln P in plots:
            idx = self._name_to_unit.keys().index('P')
            all_samples[:,idx] = np.log(all_samples[:,idx])

        return all_samples

    @classmethod
    def unpack(cls, pars):
        """
        Unpack a 2D array structure containing the orbital parameters
        without associated units. Should have shape ``(n,6)`` where ``n``
        is the number of parameters.

        Returns
        -------
        p : `~thejoker.celestialmechanics.OrbitalParams`

        """
        kw = dict()
        par_arr = np.atleast_2d(pars).T
        for i,key in enumerate(cls._name_to_unit.keys()):
            kw[key] = par_arr[i] * cls._name_to_unit[key]

        return cls(**kw)

    def copy(self):
        return self.__copy__()

    def rv_orbit(self, index=None):
        """
        Get a `~thejoker.celestialmechanics.SimulatedRVOrbit` instance
        for the orbital parameters with index ``i``.

        Parameters
        ----------
        index : int (optional)

        Returns
        -------
        orbit : `~thejoker.celestialmechanics.SimulatedRVOrbit`
        """
        from .celestialmechanics_class import SimulatedRVOrbit

        if index is None and len(self._P) == 1: # OK
            index = 0

        elif index is None and len(self._P) > 1:
            raise IndexError("You must specify the index of the set of paramters to get an "
                             "orbit for!")

        i = index
        return SimulatedRVOrbit(self[i])

    # Computed Quantities
    @property
    def asini(self):
        return (self.K/(2*np.pi) * (self.P * np.sqrt(1-self.ecc**2))).to(default_units['asini'])

    @property
    def mf(self):
        mf = self.P * self.K**3 / (2*np.pi*G) * (1 - self.ecc**2)**(3/2.)
        return mf.to(default_units['mf'])

    @staticmethod
    def mf_asini_ecc_to_P_K(mf, asini, ecc):
        P = 2*np.pi * asini**(3./2) / np.sqrt(G * mf)
        K = 2*np.pi * asini / (P * np.sqrt(1-ecc**2))
        return P.to(default_units['P']), K.to(default_units['K'])
