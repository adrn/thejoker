# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import numpy as np
import six

# Project
from ..util import quantity_from_hdf5
from ..units import usys
from .celestialmechanics_class import SimulatedRVOrbit

class OrbitalParams(object):
    # Mapping from parameter name to physical type
    _name_phystype = {
        'P': 'time',
        'asini': 'length',
        'ecc': None,
        'omega': 'angle',
        'phi0': 'angle',
        'v0': 'speed'
    }

    # Latex plot labels for the parameters
    _latex_labels = [r'$\ln (P/{\rm day})$', r'$\ln (a\,\sin i/{\rm R}_\odot)$', '$e$', r'$\omega$ [deg]',
                     r'$\phi_0$ [deg]', '$v_0$ [km s$^{-1}$]']

    @u.quantity_input(P=u.day, asini=u.R_sun, omega=u.degree, phi0=u.degree, v0=u.km/u.s)
    def __init__(self, P, asini, ecc, omega, phi0, v0):
        """
        """

        # parameters are stored internally without units (for speed) but can
        #   be accessed with units without the underscore prefix (e.g., .P vs ._P)
        self._P = np.atleast_1d(P).decompose(usys).value
        self._asini = np.atleast_1d(asini).decompose(usys).value
        self._ecc = np.atleast_1d(ecc)
        self._omega = np.atleast_1d(omega).decompose(usys).value
        self._phi0 = np.atleast_1d(phi0).decompose(usys).value
        self._v0 = np.atleast_1d(v0).decompose(usys).value

        # validate shape of inputs
        if self._P.ndim > 1:
            raise ValueError("Only ndim=1 arrays are supported!")

        for key in self._name_phystype.keys():
            if getattr(self, key).shape != self._P.shape:
                raise ValueError("All inputs must have the same length!")

    def __getattr__(self, name):
        # this is a crazy hack to automatically apply units to attributes
        #   named after each of the parameters
        if name not in self._name_phystype.keys():
            raise AttributeError("Invalid attribute name '{}'".format(name))

        # get unitless representation
        val = getattr(self, '_{}'.format(name))

        if self._name_phystype[name] is None:
            return val
        else:
            return val * usys[self._name_phystype[name]]

    def __len__(self):
        return len(self._P)

    def __copy__(self):
        kw = dict()
        for key in self._name_phystype.keys():
            kw[key] = getattr(self, key).copy()
        return self.__class__(**kw)

    def __getitem__(self, slicey):
        c = self.copy()
        for key in self._name_phystype.keys():
            slice_val = getattr(self, "_{}".format(key))[slicey]
            setattr(c, "_{}".format(key), slice_val)
        return c

    @classmethod
    def from_hdf5(cls, f):
        kwargs = dict()
        if isinstance(f, six.string_types):
            with h5py.File(f, 'r') as g:
                for key in cls._name_phystype.keys():
                    kwargs[key] = quantity_from_hdf5(g, key)

        else:
            for key in cls._name_phystype.keys():
                kwargs[key] = quantity_from_hdf5(f, key)

        return cls(**kwargs)

    def pack(self, plot_units=False):
        """
        Pack the orbital parameters into a single array structure
        without associated units. The components will have units taken
        from the unit system defined in `thejoker.units.usys`.

        Parameters
        ----------
        plot_units : bool (optional)

        Returns
        -------
        pars : `numpy.ndarray`
            A single 2D array containing the parameter values with no
            units. Will have shape ``(n, 6)``.

        """
        if not plot_units:
            all_samples = np.vstack((self._P, self._asini, self.ecc,
                                     self._omega, self._phi0, self._v0)).T

        else:
            all_samples = np.vstack((np.log(self.P.to(u.day).value),
                                     np.log(self.asini.to(u.R_sun).value),
                                     self.ecc,
                                     self.omega.to(u.degree).value % 360.,
                                     self.phi0.to(u.degree).value,
                                     self.v0.to(u.km/u.s).value)).T

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
        P, asini, ecc, omega, phi0, v0 = np.atleast_2d(pars).T
        return cls(P=P*usys['time'], asini=asini*usys['length'], ecc=ecc,
                   omega=omega*usys['angle'], phi0=phi0*usys['angle'], v0=v0*usys['speed'])

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
        i = index

        if i is None and len(self._P) == 1: # OK
            pass

        elif i is None and len(self._P) > 1:
            raise IndexError("You must specify the index of the set of paramters to get an "
                             "orbit for!")

        return SimulatedRVOrbit(P=self.P[i], a_sin_i=self.asini[i], ecc=self.ecc[i],
                                omega=self.omega[i], phi0=self.phi0[i], v0=self.v0[i])
