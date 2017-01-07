"""
Could have something like:

    # remember: right now, priors on these terms are assumed to be broad and Gaussian
    pvt = PolynomialVelocityTrend(n_terms=2) # linear
    pvt.data_mask = lambda d: d._t_bmjd < 55562.24 # only apply to data before a date

    # todo: marginalization then has to happen piece-wise?!

Future:

- Maybe make NonlinearParameter classes so that even nonlinear parameters can be extended?
- Parameter class has a .sample_prior() method, .evaluate_prior()??
- Some parameters are just required, need to be packed -- multiproc_helpers
    functions just need to unpack/pack smarter?

"""

# Third-party
from astropy.utils.misc import isiterable
import astropy.units as u

# Project
from ..celestialmechanics.trends import PolynomialVelocityTrend
# from ..util import quantity_from_hdf5, quantity_to_hdf5

__all__ = ['JokerParams']

class JokerParams(object):
    """

    Parameters
    ----------
    P_min : `astropy.units.Quantity` [time]
        Lower bound on prior over period, the smallest period considered.
    P_max : `astropy.units.Quantity` [time]
        Upper bound on prior over period, the largest period considered.
    trends : iterable (optional)
        A list of `~thejoker.TODO.PolynomialVelocityTrend` instances.
    jitter : `~astropy.units.Quantity` [speed], tuple (optional)
        Represents additional Gaussian noise in the RV signal. Default
        is to fix the value of the jitter to 0. To fix the jitter to a
        different value, pass in a single `~astropy.units.Quantity`
        object. The Joker also supports inferring the jitter as an
        additional non-linear parameter. Currently, the only prior
        pdf supported for doing this is a Gaussian in natural-log of
        the jitter squared--that is,
        :math:`p(x) = \mathcal{N}(x|\mu,\sigma)` where
        :math:`a = \log s^2`. The (dimensionless) mean and standard
        deviation of this prior can also be passed in to this argument
        by passing a length-2 tuple of numbers. If you do this, you must
        also pass in a unit for the jitter using the ``jitter_unit`` arg.
    jitter_unit : `~astropy.units.UnitBase`
        If sampling over the jitter as an extra non-linear parameter,
        you must also specify the units of the jitter prior. See note
        above about the ``jitter`` argument.

    Examples
    --------

        >>> import astropy.units as u
        >>> pars = JokerPars(jitter=5.*u.m/u.s) # fix jitter to 5 m/s
        >>> pars = JokerPars(jitter=(1., 2.), jitter_unit=u.m/u.s) # specify jitter prior

    """
    @u.quantity_input(P_min=u.day, P_max=u.day)
    def __init__(self, P_min, P_max, trends=None, jitter=None, jitter_unit=None):

        # the names of the default parameters
        self.default_params = ['P', 'K', 'ecc', 'omega', 'phi0']

        self.P_min = P_min
        self.P_max = P_max

        # validate the specified long-term velocity trends
        if trends is None:
            trends = []
            trends.append(PolynomialVelocityTrend(n_terms=1))

        elif not isiterable(trends):
            trends = [trends]

        if len(trends) > 1:
            # TODO: need to implement having different velocity trends for parts of the
            #       time series. Need to modify design_matrix() be a lot bigger, with zero'd
            #       out rows for each trend.
            raise NotImplementedError("Sorry, right now only global velocity trends "
                                      "are implemented!")

        # TODO: check somewhere (maybe in TheJoker.sample()) that the sum is equal to data length
        # data_mask_tot = 0
        # data_mask_tot += trend.mask.sum()
        for trend in trends:
            # TODO: we may want to allow more general trends in the future, but for now...
            if not isinstance(trend, PolynomialVelocityTrend):
                raise TypeError("Velocity trends must be PolynomialVelocityTrend "
                                "instances, not '{}'".format(type(trend)))

        self.trends = trends

        # validate the input jitter specification
        if jitter is None:
            jitter = 0*u.km/u.s

        if isiterable(jitter):
            if len(jitter) != 2:
                raise ValueError("If specifying parameters for the jitter prior, you "
                                 "must pass in a length-2 container containing the "
                                 "mean and standard deviation of the Gaussian over "
                                 "log(jitter^2)")

            if jitter_unit is None or not isinstance(jitter_unit, u.UnitBase):
                raise TypeError("If specifying parameters for the jitter prior, you "
                                "must also specify the units of the jitter for "
                                "evaluating the prior as an astropy.units.UnitBase "
                                "instance.")

            self._fixed_jitter = False
            self._jitter_unit = jitter_unit
            self.jitter = jitter

        else:
            self._fixed_jitter = True
            self.jitter = jitter

    @property
    def num_params(self):
        n = len(self.default_params)
        n += self._fixed_jitter
        n += sum([trend.n_terms for trend in self.trends])
        return n

    # --- All old shit below here that needs to be moved ---

    # @classmethod
    # def get_labels(cls, units=None):
    #     _u = dict()
    #     if units is None:
    #         _u = cls._name_to_unit

    #     else:
    #         for k,unit in cls._name_to_unit.items():
    #             if k in units:
    #                 _u[k] = units[k]
    #             else:
    #                 _u[k] = unit

    #     _labels = [
    #         r'$\ln (P/1\,${}$)$'.format(_u['P'].long_names[0]),
    #         '$e$',
    #         r'$\omega$ [{}]'.format(_u['omega']),
    #         r'$\phi_0$ [{}]'.format(_u['omega']),
    #         r'$\ln (s/1\,${}$)$'.format(_u['jitter'].to_string(format='latex_inline')),
    #         r'$K$ [{}]'.format(_u['K'].to_string(format='latex_inline')),
    #         '$v_0$ [{}]'.format(_u['v0'].to_string(format='latex_inline'))
    #     ]

    #     return _labels

    # @classmethod
    # def from_hdf5(cls, f):
    #     kwargs = dict()
    #     if isinstance(f, six.string_types):
    #         with h5py.File(f, 'r') as g:
    #             for key in cls._name_to_unit.keys():
    #                 kwargs[key] = quantity_from_hdf5(g, key)

    #     else:
    #         for key in cls._name_to_unit.keys():
    #             kwargs[key] = quantity_from_hdf5(f, key)

    #     return cls(**kwargs)

    # def to_hdf5(self, f):
    #     if isinstance(f, six.string_types):
    #         with h5py.File(f, 'a') as g:
    #             for key in self._name_to_unit.keys():
    #                 quantity_to_hdf5(g, key, getattr(self, key))

    #     else:
    #         for key in self._name_to_unit.keys():
    #             quantity_to_hdf5(f, key, getattr(self, key))

    # def pack(self, units=None, plot_transform=False):
    #     """
    #     Pack the orbital parameters into a single array structure
    #     without associated units. The components will have units taken
    #     from the unit system defined in `thejoker.units.usys`.

    #     Parameters
    #     ----------
    #     units : dict (optional)
    #     plot_transform : bool (optional)

    #     Returns
    #     -------
    #     pars : `numpy.ndarray`
    #         A single 2D array containing the parameter values with no
    #         units. Will have shape ``(n,6)``.

    #     """
    #     if units is None:
    #         all_samples = np.vstack([getattr(self, "_{}".format(key))
    #                                  for key in self._name_to_unit.keys()]).T

    #     else:
    #         all_samples = np.vstack([getattr(self, format(key)).to(units[key]).value
    #                                  for key in self._name_to_unit.keys()]).T

    #     if plot_transform:
    #         # ln P in plots:
    #         idx = list(self._name_to_unit.keys()).index('P')
    #         all_samples[:,idx] = np.log(all_samples[:,idx])

    #         # ln s in plots:
    #         idx = list(self._name_to_unit.keys()).index('jitter')
    #         all_samples[:,idx] = np.log(all_samples[:,idx])

    #     return all_samples

    # @classmethod
    # def unpack(cls, pars):
    #     """
    #     Unpack a 2D array structure containing the orbital parameters
    #     without associated units. Should have shape ``(n,6)`` where ``n``
    #     is the number of parameters.

    #     Returns
    #     -------
    #     p : `~thejoker.celestialmechanics.OrbitalParams`

    #     """
    #     kw = dict()
    #     par_arr = np.atleast_2d(pars).T
    #     for i,key in enumerate(cls._name_to_unit.keys()):
    #         kw[key] = par_arr[i] * cls._name_to_unit[key]

    #     return cls(**kw)

    # def copy(self):
    #     return self.__copy__()

    # def rv_orbit(self, index=None):
    #     """
    #     Get a `~thejoker.celestialmechanics.SimulatedRVOrbit` instance
    #     for the orbital parameters with index ``i``.

    #     Parameters
    #     ----------
    #     index : int (optional)

    #     Returns
    #     -------
    #     orbit : `~thejoker.celestialmechanics.SimulatedRVOrbit`
    #     """
    #     from .celestialmechanics_class import SimulatedRVOrbit

    #     if index is None and len(self._P) == 1: # OK
    #         index = 0

    #     elif index is None and len(self._P) > 1:
    #         raise IndexError("You must specify the index of the set of paramters to get an "
    #                          "orbit for!")

    #     i = index
    #     return SimulatedRVOrbit(self[i])
