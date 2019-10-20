from warnings import warn

# Third-party
from astropy.utils.misc import isiterable
import astropy.units as u
import numpy as np

__all__ = ['JokerParams']


class JokerParams:
    """

    Parameters
    ----------
    P_min : `astropy.units.Quantity` [time]
        Lower bound on prior over period, the smallest period considered.
    P_max : `astropy.units.Quantity` [time]
        Upper bound on prior over period, the largest period considered.
    jitter : `~astropy.units.Quantity` [speed], tuple (optional)
        Represents additional Gaussian noise in the RV signal. Default
        is to fix the value of the jitter to 0. To fix the jitter to a
        different value, pass in a single `~astropy.units.Quantity`
        object. The Joker also supports inferring the jitter as an
        additional non-linear parameter. Currently, the only prior
        pdf supported for doing this is a Gaussian in natural-log of
        the jitter squared--that is,
        :math:`p(a) = \mathcal{N}(a|\mu,\sigma)` where
        :math:`a = \log s^2`. The (dimensionless) mean and standard
        deviation of this prior can also be passed in to this argument
        by passing a length-2 tuple of numbers. If you do this, you must
        also pass in a unit for the jitter using the ``jitter_unit`` arg.
    jitter_unit : `~astropy.units.UnitBase`
        If sampling over the jitter as an extra non-linear parameter,
        you must also specify the units of the jitter prior. See note
        above about the ``jitter`` argument.
    poly_trend : int, optional
        If specified, sample over a polynomial velocity trend with the specified
        number of coefficients. For example, ``poly_trend=3`` will sample over
        parameters of a long-term quadratic velocity trend. Default is 1, just a
        constant velocity shift.
    linear_par_mu : array_like, optional
        Mean vector for the Gaussian prior on the linear parameters.
    linear_par_Lambda : array_like, optional
        Variance matrix that specifies the Gaussian prior on the linear
        parameters, i.e., the semi-amplitude and velocity trend paramters. The
        units must be in inverse, squared ``v_unit`` and ``v_unit/day^n`` where
        ``v_unit`` is the jitter velocity unit, and ``day^n`` corresponds to
        each polynomial trend coefficient.
    anomaly_tol : float (optional)
        Convergence tolerance passed to
        :func:`twobody.eccentric_anomaly_from_mean_anomaly`.
        Arbitrarily set to 1E-10 by default.
    anomaly_maxiter : float (optional)
        Maximum number of iterations passed to
        :func:`twobody.eccentric_anomaly_from_mean_anomaly`.
        Arbitrarily set to 128 by default.

    Examples
    --------

        >>> import astropy.units as u
        >>> pars = JokerParams(P_min=8*u.day, P_max=8192*u.day,
        ...                    linear_par_Lambda=np.diag([1e2, 1e2]) ** 2,
        ...                    jitter=5.*u.m/u.s) # fix jitter to 5 m/s
        >>> pars = JokerParams(P_min=8*u.day, P_max=8192*u.day,
        ...                    linear_par_Lambda=np.diag([1e2, 1e2]) ** 2,
        ...                    jitter=(1., 2.), jitter_unit=u.m/u.s) # specify jitter prior

    """
    @u.quantity_input(P_min=u.day, P_max=u.day)
    def __init__(self, P_min, P_max,
                 linear_par_Lambda=None, linear_par_mu=None,
                 jitter=None, jitter_unit=None,
                 poly_trend=1,
                 anomaly_tol=1E-10, anomaly_maxiter=128):

        # the names of the default parameters
        self.default_params = ['P', 'M0', 'e', 'omega', 'jitter', 'K']

        self.poly_trend = int(poly_trend)
        self.default_params += ['v{0}'.format(i)
                                for i in range(self.poly_trend)]

        # TODO: internally, time unit always taken to be days
        self.P_min = P_min
        self.P_max = P_max
        self.anomaly_tol = float(anomaly_tol)
        self.anomaly_maxiter = int(anomaly_maxiter)

        # K + the linear trend parameters
        _n_linear = 1 + self.poly_trend

        # TODO: ignoring units / assuming units are same as data here
        if linear_par_mu is None:
            linear_par_mu = np.zeros(_n_linear)

        if linear_par_Lambda is None:
            msg = (
                "You did not specify a prior for the linear parameters of "
                "The Joker, i.e. the velocity semi-amplitude, systemic "
                "velocity, or velocity trend parameters. A past version of "
                "The Joker erroneously assumed that this prior was very broad "
                "and could be ignored, but this bug has been fixed, and you "
                "must specify this prior. Currently, we only support a "
                "Gaussian prior on the linear parameters, and thus allow "
                "specifying the mean (linear_par_mu) and covariance matrix "
                "(linear_par_Lambda) of this Gaussian prior. For example, "
                "to set this to a broad prior, pass in, e.g.: "
                "linear_par_Lambda=np.diag([1e2, 1e2])**2. The default mean "
                "is assumed to be zero.")
            raise ValueError(msg)

        self.linear_par_Lambda = np.array(linear_par_Lambda)
        self.linear_par_mu = np.array(linear_par_mu)

        if self.linear_par_Lambda.shape != (_n_linear, _n_linear):
            raise ValueError("Linear parameter prior variance must have shape "
                             "({0}, {0})".format(_n_linear))

        if self.linear_par_mu.shape != (_n_linear, ):
            raise ValueError("Linear parameter prior mean must have shape "
                             "({0}, )".format(_n_linear))

        # validate the input jitter specification
        if jitter is None:
            jitter = 0 * u.km/u.s

        if isiterable(jitter):
            if len(jitter) != 2:
                raise ValueError("If specifying parameters for the jitter "
                                 "prior, you must pass in a length-2 container "
                                 "containing the mean and standard deviation "
                                 "of the Gaussian over log(jitter^2)")

            if jitter_unit is None or not isinstance(jitter_unit, u.UnitBase):
                raise TypeError("If specifying parameters for the jitter "
                                "prior, you must also specify the units of the "
                                "jitter for evaluating the prior as an "
                                "astropy.units.UnitBase instance.")

            self._fixed_jitter = False
            self._jitter_unit = jitter_unit
            self.jitter = jitter

        else:
            self._fixed_jitter = True
            self._jitter_unit = jitter.unit
            self.jitter = jitter

    @property
    def num_params(self):
        n = len(self.default_params)
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
