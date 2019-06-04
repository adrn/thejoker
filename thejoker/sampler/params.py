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
    linear_par_Vinv : array_like, optional
        Inverse variance matrix that specifies the Gaussian prior on the linear
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
        ...                    jitter=5.*u.m/u.s) # fix jitter to 5 m/s
        >>> pars = JokerParams(P_min=8*u.day, P_max=8192*u.day,
        ...                    jitter=(1., 2.), jitter_unit=u.m/u.s) # specify jitter prior

    """
    @u.quantity_input(P_min=u.day, P_max=u.day)
    def __init__(self, P_min, P_max,
                 jitter=None, jitter_unit=None,
                 poly_trend=1, linear_par_Vinv=None,
                 anomaly_tol=1E-10, anomaly_maxiter=128):

        # the names of the default parameters
        self.default_params = ['P', 'M0', 'e', 'omega', 'jitter', 'K']

        self.poly_trend = int(poly_trend)
        self.default_params += ['v{0}'.format(i)
                                for i in range(self.poly_trend)]

        self.P_min = P_min
        self.P_max = P_max
        self.anomaly_tol = float(anomaly_tol)
        self.anomaly_maxiter = int(anomaly_maxiter)

        # K, then the linear parameters
        _n_linear = 1 + self.poly_trend
        if linear_par_Vinv is None:
            self.linear_par_Vinv = 1e-10 * np.eye(_n_linear)
        else:
            self.linear_par_Vinv = np.array(linear_par_Vinv)
            if self.linear_par_Vinv.shape != (_n_linear, _n_linear):
                raise ValueError("Linear parameter inverse variance prior must "
                                 "have shape ({0}, {0})".format(_n_linear))

        # validate the input jitter specification
        if jitter is None:
            jitter = 0 * u.km/u.s

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
            self._jitter_unit = jitter.unit
            self.jitter = jitter

    @property
    def num_params(self):
        n = len(self.default_params)

        if self._fixed_jitter:
            n -= 1

        return n

    @property
    def num_params_marginalized(self):
        n = 5

        if self._fixed_jitter:
            n -= 1

        return n
