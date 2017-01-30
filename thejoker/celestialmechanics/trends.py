# Third-party
import astropy.units as u
import numpy as np

class VelocityTrend(object):
    pass

class PolynomialVelocityTrend(VelocityTrend):
    """
    Represents a long-term velocity trend to the radial velocity data.

    This can either be used to (1) specify parameters in
    `~thejoker.sampler.params.JokerParams`, in which case it is only a
    placeholder class and should just be instantiated with the number
    of coefficients in the polynomial, ``n_terms``, or (2) as a way to
    evaluate a polynomial trend at times, in which case the coefficient
    values should be passed in, ``coeffs``.

    Parameters
    ----------
    n_terms : int (optional, see note)
        The number of terms in the polynomial.
    coeffs : iterable (optional, see note)
        The coefficients of the polynomial. The power is the index of
        the coefficient, so the 0th coefficient is the constant, the
        1st coefficient is the linear term, etc.

    TODO
    ----
    - We could add support for a reference epoch (other than mjd=0) in here...

    """
    def __init__(self, n_terms=None, coeffs=None):

        if n_terms is None and coeffs is None:
            raise ValueError("You must specify either n_terms or coeffs.")

        elif n_terms is not None and coeffs is not None:
            raise ValueError("You must specify either n_terms or coeffs, "
                             "not both.")

        if coeffs is not None:
            self.coeffs = list(coeffs)

            _unit = u.km/u.s
            for coeff in self.coeffs:
                if not hasattr(coeff, 'unit'):
                    raise ValueError("Input coefficients must be a Quantity with "
                                     "velocity per time^i units!")
                elif not coeff.unit.is_equivalent(_unit):
                    raise u.UnitsError("Input coefficients must have velocity per "
                                       "time^i units!")

                _unit = _unit / u.day

        else:
            self.coeffs = None

        if n_terms is None and coeffs is not None:
            self.n_terms = len(coeffs)
        else:
            self.n_terms = int(n_terms)

    def __call__(self, t):
        if self.coeffs is None:
            raise ValueError("To evaluate the trend, you must have supplied coefficient "
                             "values at creation.")
        t = np.atleast_1d(t)

        if not hasattr(t, 'unit'): # assume bare array has units = day
            t = t*u.day

        if t.unit.physical_type != 'time':
            raise u.UnitsError("Input time(s) must be a Quantity with time units!")

        A = np.vander(t.to(u.day).value, N=self.n_terms, increasing=True)
        return sum([A[:,i]*u.day**i * self.coeffs[i] for i in range(A.shape[1])])

class PiecewisePolynomialVelocityTrend(VelocityTrend):
    # TODO:
    # This class can also represent different, independent sections of the
    # data to handle, e.g., calibration offsets between epochs. See the
    # ``data_mask`` argument documentation below for more info.
    pass