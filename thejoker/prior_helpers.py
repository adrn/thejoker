# Third-party
import astropy.units as u

__all__ = ['get_nonlinear_equiv_units',
           'validate_poly_trend',
           'get_linear_equiv_units',
           'validate_n_offsets',
           'get_v0_offsets_equiv_units']


# Nonlinear parameter helpers:
def get_nonlinear_equiv_units():
    return {
        'P': u.day,
        'e': u.one,
        'omega': u.radian,
        'M0': u.radian,
        's': u.m/u.s,
    }


# Linear parameter helpers:
def validate_poly_trend(poly_trend):
    try:
        poly_trend = int(poly_trend)
    except Exception:
        raise ValueError("poly_trend must be an integer that specifies the "
                         "number of polynomial (in time) trend terms to "
                         "include in The Joker.")
    vtrend_names = ['v{0}'.format(i) for i in range(poly_trend)]
    return poly_trend, vtrend_names


def get_linear_equiv_units(poly_trend):
    poly_trend, v_names = validate_poly_trend(poly_trend)
    return {
        'K': u.m/u.s,
        **{name: u.m/u.s/u.day**i for i, name in enumerate(v_names)}
    }


def validate_sigma_v(sigma_v, poly_trend, v_names):
    if isinstance(sigma_v, u.Quantity):
        if not sigma_v.isscalar:
            raise ValueError("You must pass in a scalar value for sigma_v if "
                             "passing in a single quantity.")
        sigma_v = {'v0': sigma_v}

    if hasattr(sigma_v, 'keys'):
        for name in v_names:
            if name not in sigma_v.keys():
                raise ValueError(
                    "If specifying the standard-deviations of the polynomial "
                    "trend parameter prior, you must pass in values for all "
                    f"parameter names. Expected keys: {v_names}, received: "
                    f"{sigma_v.keys()}")
        return sigma_v

    try:
        if len(sigma_v) != poly_trend:
            raise ValueError(
                "You must pass in a single sigma value for each velocity trend "
                f"parameter: You passed in {len(sigma_v)} values, but "
                f"poly_trend={poly_trend}")
        sigma_v = {name: val for name, val in zip(v_names, sigma_v)}

    except TypeError:
        raise TypeError("Invalid input for velocity trend prior sigma "
                        "values. This must either be a scalar Quantity (if "
                        "poly_trend=1) or an iterable of Quantity objects "
                        "(if poly_trend>1)")

    return sigma_v


# Offsets in v0 helpers:
def validate_n_offsets(n_offsets):
    try:
        n_offsets = int(n_offsets)
    except Exception:
        raise ValueError("n_offsets must be an integer that specifies the "
                         "number of v0 offset parameters to include in "
                         "The Joker. These parameters allow passing in data "
                         "from multiple surveys that may have unknown "
                         "calibration offsets.")
    offset_names = ['dv0_{0}'.format(i) for i in range(1, n_offsets+1)]
    return n_offsets, offset_names


def get_v0_offsets_equiv_units(n_offsets):
    n_offsets, names = validate_n_offsets(n_offsets)
    return {name: u.m/u.s for i, name in enumerate(names)}
