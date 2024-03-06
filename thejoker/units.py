"""Originally from the exoplanet project"""

__all__ = ["with_unit", "has_unit", "to_unit"]

from pytensor.tensor import as_tensor_variable

UNIT_ATTR_NAME = "__tensor_unit__"


def with_unit(obj, unit):
    """Decorate a tensor with Astropy units

    Parameters
    ----------
    obj
        The tensor object
    unit : astropy.units.Unit
        The units for this object

    """
    if hasattr(obj, UNIT_ATTR_NAME):
        msg = f"{obj!r} already has units"
        raise TypeError(msg)
    obj = as_tensor_variable(obj)
    setattr(obj, UNIT_ATTR_NAME, unit)
    return obj


def has_unit(obj):
    """Does an object have units as defined by exoplanet?"""
    return hasattr(obj, UNIT_ATTR_NAME)


def to_unit(obj, target):
    """Convert a Theano tensor with units to a target set of units

    Parameters
    ----------
    obj
        The Theano tensor
    target : astropy.units.Unit
        The target units

    Returns
    -------
        A tensor in the right units

    """
    if not has_unit(obj):
        return obj
    base = getattr(obj, UNIT_ATTR_NAME)
    return obj * base.to(target)
