"""
Note: copied out of gala, https://github.com/adrn/gala
"""

from collections import OrderedDict

# Third-party
import astropy.units as u

__all__ = ['default_units']

default_units = OrderedDict()
default_units['P'] = u.day
default_units['K'] = u.m/u.s
default_units['ecc'] = u.one
default_units['omega'] = u.radian
default_units['phi0'] = u.radian
default_units['v0'] = u.m/u.s
default_units['jitter'] = u.m/u.s
default_units['asini'] = u.au
default_units['mf'] = u.Msun

default_plot_units = OrderedDict()
default_plot_units['P'] = u.day
default_plot_units['K'] = u.m/u.s
default_plot_units['ecc'] = u.one
default_plot_units['omega'] = u.degree
default_plot_units['phi0'] = u.degree
default_plot_units['v0'] = u.km/u.s
default_plot_units['jitter'] = u.m/u.s
