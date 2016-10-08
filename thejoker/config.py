import astropy.units as u

defaults = {
    'P_min': 16. * u.day, # MAGIC NUMBER,
    'P_max': 8192. * u.day, # MAGIC NUMBER
    'fixed_jitter': None, # by default, infer it
    'M_min': 128 # MAGIC NUMBER
}
