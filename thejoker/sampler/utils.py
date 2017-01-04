from __future__ import division, print_function

def get_ivar(data, s2):
    return data.ivar.value / (1 + s2 * data.ivar.value)
