from __future__ import division, print_function

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np

# Project
from ..data import RVData

class TheJoker(object):

    """
    A custom Monte-Carlo sampler for two-body systems.

    Parameters
    ----------
    data : `~thejoker.data.RVData`
        The radial velocity data.
    pool : ``schwimmbad.BasePool`` (optional)
        A processing pool (default is a ``schwimmbad.SerialPool`` instance).
    random_state : `numpy.random.RandomState` (optional)
        A ``RandomState`` instance to serve as a parent for the random
        number generators. See the :ref:`random-numbers` page for more
        information.
    **hyper_pars
        All other kwargs are used to set the hyper-parameter values.
        These can be:

            - ``P_min`` : `astropy.units.Quantity`
                Lower bound on prior over period (smallest period considered).
            - ``P_max`` : `astropy.units.Quantity`
                Upper bound on prior over period (largest period considered).
            - ``jitter`` : `astropy.units.Quantity` (optional)
                Fixed value of jitter. Set to ``None`` or don't specify to
                infer the jitter, but see next two parameters.
            - ``log_jitter2_prior_mean`` : numeric (option)
            - ``log_jitter2_prior_stddev`` : numeric (option)

    TODO
    ----
    - Delete data when pickling / serializing?

    """
    def __init__(self, data, pool=None, random_state=None, **hyper_pars):

        # validate input data
        if not isinstance(data, RVData):
            raise TypeError("Input data must be an RVData instance, not '{}'"
                            .format(type(data)))
        self.data = data

        # set the processing pool
        if pool is None:
            import schwimmbad
            pool = schwimmbad.SerialPool()

        elif not hasattr(pool, 'map') or not hasattr(pool, 'close'):
            raise TypeError("Input pool object must have .map() and .close() "
                            "methods. We recommend using `schwimmbad` pools.")

        self.pool = pool

        # set the parent random state - child processes get different states based on the parent
        if random_state is None:
            random_state = np.random.RandomState()

        elif not isinstance(random_state, np.random.RandomState):
            raise TypeError("Random state object must be a numpy RandomState instance, "
                            "not '{}'".format(type(random_state)))

        self.random_state = random_state

        # TODO: validate hyper-parameters
