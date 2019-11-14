from ._astropy_init import *

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

__minimum_python_version__ = "3.6"

class UnsupportedPythonError(Exception):
    pass

if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("thejoker does not support Python < {}"
                                 .format(__minimum_python_version__))

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .thejoker import TheJoker
    from .data import RVData
    from .samples import JokerSamples
    from .prior import JokerPrior
