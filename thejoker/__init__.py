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

__bibtex__ = __citation__ = """@ARTICLE{thejoker,
       author = {{Price-Whelan}, Adrian M. and {Hogg}, David W. and
         {Foreman-Mackey}, Daniel and {Rix}, Hans-Walter},
        title = "{The Joker: A Custom Monte Carlo Sampler for Binary-star and Exoplanet Radial Veloc\
ity Data}",
      journal = {\apj},
     keywords = {binaries: spectroscopic, methods: data analysis, methods: statistical, planets and \
satellites: fundamental parameters, surveys, techniques: radial velocities, Astrophysics - Solar and\
 Stellar Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
         year = "2017",
        month = "Mar",
       volume = {837},
       number = {1},
          eid = {20},
        pages = {20},
          doi = {10.3847/1538-4357/aa5e50},
archivePrefix = {arXiv},
       eprint = {1610.07602},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2017ApJ...837...20P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
