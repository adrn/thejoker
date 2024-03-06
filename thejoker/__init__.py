from ._version import version as __version__
from .data import RVData
from .plot import plot_phase_fold, plot_rv_curves
from .prior import JokerPrior
from .samples import JokerSamples
from .samples_analysis import (
    MAP_sample,
    is_P_Kmodal,
    is_P_unimodal,
    max_phase_gap,
    periods_spanned,
    phase_coverage,
    phase_coverage_per_period,
)
from .thejoker import TheJoker

__all__ = [
    "__version__",
    "TheJoker",
    "RVData",
    "JokerSamples",
    "JokerPrior",
    "plot_rv_curves",
    "plot_phase_fold",
    "MAP_sample",
    "is_P_unimodal",
    "is_P_Kmodal",
    "max_phase_gap",
    "phase_coverage",
    "periods_spanned",
    "phase_coverage_per_period",
]


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
