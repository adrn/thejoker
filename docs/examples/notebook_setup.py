import pytest
pytest.skip()

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

import logging
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)


plt.style.use("default")
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Liberation Serif"]
plt.rcParams["font.cursive"] = ["Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
