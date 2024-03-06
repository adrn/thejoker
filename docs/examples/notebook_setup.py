# from IPython.core.getipython import get_ipython

# ipy = get_ipython()
# if ipy is not None:
#     ipy.magic('config InlineBackend.figure_format = "retina"')

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# logger = logging.getLogger("pytensor.gof.compilelock")
# logger.setLevel(logging.ERROR)


plt.style.use("default")
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "serif"
