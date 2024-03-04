import os
from collections import defaultdict

from setuptools import Extension, setup

exts = []

import numpy as np

try:
    import twobody
except ImportError:
    msg = "The twobody package is required to install TheJoker. "
    raise ImportError(msg)

cfg = defaultdict(list)
cfg["include_dirs"].append(np.get_include())

twobody_path = os.path.dirname(twobody.__file__)
cfg["include_dirs"].append(twobody_path)
cfg["sources"].append(os.path.join(twobody_path, "src/twobody.c"))

cfg["extra_compile_args"].append("--std=gnu99")
cfg["sources"].append("thejoker/src/fast_likelihood.pyx")
exts.append(Extension("thejoker.src.fast_likelihood", **cfg))

setup(ext_modules=exts)
