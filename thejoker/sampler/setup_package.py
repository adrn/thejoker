from __future__ import absolute_import
from distutils.core import Extension
from astropy_helpers import setup_helpers

def get_extensions():
    exts = []

    from os import path
    import twobody
    twobody_path = path.dirname(twobody.__file__)

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(twobody_path)
    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('thejoker/sampler/fast_likelihood.pyx')
    # cfg['sources'].append('twobody/celestial/src/twobody.c')
    exts.append(Extension('thejoker.sampler.fast_likelihood', **cfg))

    return exts

# def get_package_data():
#     return {'twobody.celestial': ['src/*.h', 'src/*.c', '*.pyx', '*.pxd']}
