from distutils.core import Extension
from collections import defaultdict


def get_extensions():
    exts = []

    from os import path
    try:
        # Get twobody path, or ignore failure -- for egg_info build
        import twobody
        twobody_path = path.dirname(twobody.__file__)
    except ImportError:
        twobody_path = None

    cfg = defaultdict(list)
    cfg['include_dirs'].append('numpy')

    if twobody_path is not None:
        cfg['include_dirs'].append(twobody_path)
        cfg['sources'].append(path.join(twobody_path,
                                        'src/twobody.c'))

    cfg['extra_compile_args'].append('--std=gnu99')
    cfg['sources'].append('thejoker/src/fast_likelihood.pyx')
    exts.append(Extension('thejoker.src.fast_likelihood', **cfg))

    return exts


def get_package_data():
    return {'thejoker.src': ['fast_likelihood.pyx']}
