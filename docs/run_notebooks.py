#!/usr/bin/env python

# Standard library
import glob
import os
import sys
import logging


def process_notebook(filename, kernel_name=None):
    import nbformat
    from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

    path = os.path.join(
        os.path.abspath("theano_cache"), "p{0}".format(os.getpid())
    )
    os.makedirs(path, exist_ok=True)
    os.environ["THEANO_FLAGS"] = "base_compiledir={0}".format(path)
    os.environ["AESARA_FLAGS"] = os.environ["THEANO_FLAGS"]

    with open(filename) as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1, kernel_name=kernel_name)
    ep.log.setLevel(logging.DEBUG)
    ep.log.addHandler(logging.StreamHandler())

    try:
        ep.preprocess(notebook, {"metadata": {"path": "examples/"}})
    except CellExecutionError as e:
        msg = "error while running: {0}\n\n".format(filename)
        msg += e.traceback
        print(msg)
    finally:
        with open(os.path.join(filename), mode="w") as f:
            nbformat.write(notebook, f)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        pattern = sys.argv[1]

    else:
        pattern = "examples/*.ipynb"

    nbsphinx_kernel_name = os.environ.get('NBSPHINX_KERNEL', 'python3')

    for filename in sorted(glob.glob(pattern)):
        process_notebook(filename, kernel_name=nbsphinx_kernel_name)
