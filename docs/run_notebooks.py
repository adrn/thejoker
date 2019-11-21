#!/usr/bin/env python

# Standard library
import glob
import os
import sys

# Third-party
import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


def process_notebook(filename, kernel_name=None):
    path = os.path.join(
        os.path.abspath("theano_cache"), "p{0}".format(os.getpid())
    )
    os.makedirs(path, exist_ok=True)
    os.environ["THEANO_FLAGS"] = "base_compiledir={0}".format(path)

    with open(filename) as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1, kernel_name=kernel_name)

    print("running: {0}".format(filename))
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

    for filename in glob.glob(pattern):
        process_notebook(filename, kernel_name=nbsphinx_kernel_name)
