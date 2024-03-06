#!/usr/bin/env python

# Standard library
import glob
import logging
import os
import sys


def process_notebook(filename, kernel_name=None):
    import nbformat
    from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

    path = os.path.join(os.path.abspath("cache"), f"p{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={path}"

    with open(filename) as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=-1, kernel_name=kernel_name)
    ep.log.setLevel(logging.INFO)
    ep.log.addHandler(logging.StreamHandler())

    success = True
    try:
        ep.preprocess(notebook, {"metadata": {"path": "examples/"}})
    except CellExecutionError as e:
        msg = f"Error while running: {filename}\n\n"
        msg += e.traceback
        print(msg)
        success = False
    finally:
        with open(os.path.join(filename), mode="w") as f:
            nbformat.write(notebook, f)

    return success


if __name__ == "__main__":
    if len(sys.argv) == 2:
        pattern = sys.argv[1]

    else:
        pattern = "examples/*.ipynb"

    nbsphinx_kernel_name = os.environ.get("NBSPHINX_KERNEL", "python3")

    for filename in sorted(glob.glob(pattern)):
        success = process_notebook(filename, kernel_name=nbsphinx_kernel_name)
        if not success:
            sys.exit(1)
