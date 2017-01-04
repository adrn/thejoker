************
Installation
************

From source
===========

To install from source, you'll need to clone the repository or download a zip of
the latest code from the `GitHub page <https://github.com/adrn/thejoker>`_.

We recommend installing The Joker into a new `Anaconda environment
<http://conda.pydata.org/docs/using/envs.html>`_. You can use the provided
`environment.yml <https://github.com/adrn/thejoker/>`_ file to create a new
environment for The Joker that is set up with all of the dependencies installed.
In the top-level directory of the cloned repository, do

.. code-block:: bash

    conda env create

When this finishes, activate the environment

.. code-block:: bash

    source activate thejoker

The project is installable using the standard

.. code-block:: bash

    python setup.py install

Dependencies
============

- numpy
- scipy
- astropy
- emcee

Optional Dependencies
---------------------

- matplotlib
- h5py
