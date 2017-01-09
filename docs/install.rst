*******
Install
*******

From source
===========

You can install The Joker using pip by doing:

.. code-block:: bash

    pip install git+https://github.com/adrn/thejoker.git

We recommend installing The Joker into a new `Anaconda environment
<http://conda.pydata.org/docs/using/envs.html>`_.

Alternatively, you can clone the repo (or download a zip of the latest code from
the `GitHub page <https://github.com/adrn/thejoker>`_) and use the provided
`environment.yml <https://github.com/adrn/thejoker/>`_ file to create a new
environment for The Joker that is set up with all of the dependencies installed.
To install in a new ``conda`` environment, change to the top-level directory of
the cloned repository, and run:

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
- h5py
- emcee
- schwimmbad

Optional Dependencies
---------------------

- matplotlib
