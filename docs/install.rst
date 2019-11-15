************
Installation
************

We recommend installing |thejoker| into a new `Anaconda environment
<http://conda.pydata.org/docs/using/envs.html>`_.

pip
===

You can install the latest release of The Joker using ``pip``:

.. code-block:: bash

    pip install thejoker

Or, to install the latest development version, you can also use pip with the
GitHub project link:

.. code-block:: bash

    pip install git+https://github.com/adrn/thejoker.git

From source
===========

Alternatively, you can clone the repo (or download a zip of the latest code from
the `GitHub page <https://github.com/adrn/thejoker>`_) and use the provided
`environment.yml <https://github.com/adrn/thejoker/>`_ file to create a new
environment for |thejoker| that is set up with all of the dependencies
installed. To install in a new ``conda`` environment, change to the top-level
directory of the cloned repository, and run:

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
- pytables
- exoplanet
- pymc3
- `schwimmbad <https://github.com/adrn/schwimmbad>`_
- `twobody <https://github.com/adrn/TwoBody>`_

Optional Dependencies
---------------------

- matplotlib
