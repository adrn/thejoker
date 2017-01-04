The Joker [YO-ker] /'joʊkər/
============================

.. image:: https://readthedocs.org/projects/thejoker/badge/?version=latest
        :target: http://thejoker.readthedocs.io/
.. image:: http://img.shields.io/travis/adrn/thejoker/master.svg?style=flat
        :target: http://travis-ci.org/adrn/thejoker
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
        :target: https://github.com/adrn/thejoker/blob/master/LICENSE
.. image:: http://img.shields.io/badge/arXiv-1610.07602-orange.svg?style=flat
        :target: https://arxiv.org/abs/1610.07602

A custom Monte Carlo sampler for the two-body problem.

Authors
-------

- Adrian Price-Whelan (Princeton)
- David W. Hogg (NYU, MPIA)
- Dan Foreman-Mackey (UW)

Installation
------------

The project is installable with

.. code-block:: bash

    python setup.py install

We recommend installing into a new [Anaconda
environment](http://conda.pydata.org/docs/using/envs.html). You can use the `environment.yml` file
to create a new environment for The Joker with all dependencies installed: In the top-level
directory of the cloned repository, do

.. code-block:: bash

    conda env create

When this finishes, activate the environment as usual and then install

.. code-block:: bash

    source activate thejoker
    python setup.py install

