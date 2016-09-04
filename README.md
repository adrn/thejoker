# The Joker [YO-ker] /'joʊkər/

![badge-img](https://img.shields.io/badge/Made%20at-%23AstroHackWeek-8063d5.svg?style=flat)

A custom Monte Carlo sampler for the two-body problem.

## Authors

- Adrian Price-Whelan (Princeton)
- David W. Hogg (NYU, MPIA)
- Dan Foreman-Mackey (UW)

## Installation

The project is installable with

```bash
python setup.py install
```

We recommend installing into a new [Anaconda
environment](http://conda.pydata.org/docs/using/envs.html). You can use the `environment.yml` file
to create a new environment for The Joker with all dependencies installed. In the top-level
directory of the cloned repository, do

```bash
conda env create
```

When this finishes, activate the environment as usual and then install

```bash
source activate thejoker
python setup.py install
```

## License

Copyright 2016 the Authors. Licensed under the terms of the MIT License (see LICENSE).
