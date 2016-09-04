"""
Generate samples from the prior over the non-linear parameters of the Kepler
problem. Save the samples to an HDF5 file.
"""

# Standard library
from collections import OrderedDict
import os
from os.path import abspath, join, split, exists
import time
import sys

# Third-party
from astropy import log as logger
from astropy.io import fits
import astropy.table as tbl
import astropy.coordinates as coord
import astropy.units as u
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from cycler import cycler
import numpy as np
import corner
from gala.util import get_pool

# Project
from ebak import SimulatedRVOrbit
from ebak.singleline.data import RVData
from ebak.sampler import tensor_vector_scalar, marginal_ln_likelihood
from ebak.units import usys

_basepath = split(abspath(join(__file__, "..")))[0]
if not exists(join("..", "scripts")):
    raise IOError("You must run this script from inside the scripts directory:\n{}"
                  .format(join(_basepath, "scripts")))

# HACKS: Hard-set paths
ALLVISIT_DATA_PATH = join(_basepath, "data", "allVisit-l30e.2.fits")
PLOT_PATH = join(_basepath, "plots", "sampler")
CACHE_PATH = join(_basepath, "cache")
for PATH in [PLOT_PATH, CACHE_PATH]:
    if not exists(PATH):
        os.mkdir(PATH)

P_min = 16. # day
P_max = 8192. # day
jitter = 0.5*u.km/u.s # TODO: set this same as Troup

# _palette = ['r', 'g', 'b']

def marginal_ll_worker(task):
    nl_p, data = task
    try:
        ATA,p,chi2 = tensor_vector_scalar(nl_p, data)
        ll = marginal_ln_likelihood(ATA, chi2)
    except:
        ll = np.nan
    return ll

def get_good_samples(nonlinear_p, data, pool):
    tasks = [[nonlinear_p[i], data] for i in range(len(nonlinear_p))]
    results = pool.map(marginal_ll_worker, tasks)
    marg_ll = np.squeeze(results)

    uu = np.random.uniform(size=len(nonlinear_p))
    good_samples_bool = uu < np.exp(marg_ll - marg_ll.max())
    good_samples = nonlinear_p[np.where(good_samples_bool)]
    n_good = len(good_samples)
    logger.info("{} good samples".format(n_good))

    return good_samples

def samples_to_orbital_params_worker(task):
    nl_p, data = task
    P, phi0, ecc, omega = nl_p

    ATA,p,_ = tensor_vector_scalar(nl_p, data)

    cov = np.linalg.inv(ATA)
    v0,asini = np.random.multivariate_normal(p, cov)

    if asini < 0:
        # logger.warning("Swapping asini")
        asini = np.abs(asini)
        omega += np.pi

    return [P, asini, ecc, omega, phi0, -v0] # TODO: sign of v0?

def samples_to_orbital_params(nonlinear_p, data, pool):
    tasks = [[nonlinear_p[i], data] for i in range(len(nonlinear_p))]
    orbit_pars = pool.map(samples_to_orbital_params_worker, tasks)
    return np.array(orbit_pars).T

def _getq(f, key):
    if 'unit' in f[key].attrs and f[key].attrs['unit'] is not None:
        unit = u.Unit(f[key].attrs['unit'])
    else:
        unit = 1.
    return f[key][:] * unit

def main(APOGEE_ID, pool, n_samples=1, seed=42, overwrite=False):

    output_filename = join(CACHE_PATH, "{}.h5".format(APOGEE_ID))

    # MPI shite
    # pool = get_pool(mpi=mpi, threads=n_procs)
    # need load-balancing - see: https://groups.google.com/forum/#!msg/mpi4py/OJG5eZ2f-Pg/EnhN06Ozg2oJ

    # load data from APOGEE data
    logger.debug("Reading data from Troup catalog and allVisit files...")
    all_data = RVData.from_apogee(ALLVISIT_DATA_PATH, apogee_id=APOGEE_ID)

    # HACK: add extra jitter to velocities
    data_var = all_data.stddev**2 + jitter**2
    all_data._ivar = (1 / data_var).to(all_data.ivar.unit).value

    # a time grid to plot RV curves of the model - used way later
    t_grid = np.linspace(all_data._t.min()-50, all_data._t.max()+50, 1024)

    # TODO: break this out into a separate "make-prior-samples.py" script:
    # sample from priors in nonlinear parameters
    P = np.exp(np.random.uniform(np.log(P_min), np.log(P_max), size=n_samples))
    phi0 = np.random.uniform(0, 2*np.pi, size=n_samples)
    # MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
    ecc = np.random.beta(a=0.867, b=3.03, size=n_samples)
    omega = np.random.uniform(0, 2*np.pi, size=n_samples)
