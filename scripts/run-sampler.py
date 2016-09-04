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

    # sample from priors in nonlinear parameters
    P = np.exp(np.random.uniform(np.log(P_min), np.log(P_max), size=n_samples))
    phi0 = np.random.uniform(0, 2*np.pi, size=n_samples)
    # MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
    ecc = np.random.beta(a=0.867, b=3.03, size=n_samples)
    omega = np.random.uniform(0, 2*np.pi, size=n_samples)

    # pack the nonlinear parameters into an array
    nl_p = np.vstack((P, phi0, ecc, omega)).T
    # Note: the linear parameters are (v0, asini)

    logger.info("Number of prior samples: {}".format(n_samples))

    # TODO: we may want to "curate" which datapoints are saved...
    idx = np.random.permutation(len(all_data))
    for n_delete in range(len(all_data)):
        if n_delete == 0:
            data = all_data
        else:
            data = all_data[idx[:-n_delete]]
        logger.debug("Removing {}/{} data points".format(n_delete, len(all_data)))

        # see if we already did this:
        skip_compute = False
        with h5py.File(output_filename, 'a') as f:
            if str(n_delete) in f and not overwrite:
                skip_compute = True # skip if already did this one

            elif str(n_delete) in f and overwrite:
                del f[str(n_delete)]

        if not skip_compute:
            nl_samples = get_good_samples(nl_p, data, pool) # TODO: save?
            if len(nl_samples) == 0:
                logger.error("Failed to find any good samples!")
                pool.close()
                sys.exit(1)
            orbital_params = samples_to_orbital_params(nl_samples, data, pool)

            # save the orbital parameters out to a cache file
            par_spec = OrderedDict()
            par_spec['P'] = usys['time']
            par_spec['asini'] = usys['length']
            par_spec['ecc'] = None
            par_spec['omega'] = usys['angle']
            par_spec['phi0'] = usys['angle']
            par_spec['v0'] = usys['length']/usys['time']
            with h5py.File(output_filename, 'r+') as f:
                g = f.create_group(str(n_delete))

                for i,(name,unit) in enumerate(par_spec.items()):
                    g.create_dataset(name, data=orbital_params[i])
                    if unit is not None:
                        g[name].attrs['unit'] = str(unit)

        # --------------------------------------------------------------------
        # make some plots, yo
        MAX_N_LINES = 128

        # plot samples
        fig = plt.figure(figsize=(6,6))
        gs = gridspec.GridSpec(2, 2)

        ax_rv = plt.subplot(gs[0,:])
        # ax_rv.set_prop_cycle(cycler('color', _palette))

        ax_lnP_e = plt.subplot(gs[1,0])
        ax_lnP_asini = plt.subplot(gs[1,1])

        with h5py.File(output_filename, 'r') as f:
            g = f[str(n_delete)]

            P = _getq(g,'P')
            asini = _getq(g,'asini')
            ecc = _getq(g,'ecc')
            omega = _getq(g,'omega')
            phi0 = _getq(g,'phi0')
            v0 = _getq(g,'v0')

            n_lines = min(len(P), MAX_N_LINES)

            n_pts = len(P)
            pt_alpha = min(0.9, max(0.1, 0.8 + 0.9*(np.log(2)-np.log(n_pts))/(np.log(1024)-np.log(2))))
            Q = 4. # HACK
            line_alpha = 0.05 + Q / (n_lines + Q)

            ax_lnP_e.plot(np.log(P.to(u.day).value), ecc,
                          marker='.', color='k', alpha=pt_alpha, ms=5, ls='none')
            ax_lnP_asini.plot(np.log(P.to(u.day).value), np.log(asini.to(u.au).value),
                              marker='.', color='k', alpha=pt_alpha, ms=5, ls='none')

            for i in range(len(P)):
                orbit = SimulatedRVOrbit(P=P[i], a_sin_i=asini[i], ecc=ecc[i],
                                         omega=omega[i], phi0=phi0[i], v0=v0[[i]])
                model_rv = orbit.generate_rv_curve(t_grid).to(u.km/u.s).value
                ax_rv.plot(t_grid, model_rv, linestyle='-', marker=None,
                           alpha=line_alpha, color='#3182bd')

                if i >= MAX_N_LINES:
                    break

        data.plot(ax=ax_rv, markersize=3)
        ax_rv.set_xlim(t_grid.min()-25, t_grid.max()+25)
        _rv = all_data.rv.to(u.km/u.s).value
        ax_rv.set_ylim(np.median(_rv)-20, np.median(_rv)+20)
        ax_rv.set_xlabel('MJD')
        ax_rv.set_ylabel('RV [km s$^{-1}$]')

        ax_lnP_e.set_xlim(np.log(P_min) - 0.1, 6. + 0.1) # HACK
        ax_lnP_e.set_ylim(-0.1, 1.)
        ax_lnP_e.set_xlabel(r'$\ln P$')
        ax_lnP_e.set_ylabel(r'$e$')

        ax_lnP_asini.set_xlim(ax_lnP_e.get_xlim())
        ax_lnP_asini.set_ylim(-6, 0)
        ax_lnP_asini.set_xlabel(r'$\ln P$')
        ax_lnP_asini.set_ylabel(r'$\ln (a \sin i)$')

        fig.tight_layout()
        fig.savefig(join(PLOT_PATH, 'delete-{}.png'.format(n_delete)), dpi=300)

        # fig = corner.corner(np.hstack((np.log(nl_p[:,0:1]), nl_p[:,1:])),
        #                     labels=['$\ln P$', r'$\phi_0$', '$e$', r'$\omega$'])
        # plt.savefig(join(PLOT_PATH, 'corner-leave_out_{}.png'.format(leave_out)))
        plt.close('all')

    pool.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging
    from ebak.pool import Pool

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true", help="Overwrite any existing data.")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                        help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("--id", dest="apogee_id", default=None, required=True,
                        type=str, help="APOGEE ID")
    parser.add_argument("-n", "--num-samples", dest="n_samples", default=2**20,
                        type=str, help="Number of prior samples.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    np.random.seed(args.seed)
    if args.mpi:
        logger.info("Running with MPI")
        _kwargs = {'pool': 'MPIPool'}
    elif args.n_procs != 0:
        logger.info("Running with multiprocessing on {} cores".format(args.n_procs))
        _kwargs = {'pool': 'MultiPool', 'processes': args.n_procs}
    else:
        logger.info("Running serial")
        _kwargs = {'pool': 'SerialPool'}

    try:
        n_samples = int(args.n_samples)
    except:
        n_samples = int(eval(args.n_samples)) # LOL what's security?

    with Pool(**_kwargs) as pool:
        main(APOGEE_ID=args.apogee_id, n_samples=n_samples, pool=pool,
             seed=args.seed, overwrite=args.overwrite)
