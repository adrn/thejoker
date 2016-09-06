# Standard library
from collections import OrderedDict
import os
import sys

# Third-party
from astropy import log as logger
from astropy.io import fits
import astropy.units as u
import corner
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Project
from thejoker import Paths
paths = Paths(__file__)
from thejoker.data import RVData
from thejoker.sampler import tensor_vector_scalar, marginal_ln_likelihood
from thejoker.util import quantity_from_hdf5
from thejoker.units import usys
from thejoker.celestialmechanics import SimulatedRVOrbit
from thejoker.config import P_min, P_max
from thejoker.pool import choose_pool

plt.style.use('../thejoker/thejoker.mplstyle')

# jitter = 0.5*u.km/u.s # TODO: set this same as Troup
jitter = 0.*u.km/u.s # Troup doesn't actually add any jitter!

def marginal_ll_worker(task):
    nl_p_chunk, data = task
    n_chunk = len(nl_p_chunk)

    ll = np.zeros(n_chunk)
    for i in range(n_chunk):
        try:
            ATA,p,chi2 = tensor_vector_scalar(nl_p_chunk[i], data)
            ll[i] = marginal_ln_likelihood(ATA, chi2)
        except:
            ll[i] = np.nan

    return ll

def get_good_samples(nonlinear_p, data, pool, chunk_size):
    n_total = len(nonlinear_p)
    tasks = [[nonlinear_p[i*chunk_size:(i+1)*chunk_size], data]
             for i in range(n_total//chunk_size+1)]

    results = pool.map(marginal_ll_worker, tasks)
    marg_ll = np.concatenate(results)

    assert len(marg_ll) == len(nonlinear_p)

    uu = np.random.uniform(size=len(nonlinear_p))
    good_samples_bool = uu < np.exp(marg_ll - marg_ll.max())
    good_samples = nonlinear_p[np.where(good_samples_bool)]
    n_good = len(good_samples)
    logger.info("{} good samples".format(n_good))

    return good_samples

def samples_to_orbital_params_worker(task):
    nl_p_chunk, data = task
    n_chunk = len(nl_p_chunk)

    pars = np.zeros((n_chunk, 6))
    for i in range(n_chunk):
        nl_p = nl_p_chunk[i]
        P, phi0, ecc, omega = nl_p
        ATA,p,_ = tensor_vector_scalar(nl_p, data)

        cov = np.linalg.inv(ATA)
        v0,asini = np.random.multivariate_normal(p, cov)

        if asini < 0:
            # logger.warning("Swapping asini")
            asini = np.abs(asini)
            omega += np.pi

        pars[i] = [P, asini, ecc, omega, phi0, v0]

    return pars

def samples_to_orbital_params(nonlinear_p, data, pool, chunk_size):
    n_total = len(nonlinear_p)
    tasks = [[nonlinear_p[i*chunk_size:(i+1)*chunk_size], data]
             for i in range(n_total//chunk_size+1)]
    orbit_pars = pool.map(samples_to_orbital_params_worker, tasks)
    orbit_pars = np.concatenate(orbit_pars)
    return orbit_pars.reshape(-1, orbit_pars.shape[-1])

def main(APOGEE_ID, pool_kwargs, n_samples=1, seed=42, overwrite=False):

    pool = choose_pool(**pool_kwargs)

    if pool.size > 0:
        # try chunking by the pool size
        chunk_size = n_samples // pool.size
    else:
        chunk_size = 1

    output_filename = os.path.join(paths.root, "cache", "{}.h5".format(APOGEE_ID))

    # load data from APOGEE data
    logger.debug("Reading data from Troup allVisit file...")
    all_data = RVData.from_apogee(paths.troup_allVisit, apogee_id=APOGEE_ID)
    troup_data = np.genfromtxt(os.path.join(paths.root, "data", "troup16-dr12.csv"),
                               delimiter=",", names=True, dtype=None)
    troup_data = troup_data[troup_data['APOGEE_ID'].astype(str) == APOGEE_ID]
    # HACK: in above, only troup stars accepted for now

    # Troup's parameters
    troup_P = np.random.normal(troup_data['PERIOD'], troup_data['PERIOD_ERR'], size=1024)*u.day
    troup_ecc = np.random.normal(troup_data['ECC'], troup_data['ECC_ERR'], size=1024)
    troup_K = np.random.normal(troup_data['SEMIAMP'], troup_data['SEMIAMP_ERR'], size=1024)*u.m/u.s
    _,troup_asini = SimulatedRVOrbit.P_K_ecc_to_mf_asini_ecc(troup_P, troup_K, troup_ecc)

    # HACK: add extra scatter to velocities
    data_var = all_data.stddev**2 + jitter**2
    all_data._ivar = (1 / data_var).to(all_data.ivar.unit).value

    # read prior samples from cached file of samples
    with h5py.File(paths.prior_samples) as f:
        P = quantity_from_hdf5(f, 'P', n_samples).decompose(usys).value
        ecc = quantity_from_hdf5(f, 'ecc', n_samples)
        phi0 = quantity_from_hdf5(f, 'phi0', n_samples).decompose(usys).value
        omega = quantity_from_hdf5(f, 'omega', n_samples).decompose(usys).value

    # pack the nonlinear parameters into an array
    nonlinear_p = np.vstack((P, phi0, ecc, omega)).T
    # Note: the linear parameters are (v0, asini)

    logger.debug("Number of prior samples: {}".format(n_samples))

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
            logger.debug("- not completed - running sampler...")

            nl_samples = get_good_samples(nonlinear_p, data, pool, chunk_size) # TODO: save?
            if len(nl_samples) == 0:
                logger.error("Failed to find any good samples!")
                pool.close()
                sys.exit(1)
            orbital_params = samples_to_orbital_params(nl_samples, data, pool, chunk_size)

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
                    g.create_dataset(name, data=orbital_params.T[i])
                    if unit is not None:
                        g[name].attrs['unit'] = str(unit)
        else:
            logger.debug("- sampling already completed")

        # --------------------------------------------------------------------
        # make some plots, yo
        MAX_N_LINES = 128

        # a time grid to plot RV curves of the model
        t_grid = np.linspace(all_data._t.min()-50, all_data._t.max()+50, 1024)

        # plot samples
        fig = plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(2, 2)

        ax_rv = plt.subplot(gs[0,:])
        # ax_rv.set_prop_cycle(cycler('color', _palette))

        ax_lnP_e = plt.subplot(gs[1,0])
        ax_lnP_asini = plt.subplot(gs[1,1])

        lnP_lim = [np.log(P_min) - 0.1, np.log(P_max) + 0.1]
        e_lim = [-0.05, 1.05]
        lnasini_lim = [-6, 2]

        with h5py.File(output_filename, 'r') as f:
            g = f[str(n_delete)]

            P = quantity_from_hdf5(g, 'P')
            asini = quantity_from_hdf5(g, 'asini')
            ecc = quantity_from_hdf5(g, 'ecc')
            omega = quantity_from_hdf5(g, 'omega')
            phi0 = quantity_from_hdf5(g, 'phi0')
            v0 = quantity_from_hdf5(g, 'v0')

            logger.debug("{} good samples".format(len(P)))

            # the number of lines to plot is at most 128, but may be fewer if we don't have
            #   enough good samples
            n_lines = min(len(P), MAX_N_LINES)

            # plot all of the points
            n_pts = len(P)

            # scale the transparency of the lines, points based on these klugy functions
            pt_alpha = min(1., 1. + 0.9*(np.log(2)-np.log(n_pts))/(np.log(200000)-np.log(2)))
            Q = 4. # HACK
            line_alpha = 0.05 + Q / (n_lines + Q)
            pt_style = dict(marker='.', color='#555555', alpha=pt_alpha, ms=2, ls='none')
            troup_pt_style = pt_style.copy()
            troup_pt_style['color'] = '#de2d26'
            troup_pt_style['alpha'] = 0.4

            ax_lnP_e.plot(np.log(P.to(u.day).value), ecc, **pt_style)
            ax_lnP_asini.plot(np.log(P.to(u.day).value), np.log(asini.to(u.au).value),
                              **pt_style)

            ax_lnP_e.plot(np.log(troup_P.to(u.day).value), troup_ecc, **troup_pt_style)
            ax_lnP_asini.plot(np.log(troup_P.to(u.day).value), np.log(troup_asini.to(u.au).value),
                              **troup_pt_style)

            # plot Troup's parameters

            # corner.hist2d(np.log(P.to(u.day).value), ecc, ax=ax_lnP_e,
            #               range=[lnP_lim, e_lim], bins=32,
            #               fill_contours=True, plot_density=False)
            # corner.hist2d(np.log(P.to(u.day).value), np.log(asini.to(u.au).value),
            #               ax=ax_lnP_asini, range=[lnP_lim, lnasini_lim], bins=32,
            #               fill_contours=True, plot_density=False)

            # plot orbits over the data
            for i in range(len(P)):
                orbit = SimulatedRVOrbit(P=P[i], a_sin_i=asini[i], ecc=ecc[i],
                                         omega=omega[i], phi0=phi0[i], v0=v0[[i]])
                model_rv = orbit.generate_rv_curve(t_grid).to(u.km/u.s).value
                ax_rv.plot(t_grid, model_rv, linestyle='-', marker=None,
                           alpha=line_alpha, color='#555555') #color='#3182bd')

                if i >= MAX_N_LINES:
                    break

        data.plot(ax=ax_rv, markersize=3)
        ax_rv.set_xlim(t_grid.min()-25, t_grid.max()+25)
        _rv = all_data.rv.to(u.km/u.s).value
        ax_rv.set_ylim(np.median(_rv)-20, np.median(_rv)+20)
        ax_rv.set_xlabel('MJD')
        ax_rv.set_ylabel('RV [km s$^{-1}$]')

        ax_lnP_e.set_xlim(*lnP_lim) # HACK
        ax_lnP_e.set_ylim(*e_lim)
        ax_lnP_e.set_xlabel(r'$\ln P$')
        ax_lnP_e.set_ylabel(r'$e$')

        ax_lnP_asini.set_xlim(*lnP_lim)
        ax_lnP_asini.set_ylim(*lnasini_lim)
        ax_lnP_asini.set_xlabel(r'$\ln P$')
        ax_lnP_asini.set_ylabel(r'$\ln (a \sin i)$')

        fig.tight_layout()

        os.makedirs(os.path.join(paths.plots, APOGEE_ID), exist_ok=True)
        fig.savefig(os.path.join(paths.plots, APOGEE_ID, 'delete-{:02d}.png'.format(n_delete)),
                    dpi=300)

        # fig = corner.corner(np.hstack((np.log(nl_p[:,0:1]), nl_p[:,1:])),
        #                     labels=['$\ln P$', r'$\phi_0$', '$e$', r'$\omega$'])
        # plt.savefig(join(PLOT_PATH, 'corner-leave_out_{}.png'.format(leave_out)))
        plt.close('all')

    pool.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

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
    if args.verbosity != 0:
        if args.verbosity == 1:
            logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            logger.setLevel(logging.WARNING)
        else: # anything >= 2
            logger.setLevel(logging.ERROR)

    else: # default
        logger.setLevel(logging.INFO)

    np.random.seed(args.seed)

    try:
        n_samples = int(args.n_samples)
    except:
        n_samples = int(eval(args.n_samples)) # LOL what's security?

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)

    main(APOGEE_ID=args.apogee_id, n_samples=n_samples, pool_kwargs=pool_kwargs,
         seed=args.seed, overwrite=args.overwrite)
