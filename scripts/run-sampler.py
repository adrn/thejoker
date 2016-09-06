# Standard library
from collections import OrderedDict
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import numpy as np

# Project
from thejoker import Paths
paths = Paths(__file__)
from thejoker.data import RVData
from thejoker.util import quantity_from_hdf5
from thejoker.units import usys
from thejoker.celestialmechanics import SimulatedRVOrbit
from thejoker.pool import choose_pool
from thejoker.sampler import get_good_samples, samples_to_orbital_params

# jitter = 0.5*u.km/u.s # TODO: set this same as Troup
jitter = 0.*u.km/u.s # Troup doesn't actually add any jitter!

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
