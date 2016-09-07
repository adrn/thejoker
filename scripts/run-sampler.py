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
from thejoker.sampler import get_good_samples, samples_to_orbital_params, sample_prior

# jitter = 0.5*u.km/u.s # TODO: set this same as Troup
jitter = 0.*u.km/u.s # Troup doesn't actually add any jitter!

def main(data_file, pool_kwargs, n_samples=1, seed=42, overwrite=False, continue_sampling=False):

    pool = choose_pool(**pool_kwargs)

    # do this after choosing pool so all processes don't have same seed
    if args.seed is not None:
        logger.debug("random number seed: {}".format(args.seed))
        np.random.seed(args.seed)

    if pool.size > 0:
        # try chunking by the pool size
        chunk_size = n_samples // pool.size
    else:
        chunk_size = 1

    full_path = os.path.abspath(data_file)
    basename = os.path.basename(full_path)
    name = os.path.splitext(basename)[0]
    output_filename = os.path.join(paths.root, "cache", "{}.h5".format(name))

    # see if we already did this:
    if os.path.exists(output_filename) and not overwrite and not continue_sampling:
        logger.info("Sampling already performed. Use --overwrite to redo or --continue "
                    "to keep sampling.")
        pool.close()
        sys.exit(0)

    # only accepts HDF5 data formats with units
    logger.debug("Reading data from input file at '{}'".format(full_path))
    with h5py.File(full_path, 'r') as f:
        bmjd = f['mjd'][:]
        rv = quantity_from_hdf5(f, 'rv')
        rv_err = quantity_from_hdf5(f, 'rv_err')
    data = RVData(bmjd, rv, stddev=rv_err)

    # generate prior samples on the fly
    logger.debug("Number of prior samples: {}".format(n_samples))
    prior_samples = sample_prior(n_samples)
    P = prior_samples['P'].decompose(usys).value
    ecc = prior_samples['ecc']
    phi0 = prior_samples['phi0'].decompose(usys).value
    omega = prior_samples['omega'].decompose(usys).value

    # pack the nonlinear parameters into an array
    nonlinear_p = np.vstack((P, phi0, ecc, omega)).T
    # Note: the linear parameters are (v0, asini)

    logger.debug("Running sampler...")

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

    with h5py.File(output_filename, 'a') as f:
        for i,(name,unit) in enumerate(par_spec.items()):
            if name in f:
                if overwrite: # delete old samples and overwrite
                    del f[name]
                    f.create_dataset(name, data=orbital_params.T[i])

                elif continue_sampling: # append to existing samples
                    _data = f[name][:]
                    del f[name]
                    f[name] = np.concatenate((_data, orbital_params.T[i]))

            if unit is not None: # note: could get in to a weird state with mismatched units...
                f[name].attrs['unit'] = str(unit)

    pool.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    oc_group = parser.add_mutually_exclusive_group()
    oc_group.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                          action="store_true", help="Overwrite any existing data.")
    oc_group.add_argument("-c", "--continue", dest="_continue", default=False,
                          action="store_true", help="Continue the sampler.")

    parser.add_argument("--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-f", "--file", dest="data_file", default=None, required=True,
                        type=str, help="Path to HDF5 data file to analyze.")
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

    try:
        n_samples = int(args.n_samples)
    except:
        n_samples = int(eval(args.n_samples)) # LOL what's security?

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)

    main(data_file=args.data_file, n_samples=n_samples, pool_kwargs=pool_kwargs,
         seed=args.seed, overwrite=args.overwrite, continue_sampling=args._continue)
