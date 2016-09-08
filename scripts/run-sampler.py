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
from thejoker.pool import choose_pool
from thejoker.sampler import get_good_samples, samples_to_orbital_params, sample_prior

# jitter = 0.5*u.km/u.s # TODO: set this same as Troup
jitter = 0.*u.km/u.s # Troup doesn't actually add any jitter!

def main(data_file, pool, tmp_prior_filename, n_samples=1, seed=42, cache_filename=None,
         overwrite=False, continue_sampling=False):

    full_path = os.path.abspath(data_file)
    if cache_filename is None:
        basename = os.path.basename(full_path)
        name = os.path.splitext(basename)[0]
        output_filename = os.path.join(paths.root, "cache", "{}.h5".format(name))

    else:
        output_filename = "{}.h5".format(os.path.splitext(cache_filename)[0])
        _path,_basename = os.path.split(output_filename)
        if not _path:
            output_filename = os.path.join(paths.root, "cache", output_filename)

    # see if we already did this:
    if os.path.exists(output_filename):
        if not overwrite and not continue_sampling:
            logger.info("Sampling already performed. Use --overwrite to redo or --continue "
                        "to keep sampling.")
            pool.close()
            sys.exit(0)

        elif continue_sampling: # we need to increment the random number seed appropriately
            with h5py.File(output_filename, 'r') as f:
                if 'rerun' not in f.attrs:
                    rerun = 0
                else:
                    rerun = f.attrs['rerun'] + 1

        elif overwrite: # restart rerun counter
            rerun = 0

        else:
            raise ValueError("Unknown state!")

    else:
        rerun = 0

    # do this after choosing pool so all processes don't have same seed
    if seed is not None:
        if rerun > 0:
            logger.info("This is rerun {} -- incrementing random number seed.".format(rerun))

        logger.debug("random number seed: {} (+ rerun: {})".format(seed, rerun))
        np.random.seed(seed + rerun)

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

    # cache the prior samples
    with h5py.File(tmp_prior_filename, "w") as f:
        f.create_dataset('samples', data=nonlinear_p)

    logger.debug("Running sampler...")

    good_samples_idx = get_good_samples(n_samples, tmp_prior_filename, data, pool)
    nl_samples = nonlinear_p[good_samples_idx]
    if len(nl_samples) == 0:
        logger.error("Failed to find any good samples!")
        pool.close()
        sys.exit(0)

    # compute orbital parameters for all good samples
    orbital_params = samples_to_orbital_params(good_samples_idx, tmp_prior_filename,
                                               data, pool, seed)

    # save the orbital parameters out to a cache file
    par_spec = OrderedDict()
    par_spec['P'] = usys['time']
    par_spec['asini'] = usys['length']
    par_spec['ecc'] = None
    par_spec['omega'] = usys['angle']
    par_spec['phi0'] = usys['angle']
    par_spec['v0'] = usys['length']/usys['time']

    with h5py.File(output_filename, 'a') as f:
        f.attrs['rerun'] = rerun

        for i,(name,unit) in enumerate(par_spec.items()):
            if name in f:
                if overwrite: # delete old samples and overwrite
                    del f[name]
                    f.create_dataset(name, data=orbital_params.T[i])

                elif continue_sampling: # append to existing samples
                    _data = f[name][:]
                    del f[name]
                    f[name] = np.concatenate((_data, orbital_params.T[i]))
            else:
                f.create_dataset(name, data=orbital_params.T[i])

            if unit is not None: # note: could get in to a weird state with mismatched units...
                f[name].attrs['unit'] = str(unit)

    pool.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging
    import tempfile

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
    parser.add_argument("--name", dest="cache_name", default=None,
                        type=str, help="Name to use when saving the cache file.")
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
    pool = choose_pool(**pool_kwargs)

    # use a context manager so the prior samples file always gets deleted
    with tempfile.NamedTemporaryFile(dir=os.path.join(paths.root, "cache")) as fp:
        main(data_file=args.data_file, pool=pool, n_samples=n_samples,
             seed=args.seed, overwrite=args.overwrite, continue_sampling=args._continue,
             cache_filename=args.cache_name, tmp_prior_filename=fp.name)
