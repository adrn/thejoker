# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import numpy as np
import six
from schwimmbad import choose_pool

# Project
from thejoker import Paths
paths = Paths()
from thejoker.data import RVData
from thejoker.units import default_units
# from thejoker.pool import choose_pool
from thejoker.celestialmechanics import OrbitalParams
from thejoker.sampler import get_good_samples, samples_to_orbital_params, sample_prior
from thejoker import config

def main(data_file, pool, tmp_prior_filename, n_samples=1, seed=42, hdf5_key=None,
         cache_filename=None, overwrite=False, continue_sampling=False,
         hyperpars_strs=dict(), log_jitter2_mean=0., log_jitter2_std=3.):

    full_path = os.path.abspath(data_file)
    if cache_filename is None:
        if hdf5_key is None:
            basename = os.path.basename(full_path)
            name = os.path.splitext(basename)[0]
            output_filename = os.path.join(paths.root, "cache", "{}.h5".format(name))
        else:
            output_filename = os.path.join(paths.root, "cache", "{}.h5".format(hdf5_key))

    else:
        output_filename = "{}.h5".format(os.path.splitext(cache_filename)[0])
        _path,_basename = os.path.split(output_filename)
        if not _path:
            output_filename = os.path.join(paths.root, "cache", output_filename)

    # container for hyper-parameter values
    hyperpars = dict(fixed_jitter=None, P_min=None, P_max=None)

    # see if we already did this:
    if os.path.exists(output_filename):
        if not overwrite and not continue_sampling:
            logger.info("Sampling already performed. Use --overwrite to redo or --continue "
                        "to keep sampling.")
            pool.close()
            sys.exit(0)

        elif continue_sampling: # we need to increment the random number seed appropriately
            mode = 'a' # append to output file

            with h5py.File(output_filename, 'r') as f:
                if 'rerun' not in f.attrs:
                    rerun = 0
                else:
                    rerun = f.attrs['rerun'] + 1

                if rerun != 0:
                    logger.debug("Reading hyperparameters from cache file.")

                    # HACK: this should be better
                    for name,unit in zip(['fixed_jitter', 'P_min', 'P_max'],
                                         [u.m/u.s, u.day, u.day]):
                        if np.isnan(f.attrs[name]):
                            hyperpars[name] = None
                        else:
                            hyperpars[name] = f.attrs[name] * unit

        elif overwrite: # restart rerun counter
            rerun = 0
            mode = 'w'

        else:
            raise ValueError("Unknown state!")

    else:
        mode = 'w'
        rerun = 0

    # get defaults for hyper-parameters:
    for name in hyperpars.keys():
        if hyperpars[name] is None:
            if hyperpars_strs[name] is None:
                hyperpars[name] = config.defaults[name]
            else:
                if isinstance(hyperpars_strs[name], six.string_types):
                    val,unit = str(hyperpars_strs[name]).split()
                    hyperpars[name] = float(val) * u.Unit(unit)
                else:
                    hyperpars[name] = float(hyperpars_strs[name])

        logger.debug("{}: {}".format(name, hyperpars[name]))

    # do this after choosing pool so all processes don't have same seed
    if seed is not None:
        if rerun > 0:
            logger.info("This is rerun {} -- incrementing random number seed.".format(rerun))

        logger.debug("random number seed: {} (+ rerun: {})".format(seed, rerun))
        seed = int(str(seed) + str(rerun))
        np.random.seed(seed)

    # only accepts HDF5 data formats with units
    logger.debug("Reading data from input file at '{}'".format(full_path))
    with h5py.File(full_path, 'r') as f:
        if hdf5_key is not None:
            data = RVData.from_hdf5(f[hdf5_key])
        else:
            data = RVData.from_hdf5(f)

    logger.debug("Number of prior samples: {}".format(n_samples))
    prior_samples = sample_prior(n_samples, P_min=hyperpars['P_min'], P_max=hyperpars['P_max'],
                                 log_jitter2_mean=log_jitter2_mean, log_jitter2_std=log_jitter2_std)
    P = prior_samples['P'].to(default_units['P']).value
    ecc = prior_samples['ecc']
    phi0 = prior_samples['phi0'].to(default_units['phi0']).value
    omega = prior_samples['omega'].to(default_units['omega']).value
    jitter2 = prior_samples['jitter2'].to(default_units['jitter']**2).value

    # if fixing jitter, set all values of jitter samples to the fixed value
    if hyperpars['fixed_jitter'] is not None:
        jitter2 = np.ones_like(jitter2) * hyperpars['fixed_jitter'].to(default_units['jitter']).value**2

    # pack the nonlinear parameters into an array
    nonlinear_p = np.vstack((P, phi0, ecc, omega, jitter2)).T
    # Note: the linear parameters are (v0, K)

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
    with h5py.File(output_filename, mode) as f:
        f.attrs['rerun'] = rerun
        if hyperpars['fixed_jitter'] is not None:
            f.attrs['fixed_jitter'] = hyperpars['fixed_jitter']
        else:
            f.attrs['fixed_jitter'] = np.nan
        f.attrs['P_min'] = hyperpars['P_min']
        f.attrs['P_max'] = hyperpars['P_max']

        for i,(name,unit) in enumerate(OrbitalParams._name_to_unit.items()):
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

    parser.add_argument("-s", "--seed", dest="seed", default=None, type=int,
                        help="Random number seed")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-f", "--file", dest="data_file", default=None, required=True,
                        type=str, help="Path to HDF5 data file to analyze.")
    parser.add_argument("--hdf5-key", dest="hdf5_key", default=None,
                        type=str, help="Path within an HDF5 file to the data.")
    parser.add_argument("--name", dest="cache_name", default=None,
                        type=str, help="Name to use when saving the cache file.")
    parser.add_argument("-n", "--num-samples", dest="n_samples", default=2**20,
                        type=str, help="Number of prior samples.")

    parser.add_argument("--fixed-jitter", dest="fixed_jitter", default=None, type=str,
                        help="Extra uncertainty to add in quadtrature to the RV measurement "
                             "uncertainties. Must specify a number with units, e.g., '15 m/s'")
    parser.add_argument("--Pmin", dest="P_min", default=None, type=str,
                        help="Minimum period to generate samples to (default: {})."
                             "Must specify a number with units, e.g., '2 day'"
                             .format(config.defaults['P_min']))
    parser.add_argument("--Pmax", dest="P_max", default=None, type=str,
                        help="Maximum period to generate samples to (default: {})."
                             "Must specify a number with units, e.g., '8192 day'"
                             .format(config.defaults['P_max']))

    parser.add_argument("--log-jitter2-mean", dest="log_jitter2_mean", default=0.,
                        help="TODO")
    parser.add_argument("--log-jitter2-std", dest="log_jitter2_std", default=3.,
                        help="TODO")

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
        main(data_file=args.data_file, pool=pool, n_samples=n_samples, hdf5_key=args.hdf5_key,
             seed=args.seed, overwrite=args.overwrite, continue_sampling=args._continue,
             cache_filename=args.cache_name, tmp_prior_filename=fp.name,
             hyperpars_strs=dict(fixed_jitter=args.fixed_jitter, P_min=args.P_min, P_max=args.P_max),
             log_jitter2_mean=args.log_jitter2_mean, log_jitter2_std=args.log_jitter2_std)
