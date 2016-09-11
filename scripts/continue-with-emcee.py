# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import emcee
import h5py
import numpy as np

# Project
from thejoker import Paths
paths = Paths()
from thejoker import config
from thejoker.data import RVData
from thejoker.units import usys
from thejoker.pool import choose_pool
from thejoker.sampler import mcmc
from thejoker.celestialmechanics import OrbitalParams

def main(data_file, cache_filename, pool, n_steps, overwrite=False, seed=42, hdf5_key=None):

    full_path = os.path.abspath(data_file)
    output_filename = "{}.h5".format(os.path.splitext(cache_filename)[0])
    _path,_basename = os.path.split(output_filename)
    if not _path:
        output_filename = os.path.join(paths.root, "cache", output_filename)

    # make sure The Joker has already been run
    if not os.path.exists(output_filename):
        raise IOError("The Joker cache file '{}' can't be found! Are you sure you ran "
                      "'run-sampler.py'?".format(output_filename))

    with h5py.File(output_filename, 'a') as f:
        jitter = f.attrs['jitter_m/s'] * u.m/u.s

        if 'emcee' in f:
            if overwrite:
                del f['emcee']

            else:
                logger.info("emcee already run on this sample file.")
                pool.close()
                sys.exit(0)

    # do this after choosing pool so all processes don't have same seed
    if seed is not None:
        logger.debug("random number seed: {}".format(seed))
        np.random.seed(seed)

    # only accepts HDF5 data formats with units
    logger.debug("Reading data from input file at '{}'".format(full_path))
    with h5py.File(full_path, 'r') as f:
        if hdf5_key is not None:
            data = RVData.from_hdf5(f[hdf5_key])
        else:
            data = RVData.from_hdf5(f)
    data.add_jitter(jitter)

    logger.debug("Reading cached sample(s) from file at '{}'".format(output_filename))
    opars = OrbitalParams.from_hdf5(output_filename)

    # Fire up emcee
    if len(opars) > 1:
        P = np.median(opars.P.decompose(usys).value)
        T = data._t.max() - data._t.min()
        Delta = 4*P**2 / (2*np.pi*T)
        P_rms = np.std(opars.P.decompose(usys).value)
        logger.debug("Period rms for surviving samples: {}, Delta: {}".format(P_rms, Delta))

        if P_rms > Delta:
            logger.error("Period rms > ∆! Re-run The Joker instead - your posterior pdf is "
                         "probably multi-modal")
            pool.close()
            sys.exit(0)

    else:
        logger.debug("Only one surviving sample.")

    # transform samples to the parameters we'll sample using emcee
    samples = opars.pack()
    samples_trans = mcmc.pack_mcmc(samples.T)
    j_max = np.argmax([mcmc.ln_posterior(s, data) for s in samples_trans.T])
    p0 = samples_trans[:,j_max]

    n_walkers = config.defaults['M_min']
    p0 = emcee.utils.sample_ball(p0, 1E-5*np.abs(p0), size=n_walkers)

    sampler = emcee.EnsembleSampler(n_walkers, p0.shape[1],
                                    lnpostfn=mcmc.ln_posterior, args=(data,),
                                    pool=pool)

    pos,prob,state = sampler.run_mcmc(p0, n_steps) # MAGIC NUMBER

    pool.close()

    emcee_samples = mcmc.unpack_mcmc(pos.T)
    with h5py.File(output_filename, 'a') as f:
        g = f.create_group('emcee')

        for i,(name,phystype) in enumerate(OrbitalParams._name_phystype.items()):
            g.create_dataset(name, data=emcee_samples[i])

            if phystype is not None: # note: could get in to a weird state with mismatched units...
                g[name].attrs['unit'] = str(usys[phystype])

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

    # emcee
    parser.add_argument("--nsteps", dest="n_steps", required=True, type=int,
                        help="Number of steps to take.")

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

    pool = choose_pool(mpi=args.mpi, processes=args.n_procs)

    # use a context manager so the prior samples file always gets deleted
    main(data_file=args.data_file, cache_filename=args.cache_name, pool=pool,
         n_steps=args.n_steps, hdf5_key=args.hdf5_key, seed=args.seed, overwrite=args.overwrite)
