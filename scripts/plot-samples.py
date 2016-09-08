# Standard library
import os

# Third-party
from astropy import log as logger
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Project
from thejoker import Paths
paths = Paths()
from thejoker.data import RVData
from thejoker.util import quantity_from_hdf5
from thejoker.celestialmechanics import OrbitalParams
from thejoker.plot import plot_rv_curves, plot_corner

plt.style.use('../thejoker/thejoker.mplstyle')

def main(data_file, hdf5_key, cache_filename):

    # Read the orbital parameters!
    samples_filename = "{}.h5".format(os.path.splitext(cache_filename)[0])
    _path,_basename = os.path.split(samples_filename)
    if not _path:
        samples_filename = os.path.join(paths.root, "cache", samples_filename)
    logger.debug("Reading orbital parameter samples from '{}'".format(samples_filename))

    name = os.path.splitext(_basename)[0]
    plot_path = os.path.join(paths.plots, name)
    logger.info("Saving plots to: {}".format(plot_path))
    os.makedirs(plot_path, exist_ok=True)

    with h5py.File(samples_filename, 'r') as g:
        jitter = g.attrs['jitter_m/s']*u.m/u.s
        P_min = g.attrs['P_min_day']*u.day
        P_max = g.attrs['P_max_day']*u.day

        # read the orbital parameters
        opars = OrbitalParams.from_hdf5(g)

    # ------------------------------------------------------------------------
    # Read the data
    full_path = os.path.abspath(data_file)
    logger.debug("Reading data from input file at '{}'".format(full_path))
    with h5py.File(full_path, 'r') as f:
        if hdf5_key is not None:
            g = f[hdf5_key]
        else:
            g = f

        bmjd = g['mjd'][:]
        rv = quantity_from_hdf5(g, 'rv')
        rv_err = quantity_from_hdf5(g, 'rv_err')
    data = RVData(bmjd, rv, stddev=rv_err)
    data_jitter = RVData(bmjd, rv, stddev=np.sqrt(rv_err**2 + jitter**2))

    # ------------------------------------------------------------------------
    # plot RV curves
    dmjd = bmjd.max() - bmjd.min()
    t_grid = np.linspace(bmjd.min()-0.1*dmjd, bmjd.max()+0.1*dmjd, 1024)
    fig,ax = plt.subplots(1,1,figsize=(15,5))

    # HACK: where to set the units to plot in?
    rv_unit = u.km/u.s

    # UNDER-plot the data with jitter error-bars
    data_jitter.plot(ax=ax, rv_unit=rv_unit,
                     marker=None, ecolor='#de2d26', alpha=0.5)

    # HACK: where to set the number of lines to plot?
    plot_rv_curves(opars[:512], t_grid, rv_unit=rv_unit,
                   data=data, ax=ax)

    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_path, 'rv-curves.png'), dpi=300)

    # ------------------------------------------------------------------------
    # Corner plot!
    fig = plot_corner(opars, alpha=0.1) # TODO: kwargs??
    fig.suptitle(name, fontsize=26)
    fig.savefig(os.path.join(plot_path, 'corner.png'), dpi=300)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0, dest='quietness')

    parser.add_argument("--data-file", dest="data_file", default=None, required=True,
                        type=str, help="Path to HDF5 data file to analyze.")
    parser.add_argument("--hdf5-key", dest="hdf5_key", default=None,
                        type=str, help="Path within the HDF5 data file.")

    parser.add_argument("--cache-file", dest="cache_name", default=None,
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

    main(data_file=args.data_file, hdf5_key=args.hdf5_key, cache_filename=args.cache_name)
