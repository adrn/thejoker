"""
Generate samples from the prior over the non-linear parameters of the Kepler
problem. Save the samples to an HDF5 file.
"""

# Standard library
import os

# Third-party
from astropy import log as logger
import h5py
import numpy as np

# Project
from thejoker import Paths
paths = Paths(__file__)

def main(n_samples, seed, overwrite=False):

    if os.path.exists(paths.prior_samples) and not overwrite:
        with h5py.File(paths.prior_samples, 'r') as f:
            n_samples = len(f['P'])

        raise IOError("File already exists with {} prior samples. Use '--overwrite' "
                      "to overwrite this file.".format(n_samples))

    # sample from priors in nonlinear parameters
    P_min = 16. # day - MAGIC NUMBER
    P_max = 8192. # day - MAGIC NUMBER
    P = np.exp(np.random.uniform(np.log(P_min), np.log(P_max), size=n_samples))
    phi0 = np.random.uniform(0, 2*np.pi, size=n_samples)

    # MAGIC NUMBERS below: Kipping et al. 2013 (MNRAS 434 L51)
    ecc = np.random.beta(a=0.867, b=3.03, size=n_samples)
    omega = np.random.uniform(0, 2*np.pi, size=n_samples)

    with h5py.File(paths.prior_samples, 'w') as f:
        d = f.create_dataset('P', data=P)
        d.attrs['unit'] = 'day'

        d = f.create_dataset('phi0', data=phi0)
        d.attrs['unit'] = 'radian'

        d = f.create_dataset('ecc', data=ecc)

        d = f.create_dataset('omega', data=omega)
        d.attrs['unit'] = 'radian'

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true", help="Overwrite any existing data.")
    parser.add_argument("--seed", dest="seed", default=42, type=int,
                        help="Random number seed")
    parser.add_argument("-n", "--num-samples", dest="n_samples", default=2**25, # ~1GB
                        type=str, help="Number of prior samples to generate.")

    args = parser.parse_args()

    np.random.seed(args.seed)

    try:
        n_samples = int(args.n_samples)
    except:
        n_samples = int(eval(args.n_samples)) # LOL what's security?

    main(n_samples=n_samples, seed=args.seed, overwrite=args.overwrite)

