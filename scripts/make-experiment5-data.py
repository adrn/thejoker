"""

Generate simulated data for Experiment 5:
    - Read in data from Experiment 1
    - Move 2nd data point through 1 cycle in period to show how the sampling depends on when, in
      phase, it is observed

"""

# Standard library
import os

# Third-party
import astropy.units as u
import h5py
import numpy as np

# Project
from thejoker import Paths
paths = Paths()
from thejoker.celestialmechanics import OrbitalParams
from thejoker.data import RVData
from thejoker.units import usys

def main():

    experiment1_file = os.path.join(paths.root, "data", "experiment1.h5")
    if not os.path.exists(experiment1_file):
        raise IOError("Experiment 1 data file could not be found -- did you "
                      "run make-experiment1-data.py yet?")

    with h5py.File(experiment1_file, "r") as f:
        exp1_opars = OrbitalParams.unpack(f['truth_vector'][:])
        exp1_orbit = exp1_opars.rv_orbit()

    # ------------------------------------------------------------------------
    # Generate Experiment 5 data

    data_t1 = np.random.uniform(2., 3*365, size=4) + 55555.
    data_t2 = np.exp(np.random.uniform(np.log(2.), np.log(3*365), size=4)) + 55555.

    with h5py.File(os.path.join(paths.root, "data", "experiment5.h5"), "w") as f:
        for i,t in enumerate([data_t1, data_t2]):
            rv = exp1_orbit.generate_rv_curve(t).to(usys['speed'])
            rv_err = (np.random.uniform(100,300,size=t.size)*u.m/u.s).to(usys['speed'])
            rv = np.random.normal(rv, rv_err)

            exp5_data = RVData(t=t, rv=rv, stddev=rv_err)

            g = f.create_group(str(i))
            exp5_data.to_hdf5(g)
            g.create_dataset('truth_vector', data=exp1_opars.pack())

if __name__ == '__main__':
    from argparse import ArgumentParser

    # Define parser object
    parser = ArgumentParser(description="")

    parser.add_argument("-s", "--seed", dest="seed", default=None,
                        type=int, help="Random number seed.")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    main()
