"""

Generate simulated data for Experiment 5:
    - Read in data from Experiment 1
    - Move 2nd data point through 1 cycle in period to show how the sampling depends on when, in
      phase, it is observed

"""

# Standard library
import os

# Third-party
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

# Project
from thejoker import Paths
paths = Paths()
from thejoker.celestialmechanics import OrbitalParams
from thejoker.data import RVData
from thejoker.units import usys

plt.style.use("../thejoker/thejoker.mplstyle")

def main():

    experiment1_file = os.path.join(paths.root, "data", "experiment1.h5")
    if not os.path.exists(experiment1_file):
        raise IOError("Experiment 1 data file could not be found -- did you "
                      "run make-experiment1-data.py yet?")

    with h5py.File(experiment1_file, "r") as f:
        exp1_data = RVData.from_hdf5(f)
        exp1_opars = OrbitalParams.unpack(f['truth_vector'][:])
        exp1_orbit = exp1_opars.rv_orbit()

    # number of steps to take the data point through a cycle
    n_steps = 8
    idx = len(exp1_data._t) - 1 # the index of the data point to vary

    # ------------------------------------------------------------------------
    # Generate Experiment 5 data

    exp5_data = exp1_data.copy()
    with h5py.File(os.path.join(paths.root, "data", "experiment5.h5"), "w") as f:
        for i in range(n_steps):
            t1 = exp1_data._t[idx] + (i+1) * exp1_opars._P[0] / (n_steps+1)
            exp5_data._t[idx] = t1

            _rv = exp1_orbit.generate_rv_curve(t1)[0]
            stddev = 1/np.sqrt(exp5_data._ivar[idx])
            exp5_data._rv[idx] = np.random.normal(_rv.decompose(usys).value, stddev)

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
