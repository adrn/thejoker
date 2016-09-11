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
        exp1_data = RVData.from_hdf5(f)
        exp1_opars = OrbitalParams.unpack(f['truth_vector'][:])
        exp1_orbit = exp1_opars.rv_orbit()

    # ------------------------------------------------------------------------
    # Generate Experiment 5 data

    data_t = exp1_data._t + exp1_data.t_offset
    data_std = 1 / np.sqrt(exp1_data._ivar)[0]

    with h5py.File(os.path.join(paths.root, "data", "experiment5.h5"), "w") as f:
        for i,next_t in enumerate([data_t[-1]+14, data_t[-1]+250.]): # MAGIC NUMBERS
            new_rv = exp1_orbit.generate_rv_curve(next_t)[0].to(usys['speed']).value
            new_rv = np.random.normal(new_rv, data_std)

            exp5_data = RVData(t=np.concatenate((data_t, [next_t])),
                               rv=np.concatenate((exp1_data._rv, [new_rv])) * usys['speed'],
                               ivar=np.concatenate((exp1_data._ivar, [1/data_std**2])) * usys['speed'])

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
